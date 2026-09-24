#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combat_runtime import OBS_DIM, RuntimeConfig, load_policy_and_normalizer, sha256_file, write_json

DEFAULT_RESUME = "experiments/phase3v2/phase3v2_b_selected_artifacts/gvg_seed_v000.pt"
DEFAULT_NORMALIZER = "experiments/phase3v2/normalizers/phase3v2_a_obs45_no_pressure.pt"


def freeze_actor_heads_only(policy: torch.nn.Module) -> list[str]:
    for param in policy.parameters():
        param.requires_grad_(False)
    trainable = []
    for module_name in ("actor_cont_mean", "actor_binary_logits"):
        module = getattr(policy, module_name)
        for name, param in module.named_parameters():
            param.requires_grad_(True)
            trainable.append(f"{module_name}.{name}")
    return trainable


def load_original(root: Path) -> dict[str, np.ndarray]:
    paths = sorted(root.glob("job_*/teacher_shard_v2.npz"))
    if not paths:
        raise FileNotFoundError(f"no original teacher_shard_v2.npz files under {root}")
    fields = {key: [] for key in ("obs", "action", "shoot_intent", "shoot_mask", "aim_error", "fired", "hit")}
    side = []
    source_job = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        n = len(data["obs"])
        for key in fields:
            fields[key].append(data[key])
        side.append(np.asarray(["scripted"] * n, dtype=str))
        source_job.append(np.asarray([path.parent.name] * n, dtype=str))
    out = {key: np.concatenate(value, axis=0) for key, value in fields.items()}
    out["agent_side"] = np.concatenate(side, axis=0)
    out["sample_weight"] = np.ones(len(out["obs"]), dtype=np.float32)
    out["post_reset_flag"] = np.zeros(len(out["obs"]), dtype=np.float32)
    out["target_visible"] = np.zeros(len(out["obs"]), dtype=np.float32)
    out["damage_taken"] = np.zeros(len(out["obs"]), dtype=np.float32)
    out["source_kind"] = np.asarray(["original"] * len(out["obs"]), dtype=str)
    out["source_job"] = np.concatenate(source_job, axis=0)
    return out


def load_gvg(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out = {
        "obs": data["obs"].astype(np.float32),
        "action": data["action"].astype(np.float32),
        "shoot_intent": data["shoot_intent"].astype(np.float32),
        "shoot_mask": data["shoot_mask"].astype(np.float32),
        "aim_error": data["aim_error"].astype(np.float32),
        "fired": data["fired"].astype(np.float32),
        "hit": data["hit"].astype(np.float32),
        "agent_side": data["agent_side"].astype(str),
        "sample_weight": data["sample_weight"].astype(np.float32) if "sample_weight" in data.files else np.ones(len(data["obs"]), dtype=np.float32),
        "post_reset_flag": data["post_reset_flag"].astype(np.float32) if "post_reset_flag" in data.files else np.zeros(len(data["obs"]), dtype=np.float32),
        "damage_taken": data["damage_taken"].astype(np.float32) if "damage_taken" in data.files else np.zeros(len(data["obs"]), dtype=np.float32),
        "target_visible": data["target_visible"].astype(np.float32) if "target_visible" in data.files else np.zeros(len(data["obs"]), dtype=np.float32),
        "source_kind": np.asarray(["gvg"] * len(data["obs"]), dtype=str),
        "source_job": data["source_job"].astype(str) if "source_job" in data.files else np.asarray(["gvg"] * len(data["obs"]), dtype=str),
    }
    return out


def maybe_load_pressure(root: str) -> dict[str, np.ndarray] | None:
    if not root:
        return None
    path = Path(root)
    if not path.exists():
        return None
    try:
        data = load_original(path)
    except FileNotFoundError:
        return None
    data["source_kind"] = np.asarray(["pressure"] * len(data["obs"]), dtype=str)
    return data


def concat_data(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def build_sampling_weights(
    data: dict[str, np.ndarray],
    original_frac: float,
    gvg_frac: float,
    pressure_frac: float,
    side_b_mult: float,
    underfire_mult: float,
    postreset_mult: float,
    visible_nohit_mult: float,
) -> np.ndarray:
    kind = data["source_kind"].astype(str)
    weights = np.zeros(len(kind), dtype=np.float64)
    available = {
        "original": int(np.sum(kind == "original")),
        "gvg": int(np.sum(kind == "gvg")),
        "pressure": int(np.sum(kind == "pressure")),
    }
    desired = {"original": original_frac, "gvg": gvg_frac, "pressure": pressure_frac if available["pressure"] else 0.0}
    total_desired = sum(desired.values())
    desired = {key: value / max(total_desired, 1e-9) for key, value in desired.items()}
    for key, frac in desired.items():
        mask = kind == key
        if np.any(mask) and frac > 0:
            weights[mask] = frac / float(np.sum(mask))
    gvg = kind == "gvg"
    weights[gvg] *= data["sample_weight"][gvg].astype(np.float64)
    weights[gvg & (data["agent_side"].astype(str) == "B")] *= side_b_mult
    weights[gvg & (data["damage_taken"].astype(np.float32) > 0.0)] *= underfire_mult
    weights[gvg & (data["post_reset_flag"].astype(np.float32) > 0.5)] *= postreset_mult
    visible_nohit = (
        gvg
        & (data["target_visible"].astype(np.float32) > 0.5)
        & (data["hit"].astype(np.float32) < 0.5)
        & (data["aim_error"].astype(np.float32) <= 75.0)
    )
    weights[visible_nohit] *= visible_nohit_mult
    weights = np.maximum(weights, 1e-12)
    return weights / np.mean(weights)


def predict(policy, normalizer, obs: np.ndarray, device: torch.device, batch_size: int = 8192) -> torch.Tensor:
    policy.set_ppo_mode(training=False)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            xb = torch.as_tensor(obs[start : start + batch_size], dtype=torch.float32, device=device)
            xb = normalizer.normalize_tensor(xb).view(-1, 1, OBS_DIM)
            hidden = policy.init_hidden(xb.shape[0], device)
            chunks.append(policy.predict_actions(xb, hidden, return_logits=True).squeeze(1).cpu())
    return torch.cat(chunks, dim=0)


def eval_offline(
    policy,
    normalizer,
    obs: np.ndarray,
    action: np.ndarray,
    shoot: np.ndarray,
    mask: np.ndarray,
    init_pred: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    pred = predict(policy, normalizer, obs, device)
    target = torch.as_tensor(action, dtype=torch.float32)
    shoot_t = torch.as_tensor(shoot, dtype=torch.float32)
    mask_t = torch.as_tensor(mask, dtype=torch.float32)
    prob = torch.sigmoid(pred[:, 4])
    pred_pos = prob > 0.5
    valid = mask_t > 0.5
    pos = (shoot_t > 0.5) & valid
    neg = (shoot_t < 0.5) & valid
    tp = int((pred_pos & pos).sum().item())
    fp = int((pred_pos & neg).sum().item())
    fn = int(((~pred_pos) & pos).sum().item())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    yaw_mask = torch.abs(target[:, 2]) > 0.03
    pitch_mask = torch.abs(target[:, 3]) > 0.03
    yaw_acc = float((torch.sign(pred[:, 2]) == torch.sign(target[:, 2])).float()[yaw_mask].mean().item()) if torch.any(yaw_mask) else 1.0
    pitch_acc = float((torch.sign(pred[:, 3]) == torch.sign(target[:, 3])).float()[pitch_mask].mean().item()) if torch.any(pitch_mask) else 1.0
    look_huber = float(F.huber_loss(pred[:, 2:4], target[:, 2:4], delta=0.25).item())
    init_huber = float(F.huber_loss(init_pred[:, 2:4], target[:, 2:4], delta=0.25).item())
    improvement = (init_huber - look_huber) / max(init_huber, 1e-8)
    look_abs = pred[:, 2:4].abs()
    sat95 = float((look_abs > 0.95).float().mean().item())
    sat80 = float((look_abs > 0.80).float().mean().item())
    shoot_rate = float(pred_pos.float().mean().item())
    gate = bool(
        torch.isfinite(pred).all().item()
        and 0.10 <= shoot_rate <= 0.50
        and recall >= 0.50
        and (yaw_acc + pitch_acc) / 2.0 >= 0.80
        and sat95 < 0.05
        and sat80 < 0.15
        and improvement > 0.0
    )
    return {
        "shoot_pred_positive_rate": shoot_rate,
        "shoot_recall": float(recall),
        "shoot_precision": float(precision),
        "shoot_tp": tp,
        "shoot_fp": fp,
        "shoot_fn": fn,
        "look_yaw_sign_accuracy": yaw_acc,
        "look_pitch_sign_accuracy": pitch_acc,
        "look_sign_accuracy": float((yaw_acc + pitch_acc) / 2.0),
        "look_huber": look_huber,
        "init_look_huber": init_huber,
        "look_huber_improvement": float(improvement),
        "mu_look_sat95": sat95,
        "mu_look_sat80": sat80,
        "offline_gate_pass": gate,
        "actions_finite": bool(torch.isfinite(pred).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--gvg-relabel", required=True)
    parser.add_argument("--pressure-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=DEFAULT_RESUME)
    parser.add_argument("--normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--samples-per-epoch", type=int, default=160000)
    parser.add_argument("--look-weight", type=float, default=10.0)
    parser.add_argument("--shoot-weight", type=float, default=1.0)
    parser.add_argument("--movement-weight", type=float, default=0.05)
    parser.add_argument("--other-binary-weight", type=float, default=0.05)
    parser.add_argument("--sat-penalty-weight", type=float, default=5.0)
    parser.add_argument("--sat-threshold", type=float, default=0.65)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--original-frac", type=float, default=0.40)
    parser.add_argument("--gvg-frac", type=float, default=0.40)
    parser.add_argument("--pressure-frac", type=float, default=0.20)
    parser.add_argument("--side-b-mult", type=float, default=1.0)
    parser.add_argument("--underfire-mult", type=float, default=1.0)
    parser.add_argument("--postreset-mult", type=float, default=1.0)
    parser.add_argument("--visible-nohit-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=9400)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    parts = [load_original(Path(args.original_root)), load_gvg(Path(args.gvg_relabel))]
    pressure = maybe_load_pressure(args.pressure_root)
    if pressure is not None:
        parts.append(pressure)
    data = concat_data(parts)
    weights = build_sampling_weights(
        data,
        args.original_frac,
        args.gvg_frac,
        args.pressure_frac,
        args.side_b_mult,
        args.underfire_mult,
        args.postreset_mult,
        args.visible_nohit_mult,
    )
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    cfg = RuntimeConfig(
        resume_from=args.resume_from,
        normalizer_path=args.normalizer,
        device=str(device),
        min_log_std=-20.0,
        max_log_std=5.0,
    )
    policy, normalizer, *_ = load_policy_and_normalizer(cfg)
    policy.to(device)
    trainable_names = freeze_actor_heads_only(policy)
    init_policy, init_normalizer, *_ = load_policy_and_normalizer(cfg)
    init_policy.to(device)
    init_pred = predict(init_policy, init_normalizer, data["obs"], device)
    obs_t = torch.as_tensor(data["obs"], dtype=torch.float32)
    norm_chunks = []
    for start in range(0, len(obs_t), 65536):
        norm_chunks.append(normalizer.normalize_tensor(obs_t[start : start + 65536].to(device)).cpu())
    obs_norm = torch.cat(norm_chunks, dim=0)
    action_t = torch.as_tensor(data["action"], dtype=torch.float32)
    shoot_t = torch.as_tensor(data["shoot_intent"], dtype=torch.float32)
    mask_t = torch.as_tensor(data["shoot_mask"], dtype=torch.float32)
    init_pred_t = init_pred.float()
    source_original = torch.as_tensor((data["source_kind"].astype(str) == "original").astype(np.float32))
    dataset = TensorDataset(obs_norm, action_t, shoot_t, mask_t, init_pred_t, source_original)
    n_val = max(1, int(0.10 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_weights = torch.as_tensor(weights[np.asarray(train_ds.indices)], dtype=torch.double)
    sampler = WeightedRandomSampler(train_weights, num_samples=min(args.samples_per_epoch, len(train_weights)), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW([param for param in policy.parameters() if param.requires_grad], lr=args.lr, weight_decay=1e-4)
    curves = []

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        policy.set_ppo_mode(training=train_mode)
        totals = {key: 0.0 for key in ("loss", "look", "shoot", "move", "other_bin", "sat", "anchor")}
        n = 0
        for xb, yb, sb, mb, initb, origb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            mb = mb.to(device)
            initb = initb.to(device)
            origb = origb.to(device).view(-1, 1)
            pred = policy.predict_actions(xb.view(xb.shape[0], 1, OBS_DIM), policy.init_hidden(xb.shape[0], device), return_logits=True).squeeze(1)
            look_loss = F.huber_loss(pred[:, 2:4], yb[:, 2:4], delta=0.25)
            shoot_raw = F.binary_cross_entropy_with_logits(pred[:, 4], sb, reduction="none")
            shoot_loss = (shoot_raw * mb).sum() / torch.clamp(mb.sum(), min=1.0)
            move_loss = F.mse_loss(pred[:, 0:2], yb[:, 0:2])
            other_bin = F.binary_cross_entropy_with_logits(pred[:, 5:], yb[:, 5:])
            sat_loss = torch.relu(pred[:, 2:4].abs() - args.sat_threshold).pow(2).mean()
            anchor_loss = ((pred[:, 2:4] - initb[:, 2:4]).pow(2) * origb).sum() / torch.clamp(origb.sum() * 2.0, min=1.0)
            loss = (
                args.movement_weight * move_loss
                + args.look_weight * look_loss
                + args.shoot_weight * shoot_loss
                + args.other_binary_weight * other_bin
                + args.sat_penalty_weight * sat_loss
                + args.anchor_weight * anchor_loss
            )
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([param for param in policy.parameters() if param.requires_grad], 1.0)
                optimizer.step()
            bs = len(xb)
            n += bs
            for key, value in (
                ("loss", loss),
                ("look", look_loss),
                ("shoot", shoot_loss),
                ("move", move_loss),
                ("other_bin", other_bin),
                ("sat", sat_loss),
                ("anchor", anchor_loss),
            ):
                totals[key] += float(value.item()) * bs
        return {key: value / max(n, 1) for key, value in totals.items()}

    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch, "train": run_epoch(train_loader, True), "val": run_epoch(val_loader, False)}
        curves.append(row)
        with (out / "training_curves.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    val_indices = np.asarray(val_ds.indices)
    validation = eval_offline(
        policy,
        normalizer,
        data["obs"][val_indices],
        data["action"][val_indices],
        data["shoot_intent"][val_indices],
        data["shoot_mask"][val_indices],
        init_pred[val_indices],
        device,
    )
    checkpoint = out / "gvg_dagger_v2_actor_head.pt"
    torch.save(
        {
            "actor_critic_state": policy.state_dict(),
            "normalizer_path": args.normalizer,
            "normalizer_sha256": sha256_file(Path(args.normalizer)),
            "parent_checkpoint": args.resume_from,
            "parent_checkpoint_sha256": sha256_file(Path(args.resume_from)),
            "validation": validation,
            "curves": curves,
            "trainable_param_names": trainable_names,
            "training_args": vars(args),
        },
        checkpoint,
    )
    source_counts = {key: int(np.sum(data["source_kind"].astype(str) == key)) for key in ("original", "gvg", "pressure")}
    side_counts = {key: int(np.sum(data["agent_side"].astype(str) == key)) for key in ("A", "B", "scripted")}
    summary = {
        "status": "PASS" if validation["offline_gate_pass"] else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "samples": int(len(data["obs"])),
        "source_counts": source_counts,
        "side_counts": side_counts,
        "pressure_data_available": pressure is not None,
        "trainable_param_names": trainable_names,
        "validation": validation,
        "final_train": curves[-1]["train"],
        "final_val": curves[-1]["val"],
        "args": vars(args),
    }
    write_json(out / "gvg_dagger_v2_train_summary.json", summary)
    print(json.dumps({"status": summary["status"], "checkpoint_sha256": summary["checkpoint_sha256"], "offline_gate_pass": validation["offline_gate_pass"]}, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
