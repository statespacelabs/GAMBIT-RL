#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combat_runtime import OBS_DIM, RuntimeConfig, load_policy_and_normalizer, sha256_file, write_json


INIT_PARENT = "experiments/phase3v2/phase3v2_a_policy_init_smoke/init_from_encoder.pt"
NORMALIZER = "experiments/phase3v2/normalizers/phase3v2_a_obs45_no_pressure.pt"


def load_npz(root: Path, mix: str):
    shards = sorted(root.glob("job_*/teacher_shard_v2.npz"))
    obs = []
    act = []
    shoot = []
    mask = []
    aim = []
    bot = []
    for p in shards:
        z = np.load(p, allow_pickle=True)
        mode = str(z["bot_mode"][0])
        if mix == "face_saf" and mode == "Idle":
            continue
        obs.append(z["obs"])
        act.append(z["action"])
        shoot.append(z["shoot_intent"])
        mask.append(z["shoot_mask"])
        aim.append(z["aim_error"])
        bot.append(np.asarray([mode] * len(z["obs"])))
    if not obs:
        raise FileNotFoundError(f"no usable teacher shards under {root}")
    obs = np.concatenate(obs)
    act = np.concatenate(act)
    shoot = np.concatenate(shoot)
    mask = np.concatenate(mask)
    aim = np.concatenate(aim)
    bot = np.concatenate(bot)
    if mix == "saf_heavy":
        rng = np.random.default_rng(1234)
        saf = np.where(bot == "StrafeAndFace")[0]
        other = np.where(bot != "StrafeAndFace")[0]
        target_saf = int(0.60 * len(obs))
        target_other = len(obs) - target_saf
        idx = np.concatenate(
            [
                rng.choice(saf, target_saf, replace=len(saf) < target_saf),
                rng.choice(other, target_other, replace=len(other) < target_other),
            ]
        )
        rng.shuffle(idx)
        obs, act, shoot, mask, aim, bot = obs[idx], act[idx], shoot[idx], mask[idx], aim[idx], bot[idx]
    return obs, act, shoot, mask, aim, bot


def freeze_actor_path(policy) -> None:
    for p in policy.parameters():
        p.requires_grad_(False)
    for module in [policy.gru, policy.actor_cont_mean, policy.actor_binary_logits]:
        for p in module.parameters():
            p.requires_grad_(True)
    if hasattr(policy, "actor_cont_logstd"):
        policy.actor_cont_logstd.requires_grad_(True)
    policy.value_head.requires_grad_(False)
    policy.tel_encoder.freeze_branches()


def predict(policy, normalizer, obs: np.ndarray, device: torch.device, batch_size: int = 8192) -> torch.Tensor:
    policy.set_ppo_mode(training=False)
    rows = []
    with torch.no_grad():
        for i in range(0, len(obs), batch_size):
            xb = torch.as_tensor(obs[i : i + batch_size], dtype=torch.float32, device=device)
            xb = normalizer.normalize_tensor(xb).view(-1, 1, OBS_DIM)
            rows.append(policy.predict_actions(xb, policy.init_hidden(xb.shape[0], device), return_logits=True).squeeze(1).cpu())
    return torch.cat(rows, dim=0)


def eval_offline(policy, normalizer, obs, act, shoot, mask, aim, init_pred, device):
    pred = predict(policy, normalizer, obs, device)
    target = torch.as_tensor(act, dtype=torch.float32)
    shoot_t = torch.as_tensor(shoot, dtype=torch.float32)
    mask_t = torch.as_tensor(mask, dtype=torch.float32)
    aim_t = torch.as_tensor(aim, dtype=torch.float32)
    shoot_prob = torch.sigmoid(pred[:, 4])
    shoot_pred = shoot_prob > 0.5
    valid = mask_t > 0.5
    pos = (shoot_t > 0.5) & valid
    neg = (shoot_t < 0.5) & valid
    tp = ((shoot_pred) & pos).sum().item()
    fp = ((shoot_pred) & neg).sum().item()
    fn = ((~shoot_pred) & pos).sum().item()
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    yaw_mask = torch.abs(target[:, 2]) > 0.03
    pitch_mask = torch.abs(target[:, 3]) > 0.03
    yaw_acc = float((torch.sign(pred[:, 2]) == torch.sign(target[:, 2])).float()[yaw_mask].mean().item()) if torch.any(yaw_mask) else 1.0
    pitch_acc = float((torch.sign(pred[:, 3]) == torch.sign(target[:, 3])).float()[pitch_mask].mean().item()) if torch.any(pitch_mask) else 1.0
    look_huber = float(F.huber_loss(pred[:, 2:4], target[:, 2:4], delta=0.25).item())
    init_huber = float(F.huber_loss(init_pred[:, 2:4], target[:, 2:4], delta=0.25).item())
    look_mse = float(F.mse_loss(pred[:, 2:4], target[:, 2:4]).item())
    init_mse = float(F.mse_loss(init_pred[:, 2:4], target[:, 2:4]).item())
    look_abs = torch.abs(pred[:, 2:4])
    target_look_abs = torch.abs(target[:, 2:4])
    label_sat95 = float((target_look_abs > 0.95).float().mean().item())
    label_sat80 = float((target_look_abs > 0.80).float().mean().item())
    good = aim_t < 15.0
    good_mean_abs = float(look_abs[good].mean().item()) if torch.any(good) else 999.0
    finite = bool(torch.isfinite(pred).all().item())
    shoot_rate = float(shoot_pred.float().mean().item())
    sat95 = float((look_abs > 0.95).float().mean().item())
    sat80 = float((look_abs > 0.80).float().mean().item())
    saturation_gate = bool(sat95 <= max(0.01, label_sat95 + 0.05) and sat80 <= max(0.10, label_sat80 + 0.05))
    all_neg = bool(torch.all(pred[:, 4] < -5.0).item())
    improvement = (init_huber - look_huber) / max(init_huber, 1e-8)
    gate = bool(
        finite
        and 0.10 <= shoot_rate <= 0.50
        and recall >= 0.50
        and (yaw_acc + pitch_acc) / 2.0 >= 0.75
        and improvement >= 0.50
        and saturation_gate
        and good_mean_abs < 0.25
        and not all_neg
    )
    return {
        "shoot_pred_positive_rate": shoot_rate,
        "shoot_recall": float(recall),
        "shoot_precision": float(precision),
        "look_yaw_sign_accuracy": yaw_acc,
        "look_pitch_sign_accuracy": pitch_acc,
        "look_sign_accuracy": float((yaw_acc + pitch_acc) / 2.0),
        "look_huber": look_huber,
        "init_look_huber": init_huber,
        "look_huber_improvement": float(improvement),
        "look_mse": look_mse,
        "init_look_mse": init_mse,
        "look_mse_improvement": float((init_mse - look_mse) / max(init_mse, 1e-8)),
        "look_abs_gt_0_95_rate": sat95,
        "look_abs_gt_0_80_rate": sat80,
        "label_look_abs_gt_0_95_rate": label_sat95,
        "label_look_abs_gt_0_80_rate": label_sat80,
        "look_saturation_gate_pass": saturation_gate,
        "good_aim_mean_abs_mu_look": good_mean_abs,
        "continuous_action_mean": pred[:, :4].mean(dim=0).numpy().astype(float).tolist(),
        "continuous_action_abs_mean": pred[:, :4].abs().mean(dim=0).numpy().astype(float).tolist(),
        "binary_logit_mean": pred[:, 4:].mean(dim=0).numpy().astype(float).tolist(),
        "actions_finite": finite,
        "binary_logits_not_all_negative": not all_neg,
        "offline_gate_pass": gate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resume-from", default=INIT_PARENT)
    ap.add_argument("--normalizer", default=NORMALIZER)
    ap.add_argument("--init-reference", default="")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--look-weight", type=float, default=5.0)
    ap.add_argument("--sat-penalty-weight", type=float, default=1.0)
    ap.add_argument("--sat-threshold", type=float, default=0.75)
    ap.add_argument("--shoot-handling", choices=["balanced", "posweight"], default="balanced")
    ap.add_argument("--mix", default="all")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs, act, shoot, mask, aim, bot = load_npz(Path(args.dataset_root), args.mix)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    cfg = RuntimeConfig(resume_from=args.resume_from, normalizer_path=args.normalizer, device=str(device))
    policy, normalizer, *_ = load_policy_and_normalizer(cfg)
    policy.to(device)
    freeze_actor_path(policy)

    init_reference = args.init_reference or args.resume_from or INIT_PARENT
    init_cfg = RuntimeConfig(resume_from=init_reference, normalizer_path=args.normalizer, device=str(device))
    init_policy, init_norm, *_ = load_policy_and_normalizer(init_cfg)
    init_policy.to(device)
    init_pred = predict(init_policy, init_norm, obs, device)

    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    norm = []
    for i in range(0, len(obs_t), 65536):
        norm.append(normalizer.normalize_tensor(obs_t[i : i + 65536].to(device)).cpu())
    obs_norm = torch.cat(norm, dim=0)
    act_t = torch.as_tensor(act, dtype=torch.float32)
    shoot_t = torch.as_tensor(shoot, dtype=torch.float32)
    mask_t = torch.as_tensor(mask, dtype=torch.float32)
    ds = TensorDataset(obs_norm, act_t, shoot_t, mask_t)
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(1234))

    if args.shoot_handling == "balanced":
        subset_idx = np.asarray(train_ds.indices)
        pos = (shoot[subset_idx] > 0.5) & (mask[subset_idx] > 0.5)
        neg = (shoot[subset_idx] < 0.5) & (mask[subset_idx] > 0.5)
        weights = np.ones(len(subset_idx), dtype=np.float32) * 0.2
        weights[pos] = 0.5 / max(pos.mean(), 1e-6)
        weights[neg] = 0.5 / max(neg.mean(), 1e-6)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        train = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
        pos_weight = None
    else:
        train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        pos_count = max(float(((shoot > 0.5) & (mask > 0.5)).sum()), 1.0)
        neg_count = max(float(((shoot < 0.5) & (mask > 0.5)).sum()), 1.0)
        pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32, device=device)
    val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    curves = []

    def run_epoch(loader, train_mode: bool):
        policy.set_ppo_mode(training=train_mode)
        totals = {"loss": 0.0, "look": 0.0, "move": 0.0, "shoot": 0.0, "other_bin": 0.0, "sat": 0.0}
        n = 0
        for xb, yb, sb, mb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            mb = mb.to(device)
            pred = policy.predict_actions(xb.view(xb.shape[0], 1, OBS_DIM), policy.init_hidden(xb.shape[0], device), return_logits=True).squeeze(1)
            move_loss = F.mse_loss(pred[:, 0:2], yb[:, 0:2])
            look_loss = F.huber_loss(pred[:, 2:4], yb[:, 2:4], delta=0.25)
            sat_loss = torch.relu(torch.abs(pred[:, 2:4]) - args.sat_threshold).pow(2).mean()
            shoot_raw = F.binary_cross_entropy_with_logits(pred[:, 4], sb, reduction="none", pos_weight=pos_weight)
            shoot_loss = (shoot_raw * mb).sum() / torch.clamp(mb.sum(), min=1.0)
            other_bin = F.binary_cross_entropy_with_logits(pred[:, 5:], yb[:, 5:], reduction="mean")
            loss = 0.1 * move_loss + args.look_weight * look_loss + shoot_loss + 0.1 * other_bin + args.sat_penalty_weight * sat_loss
            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
                opt.step()
            bs = len(xb)
            n += bs
            totals["loss"] += float(loss.item()) * bs
            totals["look"] += float(look_loss.item()) * bs
            totals["move"] += float(move_loss.item()) * bs
            totals["shoot"] += float(shoot_loss.item()) * bs
            totals["other_bin"] += float(other_bin.item()) * bs
            totals["sat"] += float(sat_loss.item()) * bs
        return {k: v / max(n, 1) for k, v in totals.items()}

    curve_path = out / "bc_training_curves.jsonl"
    for ep in range(1, args.epochs + 1):
        row = {"epoch": ep, "train": run_epoch(train, True), "val": run_epoch(val, False)}
        curves.append(row)
        with curve_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row), flush=True)

    validation = eval_offline(policy, normalizer, obs, act, shoot, mask, aim, init_pred, device)
    write_json(out / "bc_validation.json", validation)
    ckpt = out / "bc_actor_head.pt"
    torch.save(
        {
            "actor_critic_state": policy.state_dict(),
            "normalizer_path": cfg.normalizer_path,
            "normalizer_sha256": sha256_file(Path(cfg.normalizer_path)),
            "parent_checkpoint": args.resume_from,
        "init_reference_checkpoint": init_reference,
        "normalizer_path": args.normalizer,
            "bc_validation": validation,
            "bc_curves": curves,
            "bc_mix": args.mix,
            "bc_lr": args.lr,
            "bc_epochs": args.epochs,
            "bc_look_weight": args.look_weight,
            "bc_sat_penalty_weight": args.sat_penalty_weight,
            "bc_shoot_handling": args.shoot_handling,
        },
        ckpt,
    )
    summary = {
        "status": "PASS" if validation["offline_gate_pass"] else "FAIL",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "samples": int(len(obs)),
        "mix": args.mix,
        "epochs": args.epochs,
        "lr": args.lr,
        "look_weight": args.look_weight,
        "sat_penalty_weight": args.sat_penalty_weight,
        "shoot_handling": args.shoot_handling,
        "parent_checkpoint": args.resume_from,
        "validation": validation,
        "final_train": curves[-1]["train"],
        "final_val": curves[-1]["val"],
    }
    write_json(out / "bc_train_summary_v21.json", summary)
    print(json.dumps({"status": summary["status"], "checkpoint_sha256": summary["checkpoint_sha256"], "offline_gate_pass": validation["offline_gate_pass"]}, sort_keys=True))
    if not validation["offline_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
