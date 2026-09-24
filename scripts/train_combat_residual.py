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
from scripts.train_combat_dagger import (
    build_sampling_weights,
    concat_data,
    load_gvg,
    load_original,
    maybe_load_pressure,
    predict,
)


def freeze_for_residual(policy: torch.nn.Module) -> list[str]:
    for param in policy.parameters():
        param.requires_grad_(False)
    trainable = []
    for module_name in ("actor_cont_mean", "actor_binary_logits"):
        module = getattr(policy, module_name)
        for name, param in module.named_parameters():
            param.requires_grad_(True)
            trainable.append(f"{module_name}.{name}")
    if hasattr(policy, "actor_cont_logstd"):
        policy.actor_cont_logstd.requires_grad_(False)
    return trainable


def mask_residual_grads(policy: torch.nn.Module) -> None:
    cont = policy.actor_cont_mean
    binary = policy.actor_binary_logits
    if cont.weight.grad is not None:
        cont.weight.grad[:2].zero_()
    if cont.bias.grad is not None:
        cont.bias.grad[:2].zero_()
    if binary.weight.grad is not None:
        binary.weight.grad[1:].zero_()
    if binary.bias.grad is not None:
        binary.bias.grad[1:].zero_()


def eval_residual(
    policy,
    normalizer,
    obs: np.ndarray,
    target_action: np.ndarray,
    shoot: np.ndarray,
    mask: np.ndarray,
    init_pred: torch.Tensor,
    look_cap: float,
    shoot_logit_cap: float,
    device: torch.device,
) -> dict[str, Any]:
    pred = predict(policy, normalizer, obs, device)
    target = torch.as_tensor(target_action, dtype=torch.float32)
    shoot_t = torch.as_tensor(shoot, dtype=torch.float32)
    mask_t = torch.as_tensor(mask, dtype=torch.float32)
    look_delta = pred[:, 2:4] - init_pred[:, 2:4]
    shoot_delta = pred[:, 4] - init_pred[:, 4]
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
    return {
        "shoot_pred_positive_rate": float(pred_pos.float().mean().item()),
        "shoot_recall": float(recall),
        "shoot_precision": float(precision),
        "shoot_tp": tp,
        "shoot_fp": fp,
        "shoot_fn": fn,
        "look_sign_accuracy": float((yaw_acc + pitch_acc) / 2.0),
        "look_yaw_sign_accuracy": yaw_acc,
        "look_pitch_sign_accuracy": pitch_acc,
        "look_huber": look_huber,
        "init_look_huber": init_huber,
        "look_huber_improvement": float((init_huber - look_huber) / max(init_huber, 1e-8)),
        "look_residual_abs_max": float(look_delta.abs().max().item()),
        "look_residual_abs_gt_cap_rate": float((look_delta.abs() > (look_cap + 1e-4)).float().mean().item()),
        "shoot_logit_residual_abs_max": float(shoot_delta.abs().max().item()),
        "shoot_logit_residual_abs_gt_cap_rate": float((shoot_delta.abs() > (shoot_logit_cap + 1e-4)).float().mean().item()),
        "mu_look_sat95": float((pred[:, 2:4].abs() > 0.95).float().mean().item()),
        "mu_look_sat80": float((pred[:, 2:4].abs() > 0.80).float().mean().item()),
        "actions_finite": bool(torch.isfinite(pred).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--gvg-relabel", required=True)
    parser.add_argument("--pressure-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--samples-per-epoch", type=int, default=120000)
    parser.add_argument("--look-residual-cap", type=float, required=True)
    parser.add_argument("--shoot-logit-cap", type=float, default=0.75)
    parser.add_argument("--look-weight", type=float, default=10.0)
    parser.add_argument("--shoot-weight", type=float, default=1.0)
    parser.add_argument("--cap-penalty-weight", type=float, default=100.0)
    parser.add_argument("--original-frac", type=float, default=0.45)
    parser.add_argument("--gvg-frac", type=float, default=0.40)
    parser.add_argument("--pressure-frac", type=float, default=0.15)
    parser.add_argument("--side-b-mult", type=float, default=1.0)
    parser.add_argument("--underfire-mult", type=float, default=1.0)
    parser.add_argument("--postreset-mult", type=float, default=1.0)
    parser.add_argument("--visible-nohit-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=9900)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    cfg = RuntimeConfig(resume_from=args.resume_from, normalizer_path=args.normalizer, device=str(device), min_log_std=-20.0, max_log_std=5.0)
    policy, normalizer, *_ = load_policy_and_normalizer(cfg)
    policy.to(device)
    trainable = freeze_for_residual(policy)
    init_policy, init_normalizer, *_ = load_policy_and_normalizer(cfg)
    init_policy.to(device)
    init_pred = predict(init_policy, init_normalizer, data["obs"], device)
    label_action = torch.as_tensor(data["action"], dtype=torch.float32)
    residual_target = init_pred.clone()
    residual_target[:, 2:4] = init_pred[:, 2:4] + torch.clamp(label_action[:, 2:4] - init_pred[:, 2:4], -args.look_residual_cap, args.look_residual_cap)
    # Movement and non-shoot binary outputs are deliberately anchored to v000.
    residual_target[:, 0:2] = init_pred[:, 0:2]
    residual_target[:, 5:8] = init_pred[:, 5:8]
    obs_t = torch.as_tensor(data["obs"], dtype=torch.float32)
    norm_chunks = []
    for start in range(0, len(obs_t), 65536):
        norm_chunks.append(normalizer.normalize_tensor(obs_t[start : start + 65536].to(device)).cpu())
    obs_norm = torch.cat(norm_chunks, dim=0)
    shoot_t = torch.as_tensor(data["shoot_intent"], dtype=torch.float32)
    mask_t = torch.as_tensor(data["shoot_mask"], dtype=torch.float32)
    init_pred_t = init_pred.float()
    dataset = TensorDataset(obs_norm, residual_target.float(), shoot_t, mask_t, init_pred_t)
    n_val = max(1, int(0.10 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_weights = torch.as_tensor(weights[np.asarray(train_ds.indices)], dtype=torch.double)
    sampler = WeightedRandomSampler(train_weights, num_samples=min(args.samples_per_epoch, len(train_weights)), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    curves = []

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        policy.set_ppo_mode(training=train_mode)
        totals = {key: 0.0 for key in ("loss", "look", "shoot", "cap")}
        n = 0
        for xb, yb, sb, mb, initb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)
            mb = mb.to(device)
            initb = initb.to(device)
            pred = policy.predict_actions(xb.view(xb.shape[0], 1, OBS_DIM), policy.init_hidden(xb.shape[0], device), return_logits=True).squeeze(1)
            look_loss = F.huber_loss(pred[:, 2:4], yb[:, 2:4], delta=0.05)
            shoot_raw = F.binary_cross_entropy_with_logits(pred[:, 4], sb, reduction="none")
            shoot_loss = (shoot_raw * mb).sum() / torch.clamp(mb.sum(), min=1.0)
            look_excess = torch.relu((pred[:, 2:4] - initb[:, 2:4]).abs() - args.look_residual_cap).pow(2).mean()
            shoot_excess = torch.relu((pred[:, 4] - initb[:, 4]).abs() - args.shoot_logit_cap).pow(2).mean()
            cap_loss = look_excess + shoot_excess
            loss = args.look_weight * look_loss + args.shoot_weight * shoot_loss + args.cap_penalty_weight * cap_loss
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                mask_residual_grads(policy)
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 0.5)
                optimizer.step()
            bs = len(xb)
            n += bs
            for key, value in (("loss", loss), ("look", look_loss), ("shoot", shoot_loss), ("cap", cap_loss)):
                totals[key] += float(value.item()) * bs
        return {key: value / max(n, 1) for key, value in totals.items()}

    checkpoints = []
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch, "train": run_epoch(train_loader, True), "val": run_epoch(val_loader, False)}
        curves.append(row)
        with (out / "training_curves.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        val_indices = np.asarray(val_ds.indices)
        validation = eval_residual(
            policy,
            normalizer,
            data["obs"][val_indices],
            residual_target[val_indices].numpy(),
            data["shoot_intent"][val_indices],
            data["shoot_mask"][val_indices],
            init_pred[val_indices],
            args.look_residual_cap,
            args.shoot_logit_cap,
            device,
        )
        ckpt = out / f"bounded_residual_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "actor_critic_state": policy.state_dict(),
                "normalizer_path": args.normalizer,
                "normalizer_sha256": sha256_file(Path(args.normalizer)),
                "parent_checkpoint": args.resume_from,
                "parent_checkpoint_sha256": sha256_file(Path(args.resume_from)),
                "phase3v2_c_v22_bounded_residual": {
                    "look_residual_cap": args.look_residual_cap,
                    "shoot_logit_cap": args.shoot_logit_cap,
                    "movement_residual": "disabled",
                    "actor_cont_logstd": "frozen",
                    "trained_rows": ["actor_cont_mean[2:4]", "actor_binary_logits[0]"],
                },
                "validation": validation,
                "curves": curves,
                "training_args": vars(args),
                "trainable_param_names": trainable,
            },
            ckpt,
        )
        item = {"epoch": epoch, "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt), "validation": validation}
        checkpoints.append(item)
        print(json.dumps({"epoch": epoch, "checkpoint_sha256": item["checkpoint_sha256"], "validation": validation}, sort_keys=True), flush=True)
    summary = {
        "status": "PASS",
        "checkpoints": checkpoints,
        "final_checkpoint": checkpoints[-1]["checkpoint"],
        "final_checkpoint_sha256": checkpoints[-1]["checkpoint_sha256"],
        "samples": int(len(data["obs"])),
        "pressure_data_available": pressure is not None,
        "args": vars(args),
    }
    write_json(out / "bounded_residual_train_summary.json", summary)
    print(json.dumps({"status": "PASS", "final_checkpoint_sha256": summary["final_checkpoint_sha256"], "epochs": args.epochs}, sort_keys=True))


if __name__ == "__main__":
    main()
