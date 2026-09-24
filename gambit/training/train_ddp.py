import argparse
import torch
import torch.nn.parallel
from datetime import datetime

from gambit.training.ddp_utils import init_distributed, cleanup_distributed, seed_everything, is_main_process
from gambit.training.config import TrainConfig
from gambit.datasets.builders_ddp import build_ddp_train_loader, build_ddp_val_loader
from gambit.training.gather import concat_all_gather_no_grad, concat_all_gather_with_grad
from gambit.training.checkpoint import save_checkpoint, load_checkpoint
from gambit.training.ema import ModelEma

# Import your model components here
from gambit.models.inquisitor_encoder import InquisitorEncoder
from gambit.models.types import EncoderConfig
from gambit.models.loss import GambitLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ema", type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    cfg.validate()

    try:
        rank, local_rank, world_size, device = init_distributed()
        seed_everything(cfg.seed, rank)

        train_loader, train_sampler = build_ddp_train_loader(
            manifest_path=cfg.manifest_path,
            rank=rank,
            world_size=world_size,
            players_per_rank=cfg.players_per_rank,
            clips_per_player=cfg.clips_per_player,
            batches_per_epoch=cfg.batches_per_epoch,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )

        val_loader, val_sampler = build_ddp_val_loader(
            manifest_path=cfg.manifest_path,
            rank=rank,
            world_size=world_size,
            players_per_rank=cfg.players_per_rank,
            clips_per_player=cfg.clips_per_player,
            batches_per_epoch=cfg.val_batches_per_epoch,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )

        # Initialize model (assume default config for now, or load from another YAML)
        model_config = EncoderConfig()
        model = InquisitorEncoder(model_config).to(device)

        # Initialize EMA on the raw, uncompiled model to avoid deep-copying compiled graphs
        ema = ModelEma(model, decay=cfg.ema_decay, device=device) if args.ema else None

        # model = torch.compile(model)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
        criterion = GambitLoss().to(device)
        
        total_steps = cfg.epochs * cfg.batches_per_epoch
        warmup_steps = max(1, int(0.05 * total_steps))  # 5% linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

        global_step = 0
        start_epoch = 0
        best_val_loss = float("inf")
        patience_counter = 0

        if args.resume:
            start_epoch, global_step, best_val_loss, patience_counter = load_checkpoint(
                args.resume, model, optimizer, scaler, device, ema=ema, lr_scheduler=scheduler
            )

        total_steps = cfg.epochs * cfg.batches_per_epoch

        for epoch in range(start_epoch, cfg.epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            model.train()

            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, enabled=cfg.amp):
                    out = model(batch)

                    # Gathering logic based on config flags
                    z_global = concat_all_gather_with_grad(out.z_norm) if cfg.gather_z else out.z_norm
                    z_raw_global = concat_all_gather_with_grad(out.z_raw) if cfg.gather_z_raw else out.z_raw
                    p_global = concat_all_gather_with_grad(out.p) if cfg.gather_p else out.p
                    player_ids_global = concat_all_gather_no_grad(batch.player_ids) if cfg.gather_z else batch.player_ids
                    
                    # Double-check constraints
                    if is_main_process():
                        unique, counts = torch.unique(player_ids_global, return_counts=True)
                        assert counts.min().item() >= 2, f"Each player must have >= 2 clips, found min {counts.min().item()}"

                    if cfg.contrastive_loss_mode == "global_infonce_local_vc":
                        loss_dict = criterion(
                            z_norm=z_global,
                            z_raw=z_raw_global if cfg.vc_on_global_z_raw else out.z_raw,
                            p=p_global if cfg.vc_on_global_p else out.p,
                            player_ids=player_ids_global,
                            training_progress=global_step / max(1, total_steps),
                        )
                    elif cfg.contrastive_loss_mode == "local_anchors_global_candidates":
                        raise NotImplementedError("local_anchors_global_candidates not implemented in phase 1")
                    else:
                        raise ValueError(f"Unknown contrastive_loss_mode: {cfg.contrastive_loss_mode}")

                    loss = loss_dict["total"]

                scaler.scale(loss).backward()

                if cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.grad_clip_norm,
                    )

                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                if args.ema and ema is not None:
                    ema.update(model)

                if is_main_process() and global_step % cfg.log_every == 0:
                    unique, counts = torch.unique(player_ids_global, return_counts=True)
                    assert counts.min().item() == cfg.clips_per_player, "Min clips mismatch"
                    assert counts.max().item() == cfg.clips_per_player, "Max clips mismatch"
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"[{now}] Epoch: {epoch}, Step: {global_step}, LR: {current_lr:.6f}, Loss: {loss.item():.4f}, InfoNCE: {loss_dict['infonce'].item():.4f}, VC_Scale: {loss_dict['vc_scale'].item():.2f}, Var: {loss_dict['var_p'].item():.4f}, Cov: {loss_dict['cov_p'].item():.4f}")

                global_step += 1

            # Validation Loop
            model.eval()
            val_loss_sum = torch.zeros(1, device=device)
            val_steps_tensor = torch.zeros(1, device=device)
            
            top1_sum = torch.zeros(1, device=device)
            top5_sum = torch.zeros(1, device=device)
            val_total_samples = torch.zeros(1, device=device)
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=cfg.amp):
                        if args.ema and ema is not None:
                            out = ema.ema_model(batch)
                        else:
                            out = model(batch)
                        
                        z_global = concat_all_gather_no_grad(out.z_norm) if cfg.gather_z else out.z_norm
                        z_raw_global = concat_all_gather_no_grad(out.z_raw) if cfg.gather_z_raw else out.z_raw
                        p_global = concat_all_gather_no_grad(out.p) if cfg.gather_p else out.p
                        player_ids_global = concat_all_gather_no_grad(batch.player_ids) if cfg.gather_z else batch.player_ids
                        
                        loss_dict = criterion(
                            z_norm=z_global,
                            z_raw=z_raw_global if cfg.vc_on_global_z_raw else out.z_raw,
                            p=p_global if cfg.vc_on_global_p else out.p,
                            player_ids=player_ids_global,
                            training_progress=1.0,
                        )
                        val_loss_sum += loss_dict["total"]
                        val_steps_tensor += 1
                        
                        # Calculate Top-1 and Top-5 Retrieval Accuracy
                        sim = torch.matmul(z_global, z_global.T)
                        B = z_global.shape[0]
                        self_mask = torch.eye(B, dtype=torch.bool, device=device)
                        sim.masked_fill_(self_mask, -float('inf'))
                        
                        k = min(5, B - 1)
                        if k > 0:
                            topk_indices = sim.topk(k, dim=1).indices # [B, k]
                            topk_players = player_ids_global[topk_indices] # [B, k]
                            matches = topk_players == player_ids_global.unsqueeze(1) # [B, k]
                            
                            top1_sum += matches[:, 0].float().sum()
                            top5_sum += matches.any(dim=1).float().sum()
                        val_total_samples += B
            
            # Reduce validation loss across all ranks for a global average
            torch.distributed.all_reduce(val_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(val_steps_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(top1_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(top5_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(val_total_samples, op=torch.distributed.ReduceOp.SUM)
            
            avg_val_loss = (val_loss_sum / val_steps_tensor.clamp_min(1)).item()
            avg_top1 = (top1_sum / val_total_samples.clamp_min(1)).item() * 100.0
            avg_top5 = (top5_sum / val_total_samples.clamp_min(1)).item() * 100.0
            
            if is_main_process():
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] Epoch {epoch} Validation Loss: {avg_val_loss:.4f}, Top-1 Acc: {avg_top1:.2f}%, Top-5 Acc: {avg_top5:.2f}%")
                
                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if (epoch + 1) % cfg.save_every_epochs == 0 or is_best:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch + 1,
                        global_step=global_step,
                        cfg=cfg,
                        save_dir=cfg.output_dir,
                        ema_state_dict=ema.state_dict() if (args.ema and ema is not None) else None,
                        lr_scheduler_state_dict=scheduler.state_dict(),
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter,
                        is_best=is_best,
                    )
            
            patience_tensor = torch.tensor([patience_counter], device=device)
            torch.distributed.broadcast(patience_tensor, src=0)
            
            if patience_tensor.item() >= cfg.patience:
                if is_main_process():
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
