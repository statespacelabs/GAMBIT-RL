import os
import torch
from .ddp_utils import is_main_process, barrier

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    cfg: "TrainConfig",
    save_dir: str,
    ema_state_dict: dict = None,
    lr_scheduler_state_dict: dict = None,
    best_val_loss: float = float("inf"),
    patience_counter: int = 0,
    is_best: bool = False,
):
    if not is_main_process():
        return
        
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_{epoch:03d}.pt")
    
    # Ensure we save model.module if DDP is used to strip the 'module.' prefix
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    
    state = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": cfg.to_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }
    if ema_state_dict is not None:
        state["ema"] = ema_state_dict
    if lr_scheduler_state_dict is not None:
        state["lr_scheduler"] = lr_scheduler_state_dict

    torch.save(state, path)
    print(f"Checkpoint saved to {path}")
    
    if is_best:
        best_path = os.path.join(save_dir, "best_checkpoint.pt")
        torch.save(state, best_path)
        print(f"Best checkpoint saved to {best_path}")

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    ema=None,
    lr_scheduler=None,
) -> tuple[int, int, float, int]:
    ckpt = torch.load(path, map_location=device)
    
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(ckpt["model"])
    
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
        
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
        
    if lr_scheduler is not None and "lr_scheduler" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    patience_counter = ckpt.get("patience_counter", 0)
    
    barrier()
    return epoch, global_step, best_val_loss, patience_counter
