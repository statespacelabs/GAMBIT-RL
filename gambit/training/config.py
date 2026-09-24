from dataclasses import dataclass, asdict
from typing import Optional
import yaml

@dataclass
class TrainConfig:
    manifest_path: str
    image_size: tuple[int, int]
    seq_len: int

    epochs: int
    batches_per_epoch: int
    val_batches_per_epoch: int

    players_per_rank: int
    clips_per_player: int

    num_workers: int
    amp: bool
    lr: float
    min_lr: float
    weight_decay: float
    grad_clip_norm: Optional[float]
    
    ema_decay: float

    gather_z: bool
    gather_z_raw: bool
    gather_p: bool
    vc_on_global_p: bool
    vc_on_global_z_raw: bool
    
    contrastive_loss_mode: str

    log_every: int
    save_every_epochs: int
    patience: int
    output_dir: str
    seed: int

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            
        return cls(
            manifest_path=data.get("manifest_path", ""),
            image_size=tuple(data.get("image_size", (160, 240))),
            seq_len=data.get("seq_len", 150),
            epochs=data.get("epochs", 50),
            batches_per_epoch=data.get("batches_per_epoch", 1000),
            val_batches_per_epoch=data.get("val_batches_per_epoch", 20),
            players_per_rank=data.get("players_per_rank", 8),
            clips_per_player=data.get("clips_per_player", 4),
            num_workers=data.get("num_workers", 4),
            amp=data.get("amp", True),
            lr=data.get("lr", 0.0001),
            min_lr=float(data.get("min_lr", 1e-6)),
            weight_decay=data.get("weight_decay", 0.05),
            grad_clip_norm=data.get("grad_clip_norm", 1.0),
            ema_decay=float(data.get("ema_decay", 0.999)),
            gather_z=data.get("gather_z", True),
            gather_z_raw=data.get("gather_z_raw", True),
            gather_p=data.get("gather_p", True),
            vc_on_global_p=data.get("vc_on_global_p", True),
            vc_on_global_z_raw=data.get("vc_on_global_z_raw", True),
            contrastive_loss_mode=data.get("contrastive_loss_mode", "global_infonce_local_vc"),
            log_every=data.get("log_every", 20),
            save_every_epochs=data.get("save_every_epochs", 1),
            patience=data.get("patience", 5),
            output_dir=data.get("output_dir", "checkpoints"),
            seed=data.get("seed", 42),
        )

    def validate(self):
        if self.clips_per_player < 2:
            raise ValueError("clips_per_player must be >= 2")

        if self.vc_on_global_p and not self.gather_p:
            raise ValueError("vc_on_global_p=True requires gather_p=True")

        if self.vc_on_global_z_raw and not self.gather_z_raw:
            raise ValueError("vc_on_global_z_raw=True requires gather_z_raw=True")

        if self.contrastive_loss_mode == "global_infonce_local_vc" and not self.gather_z:
            raise ValueError("global_infonce_local_vc contrastive loss requires gather_z=True")

        if self.contrastive_loss_mode != "global_infonce_local_vc":
            raise ValueError("Only global_infonce_local_vc is implemented right now.")

    def to_dict(self):
        return asdict(self)
