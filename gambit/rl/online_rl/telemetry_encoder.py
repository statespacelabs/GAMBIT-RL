"""TelemetryEncoder: reuses pretrained telemetry branches from InquisitorEncoder.

Extracts WhereBranch, ViewBranch, RhythmBranch, and TelemetryFusion from a
Phase 1 encoder checkpoint and adds a trainable 256→512 projection layer.

The output z_tel [B,T,512] is consumed by the GRU in RecurrentActorCritic.
All operations keep native [B,T,dim] shape — works for both T=1 rollout
and T=32 sequence training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from gambit.models.where_branch import WhereBranch
from gambit.models.view_branch import ViewBranch
from gambit.models.rhythm_branch import RhythmBranch
from gambit.models.telemetry_fusion import TelemetryFusion
from gambit.models.types import TelemetryConfig

logger = logging.getLogger(__name__)

# Telemetry split boundaries within the 45-dim Unity observation
WHERE_SLICE = slice(0, 10)  # position, velocity, acceleration, distance
VIEW_SLICE = slice(10, 23)  # sin/cos angles, viewing vel/acc, crosshair
RHYTHM_SLICE = slice(23, 45)  # action history features


class TelemetryEncoder(nn.Module):
    """Pretrained telemetry pipeline for PPO observation encoding.

    Architecture:
        obs_45 [B,T,45]
          → split into where(10), view(13), rhythm(22)
          → WhereBranch(10→128), ViewBranch(13→128), RhythmBranch(22→128)
          → TelemetryFusion(where, view, rhythm) → [B,T,256]
          → projection Linear(256→512) + LayerNorm + GELU → z_tel [B,T,512]

    All branch modules are loaded from a pretrained InquisitorEncoder checkpoint.
    The projection layer (256→512) is randomly initialized and always trainable.
    """

    def __init__(
        self,
        tel_config: TelemetryConfig | None = None,
        proj_dim: int = 512,
    ):
        super().__init__()

        if tel_config is None:
            tel_config = TelemetryConfig()

        self.tel_config = tel_config

        # Pretrained branches (weights loaded from checkpoint)
        self.where_branch = WhereBranch(
            input_dim=tel_config.where_dim,
            hidden_dim=tel_config.hidden_dim,
            out_dim=tel_config.out_dim,
        )
        self.view_branch = ViewBranch(
            input_dim=tel_config.view_dim,
            hidden_dim=tel_config.hidden_dim,
            out_dim=tel_config.out_dim,
        )
        self.rhythm_branch = RhythmBranch(
            input_dim=tel_config.rhythm_dim,
            hidden_dim=tel_config.hidden_dim,
            out_dim=tel_config.out_dim,
        )
        self.telemetry_fusion = TelemetryFusion(
            in_features=tel_config.out_dim * 3,  # 384
            out_features=256,
        )

        # New trainable projection: 256 → 512
        self.projection = nn.Sequential(
            nn.Linear(256, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

    def forward(self, obs_45: torch.Tensor) -> torch.Tensor:
        """Encode 45-dim telemetry observation into 512-dim latent.

        Args:
            obs_45: [B, T, 45] telemetry observation.
                    T can be 1 (per-step rollout) or any length (sequence training).

        Returns:
            z_tel: [B, T, 512] telemetry latent.
        """
        where = self.where_branch(obs_45[..., WHERE_SLICE])  # [B,T,128]
        view = self.view_branch(obs_45[..., VIEW_SLICE])  # [B,T,128]
        rhythm = self.rhythm_branch(obs_45[..., RHYTHM_SLICE])  # [B,T,128]

        tel = self.telemetry_fusion(where, view, rhythm)  # [B,T,256]
        z_tel = self.projection(tel)  # [B,T,512]

        return z_tel

    def freeze_branches(self) -> None:
        """Freeze pretrained branch weights (for early distillation)."""
        for module in [
            self.where_branch,
            self.view_branch,
            self.rhythm_branch,
            self.telemetry_fusion,
        ]:
            for param in module.parameters():
                param.requires_grad = False
        logger.info("TelemetryEncoder: branches frozen")

    def unfreeze_branches(self) -> None:
        """Unfreeze branch weights (for fine-tuning)."""
        for module in [
            self.where_branch,
            self.view_branch,
            self.rhythm_branch,
            self.telemetry_fusion,
        ]:
            for param in module.parameters():
                param.requires_grad = True
        logger.info("TelemetryEncoder: branches unfrozen")

    @classmethod
    def from_encoder_checkpoint(
        cls,
        ckpt_path: str | Path,
        device: torch.device | str = "cpu",
        proj_dim: int = 512,
    ) -> "TelemetryEncoder":
        """Load pretrained telemetry branch weights from an InquisitorEncoder checkpoint.

        Copies weights for: where_branch, view_branch, rhythm_branch, telemetry_fusion.
        The projection layer is randomly initialized.

        Args:
            ckpt_path: Path to a Phase 1 InquisitorEncoder checkpoint.
            device: Device to load weights onto.
            proj_dim: Output projection dimension (default 512).

        Returns:
            TelemetryEncoder with pretrained branch weights.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Prefer EMA weights, then clean DDP and torch.compile prefixes.
        state_dict = ckpt.get(
            "ema", ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        )
        cleaned = {}
        for key, value in state_dict.items():
            while key.startswith(("module.", "_orig_mod.")):
                key = key.removeprefix("module.").removeprefix("_orig_mod.")
            cleaned[key] = value
        state_dict = cleaned

        # Create encoder with default telemetry config
        encoder = cls(proj_dim=proj_dim)

        # Copy branch weights
        branches = {
            "where_branch": encoder.where_branch,
            "view_branch": encoder.view_branch,
            "rhythm_branch": encoder.rhythm_branch,
            "telemetry_fusion": encoder.telemetry_fusion,
        }

        loaded_count = 0
        for branch_name, branch_module in branches.items():
            branch_prefix = f"{branch_name}."
            branch_state = {
                k[len(branch_prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(branch_prefix)
            }

            if not branch_state:
                raise RuntimeError(
                    f"Checkpoint {ckpt_path} has no weights for {branch_name}"
                )

            branch_module.load_state_dict(branch_state, strict=True)
            loaded_count += len(branch_state)

        logger.info(
            "TelemetryEncoder: loaded %d parameters from %s "
            "(projection layer randomly initialized)",
            loaded_count,
            ckpt_path.name,
        )

        encoder.to(device)
        return encoder
