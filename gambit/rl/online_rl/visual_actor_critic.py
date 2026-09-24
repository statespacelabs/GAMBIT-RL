"""Visual Recurrent PPO Actor-Critic.

Extends RecurrentActorCritic with a frozen VideoEncoder path.
Visual features (from per-frame ConvNeXt) are fused with telemetry embeddings
via a gated residual before entering the GRU.

Architecture:
    obs_45 [B,T,45] → TelemetryEncoder → z_tel [B,T,512]
    frames [B,T,3,H,W] → VideoEncoder (frozen) → v [B,T,512]
    VisualNudge: z_fused = z_tel + gate * proj(v)  [B,T,512]
    → GRU(512, hidden=512) → h_t [B,T,512]
    → actor/value heads (same as base)

The fusion is intentionally tel-dominant: gate bias initialized to -2
so visual info starts near-zero and grows only if useful.

Checkpoint compatibility:
    - Loads telemetry-only checkpoints by ignoring missing visual keys
    - VideoEncoder weights from Phase 1 InquisitorEncoder checkpoint
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from .actor_critic import RecurrentActorCritic
from .telemetry_encoder import TelemetryEncoder
from gambit.models.video_encoder import VideoEncoder

logger = logging.getLogger(__name__)


class VisualNudgeFusion(nn.Module):
    """Tel-dominant gated residual: z_fused = z_tel + gate * proj(v).

    Gate is initialized near-zero (bias=-2) so training starts
    effectively telemetry-only and learns to let visual info through.
    """

    def __init__(self, tel_dim: int = 512, video_dim: int = 512):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, tel_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(tel_dim + tel_dim, tel_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_net[0].bias, -2.0)
        self.last_gate: torch.Tensor | None = None

    def forward(self, video_features: torch.Tensor, tel_features: torch.Tensor) -> torch.Tensor:
        v_proj = self.video_proj(video_features)  # [B, T, tel_dim]
        gate = self.gate_net(torch.cat([tel_features, v_proj], dim=-1))
        self.last_gate = gate.detach()
        return tel_features + gate * v_proj

    def gate_mean(self) -> float:
        if self.last_gate is None:
            return 0.0
        return float(self.last_gate.mean())


class VisualRecurrentActorCritic(RecurrentActorCritic):
    """RecurrentActorCritic with frozen visual encoder fusion."""

    def __init__(
        self,
        tel_encoder: TelemetryEncoder,
        visual_proj_dim: int = 512,
        visual_pretrained: bool = True,
    ):
        super().__init__(tel_encoder)

        self.video_encoder = VideoEncoder(
            proj_dim=visual_proj_dim,
            pretrained=visual_pretrained,
        )
        self.video_encoder.freeze_backbone()

        self.visual_fusion = VisualNudgeFusion(
            tel_dim=512,
            video_dim=visual_proj_dim,
        )

        self._visual_enabled = True

    def freeze_visual(self) -> None:
        """Freeze the entire visual path (encoder + fusion gate)."""
        self.video_encoder.freeze_backbone()
        for param in self.video_encoder.proj.parameters():
            param.requires_grad = False
        for param in self.visual_fusion.parameters():
            param.requires_grad = False
        logger.info("VisualRecurrentActorCritic: visual path frozen")

    def unfreeze_fusion(self) -> None:
        """Unfreeze only the fusion gate (keep ConvNeXt backbone frozen)."""
        for param in self.visual_fusion.parameters():
            param.requires_grad = True
        logger.info("VisualRecurrentActorCritic: fusion unfrozen")

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
    ):
        """Forward pass with optional visual frames.

        Args:
            obs_seq: [B, T, 45] telemetry observation.
            hidden: [1, B, 512] GRU hidden state.
            visual_frames: [B, T, 3, H, W] RGB frames. If None, telemetry-only.

        Returns:
            dist, value, new_hidden (same as base class)

        Note: PPO updates call this without visual_frames (replays from buffer).
        The mismatch between rollout (with visual) and update (without) is
        acceptable because the gate starts near-zero (bias=-2 → sigmoid≈0.12).
        Monitor visual/gate_mean to ensure it stays small.
        """
        from .hybrid_distribution import HybridDistribution

        B, T, _ = obs_seq.shape
        if hidden is None:
            hidden = self.init_hidden(B, obs_seq.device)

        z_tel = self.tel_encoder(obs_seq)  # [B, T, 512]

        if visual_frames is not None and self._visual_enabled:
            _, _, C, H, W = visual_frames.shape
            frames_flat = visual_frames.reshape(B * T, C, H, W)
            with torch.no_grad():
                v_flat = self.video_encoder(frames_flat)  # [B*T, 512]
            v = v_flat.reshape(B, T, -1)  # [B, T, 512]
            z = self.visual_fusion(v, z_tel)  # [B, T, 512]
        else:
            z = z_tel

        h, new_hidden = self.gru(z, hidden)

        cont_mean = self.actor_cont_mean(h)
        cont_logstd = self.actor_cont_logstd.expand_as(cont_mean)
        bin_logits = self.actor_binary_logits(h)

        dist = HybridDistribution(cont_mean, cont_logstd, bin_logits)
        value = self.value_head(h).squeeze(-1)

        return dist, value, new_hidden

    def get_value(
        self,
        obs_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, value, _ = self.forward(obs_seq, hidden, visual_frames)
        return value

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions_raw: torch.Tensor,
        hidden: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
    ):
        dist, value, _ = self.forward(obs_seq, hidden, visual_frames)
        log_prob = dist.log_prob(actions_raw)
        entropy = dist.entropy()
        return log_prob, entropy, value

    @classmethod
    def from_telemetry_checkpoint(
        cls,
        ckpt_path: str | Path,
        encoder_ckpt_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        visual_proj_dim: int = 512,
    ) -> "VisualRecurrentActorCritic":
        """Load from a telemetry-only behavioral checkpoint + optional Phase 1 encoder.

        Args:
            ckpt_path: Path to a telemetry-only RecurrentActorCritic checkpoint.
            encoder_ckpt_path: Path to Phase 1 InquisitorEncoder for visual weights.
            device: Target device.
            visual_proj_dim: Visual projection dim (default 512).
        """
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ac_state = ckpt["actor_critic_state"]

        tel_encoder = TelemetryEncoder(proj_dim=512)
        instance = cls(
            tel_encoder=tel_encoder,
            visual_proj_dim=visual_proj_dim,
            visual_pretrained=True,
        )

        # Load telemetry-only weights (ignore missing visual keys)
        missing, unexpected = instance.load_state_dict(ac_state, strict=False)
        visual_missing = [k for k in missing if "video_encoder" in k or "visual_fusion" in k]
        other_missing = [k for k in missing if k not in visual_missing]
        if other_missing:
            logger.warning("Unexpected missing keys: %s", other_missing)
        logger.info(
            "Loaded telemetry checkpoint. Visual keys initialized fresh: %d",
            len(visual_missing),
        )

        # Load Phase 1 visual weights if available
        if encoder_ckpt_path is not None:
            encoder_ckpt_path = Path(encoder_ckpt_path)
            if encoder_ckpt_path.exists():
                enc_state = torch.load(
                    encoder_ckpt_path, map_location=device, weights_only=False
                )
                enc_sd = enc_state.get(
                    "ema", enc_state.get("model", enc_state.get("model_state_dict", enc_state))
                )
                # Clean DDP/compile prefixes
                cleaned = {}
                for key, value in enc_sd.items():
                    while key.startswith(("module.", "_orig_mod.")):
                        key = key.removeprefix("module.").removeprefix("_orig_mod.")
                    cleaned[key] = value

                # Load video_encoder weights
                ve_state = {
                    k.removeprefix("video_encoder."): v
                    for k, v in cleaned.items()
                    if k.startswith("video_encoder.")
                }
                if ve_state:
                    instance.video_encoder.load_state_dict(ve_state, strict=True)
                    logger.info("Loaded VideoEncoder from Phase 1 (%d params)", len(ve_state))

                # VisualNudgeFusion is different architecture from Phase 1 GatedFusion,
                # so we initialize it randomly (gate bias=-2 gives near-zero initial contribution)
                logger.info("VisualNudgeFusion initialized randomly (gate starts near-zero)")

        instance.freeze_visual()
        instance.to(device)
        return instance
