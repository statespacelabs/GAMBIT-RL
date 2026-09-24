import torch
from dataclasses import dataclass, field

@dataclass
class EncoderOutput:
    z_raw: torch.Tensor  # [B, 512] - unnormalized style embedding from transformer
    z_norm: torch.Tensor # [B, 512] - L2-normalized version for InfoNCE and retrieval
    p: torch.Tensor      # [B, p_dim] - projector output for VC regularization

@dataclass
class ClipBatch:
    video: torch.Tensor       # [B, T, C, H, W]
    where_tel: torch.Tensor   # [B, T, 10]
    view_tel: torch.Tensor    # [B, T, 13]
    rhythm_tel: torch.Tensor  # [B, T, A+8] where A is action vocab size
    player_ids: torch.Tensor  # [B] - identifiers for players
    clip_ids: list[str] | None = None
    session_ids: list[str] | None = None
    # frame and time explicitly excluded per spec

    def to(self, device, non_blocking=False):
        return ClipBatch(
            video=self.video.to(device, non_blocking=non_blocking),
            where_tel=self.where_tel.to(device, non_blocking=non_blocking),
            view_tel=self.view_tel.to(device, non_blocking=non_blocking),
            rhythm_tel=self.rhythm_tel.to(device, non_blocking=non_blocking),
            player_ids=self.player_ids.to(device, non_blocking=non_blocking),
            clip_ids=self.clip_ids,
            session_ids=self.session_ids,
        )

@dataclass
class TelemetryConfig:
    where_dim: int = 10
    view_dim: int = 13
    rhythm_dim: int = 22  # action_vocab_size=14, so 14 + 8 = 22
    hidden_dim: int = 64
    out_dim: int = 128

@dataclass
class TransformerConfig:
    d_model: int = 512
    nhead: int = 16
    num_layers: int = 8
    dim_feedforward: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 151 # 150 frames + 1 style token

@dataclass
class EncoderConfig:
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    video_proj_dim: int = 512
    fused_dim: int = 1024
    projector_dim: int = 4096
    video_pretrained: bool = True
