import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, tel_dim: int = 256, video_dim: int = 512):
        super().__init__()
        self.tel_proj = nn.Linear(tel_dim, video_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(video_dim * 2, video_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_net[0].bias, -1.0)
        self.last_gate: torch.Tensor | None = None

    def forward(self, video_features: torch.Tensor, tel_features: torch.Tensor) -> torch.Tensor:
        # video_features: [B, T, 512]
        # tel_features: [B, T, 256]
        tel_proj = self.tel_proj(tel_features)
        gate = self.gate_net(torch.cat([video_features, tel_proj], dim=-1))
        self.last_gate = gate.detach()
        return video_features + gate * tel_proj

    def gate_mean(self) -> torch.Tensor | None:
        if self.last_gate is None:
            return None
        return self.last_gate.mean()
