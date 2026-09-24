import torch
import torch.nn as nn

class TelemetryFusion(nn.Module):
    def __init__(self, in_features: int = 384, out_features: int = 256):
        super().__init__()
        # Concat three [B, T, 128] branches along feature dim -> [B, T, 384]
        # Then linear(384, 256) -> LayerNorm -> GELU -> [B, T, 256]
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )
        
    def forward(self, where_out: torch.Tensor, view_out: torch.Tensor, rhythm_out: torch.Tensor) -> torch.Tensor:
        # Concatenate along the feature dimension (dim=-1)
        fused = torch.cat([where_out, view_out, rhythm_out], dim=-1) # [B, T, 384]
        return self.net(fused)
