import torch
import torch.nn as nn

class ProjectorHead(nn.Module):
    def __init__(self, in_dim: int = 512, p_dim: int = 4096):
        super().__init__()
        # Architecture: Linear(512 → 2048), GELU, LayerNorm, 
        # Linear(2048 → 4096), GELU, LayerNorm, Linear(4096 → p_dim). No activation after the final layer.
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.LayerNorm(4096),
            nn.Linear(4096, p_dim)
        )
        
    def forward(self, z_raw: torch.Tensor) -> torch.Tensor:
        return self.net(z_raw)
