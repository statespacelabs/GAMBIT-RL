import torch
import torch.nn as nn
from .types import TransformerConfig

class TemporalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.d_model = config.d_model
        
        # Learnable [STYLE] token parameter [1, 1, 512] initialized to zeros
        self.style_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        
        # Learned positional embeddings [1, 151, 512] initialized from truncated normal (std=0.02)
        self.pos_embed = nn.Parameter(torch.empty(1, config.max_seq_len, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 6-layer TransformerEncoder (d_model=512, 8 heads, ffn=2048, GELU, dropout=0.1, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 512]
        B, T, _ = x.shape
        if T + 1 > self.pos_embed.shape[1]:
            raise ValueError(
                f"Sequence length {T} exceeds configured max of {self.pos_embed.shape[1] - 1}"
            )

        # Prepend style token
        style_tokens = self.style_token.expand(B, -1, -1)
        x = torch.cat((style_tokens, x), dim=1) # [B, T+1, 512]
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :T+1, :]
        
        # Run transformer
        x = self.transformer(x)
        
        # Extract position 0 for the representation (unnormalized z_raw)
        z_raw = x[:, 0, :] # [B, 512]
        
        return z_raw
