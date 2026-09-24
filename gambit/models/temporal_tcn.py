import torch
import torch.nn as nn

class TemporalTCN(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        # Two Conv1d layers with residual connection around the block.
        # Kernel size 5, dilations [1, 2], padding computed to preserve sequence length.
        # LayerNorm + GELU after each conv.
        
        # For kernel=5, dilation=1, padding=(5-1)*1/2 = 2
        # For kernel=5, dilation=2, padding=(5-1)*2/2 = 4
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, dilation=1, padding=2)
        self.ln1 = nn.LayerNorm(channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, dilation=2, padding=4)
        self.ln2 = nn.LayerNorm(channels)
        self.act2 = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        input_time = x.shape[1]

        # TCN expects [B, C, T]
        x_t = x.transpose(1, 2)
        
        # Block 1
        out = self.conv1(x_t)
        # LayerNorm expects [B, T, C], so we transpose back temporarily
        out = self.ln1(out.transpose(1, 2)).transpose(1, 2)
        out = self.act1(out)
        
        # Block 2
        out = self.conv2(out)
        out = self.ln2(out.transpose(1, 2)).transpose(1, 2)
        out = self.act2(out)
        
        # Residual connection
        out = out + x_t
        
        # Return [B, T, C]
        out = out.transpose(1, 2)
        assert out.shape[1] == input_time, (
            f"TemporalTCN changed sequence length from {input_time} to {out.shape[1]}"
        )
        return out
