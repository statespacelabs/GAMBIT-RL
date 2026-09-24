import torch
import torch.nn as nn
import torchvision.models as models


class VideoEncoder(nn.Module):
    def __init__(self, proj_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_tiny(weights=weights)
        self.backbone = convnext.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(768, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.is_frozen = False

    def freeze_backbone(self):
        self.is_frozen = True
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        self.is_frozen = False
        self.backbone.train(self.training)
        for param in self.backbone.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.is_frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"VideoEncoder expects [N, C, H, W], got shape {tuple(x.shape)}")

        if self.is_frozen:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        pooled = self.avgpool(features).flatten(1)
        return self.proj(pooled)
