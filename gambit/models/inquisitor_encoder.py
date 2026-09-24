import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import EncoderConfig, ClipBatch, EncoderOutput
from .video_encoder import VideoEncoder
from .where_branch import WhereBranch
from .view_branch import ViewBranch
from .rhythm_branch import RhythmBranch
from .telemetry_fusion import TelemetryFusion
from .temporal_tcn import TemporalTCN
from .gated_fusion import GatedFusion
from .temporal_transformer import TemporalTransformer
from .projector_head import ProjectorHead

class InquisitorEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        if config.fused_dim != config.video_proj_dim * 2:
            raise ValueError(
                f"fused_dim must equal 2 * video_proj_dim for gated fusion, got {config.fused_dim}"
            )

        # Modality Encoders
        self.video_encoder = VideoEncoder(
            proj_dim=config.video_proj_dim,
            pretrained=config.video_pretrained,
        )
        
        self.where_branch = WhereBranch(
            input_dim=config.telemetry.where_dim,
            hidden_dim=config.telemetry.hidden_dim,
            out_dim=config.telemetry.out_dim
        )
        self.view_branch = ViewBranch(
            input_dim=config.telemetry.view_dim,
            hidden_dim=config.telemetry.hidden_dim,
            out_dim=config.telemetry.out_dim
        )
        self.rhythm_branch = RhythmBranch(
            input_dim=config.telemetry.rhythm_dim,
            hidden_dim=config.telemetry.hidden_dim,
            out_dim=config.telemetry.out_dim
        )
        
        # Fusion & Temporal
        self.telemetry_fusion = TelemetryFusion(
            in_features=config.telemetry.out_dim * 3,
            out_features=256
        )
        
        self.temporal_tcn = TemporalTCN(channels=256)
        
        self.gated_fusion = GatedFusion(
            tel_dim=256,
            video_dim=config.video_proj_dim,
        )
        
        self.temporal_transformer = TemporalTransformer(config=config.transformer)
        
        # Projector Head
        self.projector_head = ProjectorHead(
            in_dim=config.transformer.d_model,
            p_dim=config.projector_dim
        )
        
    def forward(self, batch: ClipBatch) -> EncoderOutput:
        # Assuming video is [B, T, C, H, W], we fold B and T for VideoEncoder
        B, T, C, H, W = batch.video.shape
        video_flat = batch.video.reshape(B * T, C, H, W)
        video_tokens_flat = self.video_encoder(video_flat)
        video_tokens = video_tokens_flat.reshape(B, T, -1)
        
        where_out = self.where_branch(batch.where_tel)
        view_out = self.view_branch(batch.view_tel)
        rhythm_out = self.rhythm_branch(batch.rhythm_tel)
        
        tel = self.telemetry_fusion(where_out, view_out, rhythm_out)
        tel = self.temporal_tcn(tel)
        
        fused = self.gated_fusion(video_tokens, tel)
        
        z_raw = self.temporal_transformer(fused) # Raw, no normalize
        z_norm = F.normalize(z_raw, dim=-1) # Normalization happens here
        p = self.projector_head(z_raw) # Projector takes raw

        return EncoderOutput(z_raw=z_raw, z_norm=z_norm, p=p)
        
    def get_embedding(self, batch: ClipBatch, normalize: bool = True) -> torch.Tensor:
        out = self.forward(batch)
        return out.z_norm if normalize else out.z_raw
        
    def parameter_groups(self):
        # Separate parameter groups for different learning rates
        return [
            {"params": self.video_encoder.backbone.parameters(), "name": "backbone"},
            {"params": self.video_encoder.proj.parameters(), "name": "video_projection"},
            {"params": list(self.where_branch.parameters()) + 
                       list(self.view_branch.parameters()) + 
                       list(self.rhythm_branch.parameters()) + 
                       list(self.telemetry_fusion.parameters()) + 
                       list(self.temporal_tcn.parameters()) + 
                       list(self.gated_fusion.parameters()), "name": "telemetry_and_fusion"},
            {"params": self.temporal_transformer.parameters(), "name": "transformer"},
            {"params": self.projector_head.parameters(), "name": "projector"}
        ]
