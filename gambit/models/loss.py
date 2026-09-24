import torch
import torch.nn as nn
import torch.nn.functional as F


class GambitLoss(nn.Module):
    def __init__(
        self,
        tau: float = 0.1,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        z_raw_var_weight: float = 0.001,
        z_raw_cov_weight: float = 0.001,
    ):
        super().__init__()
        self.tau = tau
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.z_raw_var_weight = z_raw_var_weight
        self.z_raw_cov_weight = z_raw_cov_weight

    @staticmethod
    def vc_ramp_scale(training_progress: float) -> float:
        progress = max(0.0, min(1.0, float(training_progress)))
        if progress <= 0.2:
            return 0.0
        if progress >= 0.6:
            return 1.0
        return (progress - 0.2) / 0.4

    def info_nce_loss(self, z_norm: torch.Tensor, player_ids: torch.Tensor) -> torch.Tensor:
        # z_norm: [B, 512], L2-normalized
        # player_ids: [B]
        B = z_norm.shape[0]
        sim = torch.matmul(z_norm, z_norm.T) / self.tau

        same_player = player_ids[:, None].eq(player_ids[None, :])
        self_mask = torch.eye(B, device=z_norm.device, dtype=torch.bool)
        positive_mask = same_player & ~self_mask
        non_self_mask = ~self_mask

        if not torch.all(positive_mask.sum(dim=1) > 0):
            raise ValueError("Every player must have at least 2 clips in the batch")

        masked_sim = sim.masked_fill(self_mask, -torch.inf)
        row_max = masked_sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim - row_max).masked_fill(self_mask, 0.0)

        numerator = exp_sim.masked_fill(~positive_mask, 0.0).sum(dim=1)
        denominator = exp_sim.masked_fill(~non_self_mask, 0.0).sum(dim=1)
        return -torch.log((numerator + 1e-8) / (denominator + 1e-8)).mean()

    def vc_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, D]
        B, D = x.shape
        if B < 2:
            raise ValueError("VC loss requires at least 2 samples")

        x_centered = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x_centered.var(dim=0, unbiased=True) + 1e-8)
        var_loss = F.relu(1.0 - std).mean()

        cov = (x_centered.T @ x_centered) / (B - 1)
        cov = cov - torch.diag(torch.diag(cov))
        cov_loss = (cov ** 2).sum() / D
        return var_loss, cov_loss

    def forward(
        self,
        z_norm: torch.Tensor,
        z_raw: torch.Tensor,
        p: torch.Tensor,
        player_ids: torch.Tensor,
        training_progress: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        assert z_norm.shape[0] == player_ids.shape[0], (
            "z_norm and player_ids must have the same batch size"
        )
        assert z_raw.ndim == 2, "z_raw must be [B, D]"
        assert p.ndim == 2, "p must be [B, D]"
        
        infonce = self.info_nce_loss(z_norm, player_ids)
        var_p, cov_p = self.vc_loss(p)
        var_z_raw, cov_z_raw = self.vc_loss(z_raw)

        vc_scale = self.vc_ramp_scale(training_progress)
        lambda_var_p = vc_scale * self.var_weight
        lambda_cov_p = vc_scale * self.cov_weight
        lambda_var_z_raw = vc_scale * self.z_raw_var_weight
        lambda_cov_z_raw = vc_scale * self.z_raw_cov_weight

        total = (
            infonce
            + lambda_var_p * var_p
            + lambda_cov_p * cov_p
            + lambda_var_z_raw * var_z_raw
            + lambda_cov_z_raw * cov_z_raw
        )

        return {
            "total": total,
            "infonce": infonce,
            "var_p": var_p,
            "cov_p": cov_p,
            "var_z_raw": var_z_raw,
            "cov_z_raw": cov_z_raw,
            "vc_scale": torch.as_tensor(vc_scale, device=total.device),
            "lambda_var_p": torch.as_tensor(lambda_var_p, device=total.device),
            "lambda_cov_p": torch.as_tensor(lambda_cov_p, device=total.device),
            "lambda_var_z_raw": torch.as_tensor(lambda_var_z_raw, device=total.device),
            "lambda_cov_z_raw": torch.as_tensor(lambda_cov_z_raw, device=total.device),
        }
