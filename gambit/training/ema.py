import copy
import torch
import torch.nn as nn

class ModelEma(nn.Module):
    """
    Lightweight Exponential Moving Average (EMA) for PyTorch models.
    Maintains a shadow copy of the model parameters.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, device: torch.device | None = None):
        super().__init__()
        self.decay = decay
        
        # Strip DDP module wrapper if present
        unwrapped_model = model.module if hasattr(model, "module") else model
        
        # Deep copy to maintain separate parameters
        self.ema_model = copy.deepcopy(unwrapped_model)
        self.ema_model.eval()
        
        if device is not None:
            self.ema_model.to(device)
            
        # Freeze EMA parameters
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters with current model parameters."""
        unwrapped_model = model.module if hasattr(model, "module") else model
        
        model_params = dict(unwrapped_model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())
        
        for name, param in model_params.items():
            if name in ema_params:
                ema_param = ema_params[name]
                if param.dtype.is_floating_point:
                    ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
