import torch
import torch.distributed as dist

def concat_all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensor from all ranks while preserving gradients.
    """
    if not dist.is_available() or not dist.is_initialized():
        return x
    
    # Use PyTorch's distributed autograd-aware gather
    from torch.distributed.nn.functional import all_gather
    gathered = all_gather(x)
    return torch.cat(list(gathered), dim=0)

def concat_all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensor from all ranks without gradient.
    Useful for labels/player_ids.
    """
    if not dist.is_available() or not dist.is_initialized():
        return x

    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)
