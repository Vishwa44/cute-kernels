import torch
import torch.nn.functional as F

def layernorm_torch(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    return F.layer_norm(x, weight.shape, weight=weight, bias=bias, eps=eps)