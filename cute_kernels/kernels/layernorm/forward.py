import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import cutotune
from ...math import get_next_power_of_2
from ...utils import get_num_elements_and_hidden_size
# from .parameters import get_cutotune_parameters
from .triton_implementation import layernorm_forward_triton


# @cutotune(**get_cutotune_parameters())
def _forward(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    kernel_backend: str,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)
    x = x.reshape(-1, x.shape[-1])
    output = torch.empty_like(x)

    if kernel_backend == "triton":
        # BLOCK_SIZE = get_next_power_of_2(hidden_size)
        # assert BLOCK_SIZE <= MAX_TRITON_BLOCK_SIZE
        
        mean = torch.empty(num_elements, device=x.device, dtype=torch.float32)
        rstd = torch.empty(num_elements, device=x.device, dtype=torch.float32)
        layernorm_forward_triton(
            x=x,
            weight=weight,
            bias=bias,
            output=output,
            mean=mean,
            rstd=rstd,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output