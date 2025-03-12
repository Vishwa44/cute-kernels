import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import cutotune
from ...math import get_next_power_of_2
from .parameters import get_cutotune_parameters
from ...utils import cute_op, get_num_elements_and_hidden_size
from .triton_implementation import layernorm_backward_triton


def _backward(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    output_grad: torch.Tensor,
    bias: torch.Tensor,
    kernel_backend: str,
) -> tuple[torch.Tensor | None]:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)
    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    x_grad = torch.empty_like(x)

    weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)
    bias_grad = None if bias is None else torch.zeros_like(bias, dtype=torch.float32)

    if kernel_backend == "triton":
        BLOCK_SIZE = get_next_power_of_2(hidden_size)
        assert BLOCK_SIZE <= MAX_TRITON_BLOCK_SIZE

        layernorm_backward_triton(
            x=x,
            weight=weight,
            output_grad=output_grad,
            mean=mean,
            rstd=rstd,
            x_grad=x_grad,
            weight_grad=weight_grad,
            bias_grad=bias_grad,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    if weight_grad is not None:
        weight_grad = weight_grad.type_as(weight)

    if bias_grad is not None:
        bias_grad = bias_grad.type_as(bias)

    return x_grad.reshape(x_shape), weight_grad, bias_grad
