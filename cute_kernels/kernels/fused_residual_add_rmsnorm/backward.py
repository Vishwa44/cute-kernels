import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...cutotune import cutotune
from ...math import get_next_power_of_2
from ..rmsnorm.parameters import get_cutotune_parameters
from .triton_implementation import fused_residual_add_rmsnorm_backward_triton


@cutotune(**get_cutotune_parameters(triggers={"added_x_residual.dtype"}))
def _backward(
    added_x_residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    multiplier: float | None,
    rmsnorm_denominator: torch.Tensor,
    output_grad: torch.Tensor,
    added_x_residual_grad: torch.Tensor,
    kernel_backend: str,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> tuple[torch.Tensor | None]:
    hidden_size = added_x_residual.size(-1)

    x_grad = torch.empty_like(added_x_residual)
    residual_grad = torch.empty_like(added_x_residual)
    weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

    if kernel_backend == "triton":
        BLOCK_SIZE_H = get_next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        fused_residual_add_rmsnorm_backward_triton(
            added_x_residual=added_x_residual,
            weight=weight,
            output_grad=output_grad,
            added_x_residual_grad=added_x_residual_grad,
            rmsnorm_denominator=rmsnorm_denominator,
            x_grad=x_grad,
            residual_grad=residual_grad,
            weight_grad=weight_grad,
            eps=eps,
            multiplier=multiplier,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    if weight_grad is not None:
        weight_grad = weight_grad.type_as(weight)

    return x_grad, residual_grad, weight_grad
