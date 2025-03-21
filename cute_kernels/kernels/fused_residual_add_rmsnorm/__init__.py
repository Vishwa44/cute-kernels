import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .backward import _backward
from .forward import _forward
from .torch_implementation import fused_residual_add_rmsnorm_torch


class _FusedResidualAddRMSNorm_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
        kernel_backend_forward: str,
        kernel_backend_backward: str,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> tuple[torch.Tensor]:
        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        is_x_1d = x.dim() == 1
        if is_x_1d:
            x = x.unsqueeze(0)

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        output, added_x_residual, rmsnorm_denominator = _forward(
            x=x,
            residual=residual,
            weight=weight,
            eps=eps,
            multiplier=multiplier,
            memory_efficient=memory_efficient,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
        )

        ctx.save_for_backward(added_x_residual, weight, rmsnorm_denominator)

        if is_x_1d:
            output = output.squeeze(0)
            added_x_residual = added_x_residual.squeeze(0)

        ctx.is_x_1d = is_x_1d
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.eps = eps
        ctx.multiplier = multiplier
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output, added_x_residual

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor, added_x_residual_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        added_x_residual, weight, rmsnorm_denominator = ctx.saved_tensors

        x_grad, residual_grad, weight_grad = _backward(
            added_x_residual=added_x_residual,
            weight=weight,
            eps=ctx.eps,
            multiplier=ctx.multiplier,
            rmsnorm_denominator=rmsnorm_denominator,
            output_grad=output_grad,
            added_x_residual_grad=added_x_residual_grad,
            kernel_backend=ctx.kernel_backend_backward,
            BLOCK_SIZE_B=ctx.BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H=ctx.BLOCK_SIZE_H_backward,
        )

        if ctx.is_x_1d:
            x_grad = x_grad.squeeze(0)
            residual_grad = residual_grad.squeeze(0)

        return x_grad, residual_grad, weight_grad, *[None] * 9


def fused_residual_add_rmsnorm_cute(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None = None,
    memory_efficient: bool = False,
    kernel_backend_forward: str = CutoTuneParameter(),
    kernel_backend_backward: str = CutoTuneParameter(),
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
) -> tuple[torch.Tensor]:
    return _FusedResidualAddRMSNorm_Cute.apply(
        x,
        residual,
        weight,
        eps,
        multiplier,
        memory_efficient,
        kernel_backend_forward,
        kernel_backend_backward,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_forward,
        BLOCK_SIZE_H_backward,
    )
