import torch

from ...cutotune import CutoTuneParameter
from ...utils import ensure_contiguous
from .backward import _backward
from .forward import _forward
from .torch_implementation import layernorm_torch


class _LayerNorm_Cute(torch.autograd.Function):
    @staticmethod
    # @ensure_contiguous
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float | None,
        kernel_backend_forward: str,
        kernel_backend_backward: str,
        BLOCK_SIZE_forward: int,
    ) -> torch.Tensor:
        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        is_x_1d = x.dim() == 1
        if is_x_1d:
            x = x.unsqueeze(0)

        if eps is None:
            eps = torch.finfo(x.dtype).eps
        
        output, mean, rstd = _forward(
            x=x,
            weight=weight,
            eps=eps,
            bias=bias,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE=BLOCK_SIZE_forward,
        )

        if is_x_1d:
            output = output.squeeze(0)

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.is_x_1d = is_x_1d
        ctx.kernel_backend_backward = kernel_backend_backward
        return output

    @staticmethod
    # @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        x, weight, bias, mean, rstd = ctx.saved_tensors

        x_grad, weight_grad, bias_grad = _backward(
            x=x,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            output_grad=output_grad,
            kernel_backend=ctx.kernel_backend_backward,
        )

        if ctx.is_x_1d:
            x_grad = x_grad.squeeze(0)

        return x_grad, weight_grad, bias_grad, None, None, None, None, None

def layernorm_cute(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float | None,
    kernel_backend_forward: str,
    kernel_backend_backward: str,
    BLOCK_SIZE_forward: int,
) -> torch.Tensor:
    return _LayerNorm_Cute.apply(
        x,
        weight,
        bias,
        eps,
        kernel_backend_forward,
        kernel_backend_backward,
        BLOCK_SIZE_forward,
    )
