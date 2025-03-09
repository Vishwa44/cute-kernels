import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....cutotune import cutotune
from ....math import ceil_divide
from ....utils import cute_op, get_num_elements_and_hidden_size
# from .parameters import get_cutotune_parameters

_KERNEL_NAME = "layernorm_forward_triton"

@triton.jit
def _layernorm_forward_triton_kernel(
    X_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    output_ptr += row * stride
    X_ptr += row * stride

    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(weight_ptr + cols, mask=mask)
        b = tl.load(bias_ptr + cols, mask=mask)
        x = tl.load(X_ptr + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b

        # Write output
        tl.store(output_ptr + cols, y, mask=mask)


def layernorm_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    BLOCK_SIZE: int,
) -> None:
    # num_elements, hidden_size = get_num_elements_and_hidden_size(x)
    num_elements, hidden_size = x.shape
    print(num_elements, hidden_size, BLOCK_SIZE)
    if BLOCK_SIZE > hidden_size:
        raise ValueError(f"hidden_size {hidden_size} should be more than the BLOCK_SIZE {BLOCK_SIZE}")
    print("checking")
    with torch.device(x.device):
        _layernorm_forward_triton_kernel[(num_elements, )](
            X_ptr=x,
            output_ptr=output,
            weight_ptr=weight,
            bias_ptr=bias,
            mean_ptr=mean,
            rstd_ptr=rstd,
            eps=eps,
            stride=x.stride(0),
            N=hidden_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
