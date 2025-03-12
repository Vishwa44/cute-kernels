import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....cutotune import cutotune
from ....math import ceil_divide
from ....utils import cute_op, get_num_elements_and_hidden_size, get_sm_count
from .parameters import get_cutotune_parameters

_KERNEL_NAME = "layernorm_backward_triton"

@triton.jit
def _layer_norm_bwd_dx_fused(DX,
                             DY,
                             DW,
                             DB,
                             X,
                             W,
                             Mean,
                             Rstd,
                             Lock,
                             stride,
                             N,
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,
                         DB,
                         FINAL_DW,
                         FINAL_DB,
                         M,
                         N,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


def layernorm_backward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    output_grad: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    x_grad: torch.Tensor,
    weight_grad: torch.Tensor | None,
    bias_grad: torch.Tensor,
    BLOCK_SIZE: int
) -> None:
    num_elements, hidden_size = get_num_elements_and_hidden_size(x)

    GROUP_SIZE_M = 64
    if hidden_size <= 8192: GROUP_SIZE_M = 96
    if hidden_size <= 4096: GROUP_SIZE_M = 128
    if hidden_size <= 1024: GROUP_SIZE_M = 256

    locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=weight.device)
    _dw = torch.zeros((GROUP_SIZE_M, hidden_size), dtype=x.dtype, device=weight.device)
    _db = torch.zeros((GROUP_SIZE_M, hidden_size), dtype=x.dtype, device=weight.device)

    with torch.device(x.device):
        _layer_norm_bwd_dx_fused[(num_elements, )](
            x_grad, output_grad, _dw, _db, x, weight, mean, rstd, locks,
            x.stride(0), hidden_size,
            BLOCK_SIZE_N=BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M)
        grid = lambda meta: [triton.cdiv(hidden_size, meta['BLOCK_SIZE_N'])]
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, weight_grad, bias_grad, min(GROUP_SIZE_M, num_elements), hidden_size,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128, num_ctas=1)
