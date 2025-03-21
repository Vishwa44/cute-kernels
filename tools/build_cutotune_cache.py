from typing import Callable

import torch

from cute_kernels import (
    add_scalar_cute,
    add_tensor_cute,
    continuous_count_cute,
    cross_entropy_cute,
    embedding_cute,
    fused_residual_add_rmsnorm_cute,
    gemm_cute,
    get_powers_of_2,
    rmsnorm_cute,
    save_cutotune_cache,
    set_seed,
    softmax_cute,
    swiglu_cute,
    swiglu_unchunked_cute,
)


def forward_backward(kernel: Callable, *args, **kwargs) -> None:
    output = kernel(*args, **kwargs)
    output.sum().backward()


set_seed(42)
all_dtypes = [torch.float32, torch.float16, torch.bfloat16]


for dtype in all_dtypes:
    size = 104857600
    x = torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)

    forward_backward(add_scalar_cute, x, 3)
    forward_backward(add_tensor_cute, x, x)
    forward_backward(swiglu_cute, x, x)

    size = (81920, 8192)
    forward_backward(
        swiglu_unchunked_cute,
        torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True),
    )

    for power_of_2 in get_powers_of_2(1, 65536):
        size = (2048, power_of_2)
        x = torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)

        forward_backward(rmsnorm_cute, x, weight=None, eps=1e-5)
        forward_backward(
            rmsnorm_cute,
            x,
            weight=torch.randn(x.size(-1), dtype=dtype, device=torch.cuda.current_device(), requires_grad=True),
            eps=1e-5,
        )

        forward_backward(fused_residual_add_rmsnorm_cute, x, x, weight=None, eps=1e-5)
        forward_backward(
            fused_residual_add_rmsnorm_cute,
            x,
            x,
            weight=torch.randn(x.size(-1), dtype=dtype, device=torch.cuda.current_device(), requires_grad=True),
            eps=1e-5,
        )

    size = (81920, 8192)
    x = torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)

    forward_backward(softmax_cute, x)

    input_ids_size = (32, 4096)
    weight_size = (131072, 4096)
    forward_backward(
        embedding_cute,
        input_ids=torch.randint(
            0, weight_size[0] - 1, input_ids_size, device=torch.cuda.current_device(), dtype=torch.long
        ),
        weight=torch.randn(weight_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
    )

    input_size = (4096, 4096)
    weight_size = (4096, 4096)

    for is_A_transposed in [False, True]:
        for is_B_transposed in [False, True]:
            gemm_cute(
                A=torch.randn(*input_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
                B=torch.randn(*weight_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
                C=None,
                is_A_transposed=is_A_transposed,
                is_B_transposed=is_B_transposed,
                beta=0,
            )

    size = (16384, 4096)
    x = torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)
    labels = torch.randint(0, size[1], (x.size(0),), device=torch.cuda.current_device())
    forward_backward(cross_entropy_cute, x=x, labels=labels)

size = 104857600
for dtype in [torch.long, torch.int32]:
    for n in get_powers_of_2(1, 16384):
        x = torch.randint(0, n, (size,), dtype=dtype, device=torch.cuda.current_device())
        continuous_count_cute(x, n)


save_cutotune_cache()
