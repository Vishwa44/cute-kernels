#include <torch/extension.h>

void contiguous_count_cuda(const torch::Tensor &x,
                           torch::Tensor &output,
                           const uint &sm_count,
                           const uint &thread_block_cluster_size,
                           const uint &C,
                           const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("contiguous_count_cuda", &contiguous_count_cuda, "contiguous count forward (CUDA)");
}
