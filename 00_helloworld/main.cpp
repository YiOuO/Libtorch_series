#include <torch/torch.h>
#include <iostream>

int main() {
    // 在 CPU 上创建张量
    torch::Tensor tensor_cpu = torch::randn({3, 3});  // 生成一个 3x3 的张量
    std::cout << "Tensor on CPU:\n" << tensor_cpu << std::endl;
    std::cout << "Device of tensor_cpu: " << tensor_cpu.device() << std::endl;

    // 在 GPU 上创建张量（假设有可用的 GPU）
    if (torch::cuda::is_available()) {
        torch::Tensor tensor_gpu = torch::randn({3, 3}, torch::device(torch::kCUDA));  // 在 GPU 上生成张量
        std::cout << "\nTensor on GPU:\n" << tensor_gpu << std::endl;
        std::cout << "Device of tensor_gpu: " << tensor_gpu.device() << std::endl;
    } else {
        std::cout << "\nCUDA is not available. Skipping GPU tensor creation." << std::endl;
    }

    return 0;
}
