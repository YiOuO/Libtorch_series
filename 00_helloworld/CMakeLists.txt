cmake_minimum_required(VERSION 3.10)
project(LibTorchExample)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
# 手动指定 libtorch 的路径
set(Torch_DIR "/home/yi/cuda/libtorch/share/cmake/Torch")

# 查找 LibTorch
find_package(Torch REQUIRED)

# Set this at the beginning of your CMakeLists.txt
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
find_package(CUDA REQUIRED)


# 添加可执行文件
add_executable(main main.cpp)

# 链接 LibTorch 和 CUDA
target_link_libraries(main "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES})

# 设置 include 路径
target_include_directories(main PRIVATE ${TORCH_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})