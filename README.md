# High Performance Matrix Multiplication

CUDA implementations for benchmarking matrix multiplication algorithms, based on the research paper *"Comparing Matrix Multiplication Algorithms"* ([arXiv:2509.04594](https://arxiv.org/abs/2509.04594)).

This project evaluates performance (GFLOPS) across these implementations:

| Implementation | Precision | Description |
|---|---|---|
| **cuBLAS F32** | Single | NVIDIA's optimized library, `cublasSgemm` |
| **Custom CUDA Kernel F32** | Single | Hand-rolled tiled GPU kernel (32×32 tiles) |
| **cuBLAS F64** | Double | NVIDIA's optimized library, `cublasDgemm` |
| **Custom CUDA Kernel F64** | Double | Hand-rolled tiled GPU kernel (32×32 tiles) |

## Prerequisites

- NVIDIA CUDA Toolkit (with `nvcc`)
- cuBLAS (included with CUDA Toolkit)

## Project Structure

```
src/
├── cublas_matmul_f32.cu        # cuBLAS F32 (single precision)
├── cublas_matmul_f64.cu        # cuBLAS F64 (double precision)
├── hand_rolled_cuda_f32.cu     # Custom tiled kernel F32
└── hand_rolled_cuda_f64.cu     # Custom tiled kernel F64
```

## Build & Run

### cuBLAS F32

```bash
nvcc -O3 -std=c++20 src/cublas_matmul_f32.cu -o cublas_f32 -lcublas
./cublas_f32
```

### cuBLAS F64

```bash
nvcc -O3 -std=c++20 src/cublas_matmul_f64.cu -o cublas_f64 -lcublas
./cublas_f64
```

### Custom CUDA Kernel F32

```bash
nvcc -O3 -std=c++20 src/hand_rolled_cuda_f32.cu -o custom_f32
./custom_f32
```

### Custom CUDA Kernel F64

```bash
nvcc -O3 -std=c++20 src/hand_rolled_cuda_f64.cu -o custom_f64
./custom_f64
```

## Benchmark Results

N = 10000×10000 matrices, measured on an NVIDIA GPU:

| Implementation | Precision | Time (s) | GFLOPS |
|---|---|---|---|
| cuBLAS | F32 | 0.34 | 5880.03 |
| Custom CUDA Kernel | F32 | 1.14 | 1739.33 |
| cuBLAS | F64 | 14.98 | 133.51 |
| Custom CUDA Kernel | F64 | 14.00 | 142.87 |

## License\

MIT License — see [LICENSE](LICENSE).

## Reference

- Paper: [Comparing Matrix Multiplication Algorithms](https://arxiv.org/abs/2509.04594) (March 2025)
