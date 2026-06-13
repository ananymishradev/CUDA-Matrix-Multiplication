#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(-1); }

__global__ void handRolledTiledMatMul(const double* A, const double* B, double* C, int N) {
    __shared__ double As[32][32];
    __shared__ double Bs[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;

    double value = 0.0;

    int num_tiles = (N + 31) / 32;

    for (int k_step = 0; k_step < num_tiles; ++k_step) {
        if (row < N && (k_step * 32 + tx) < N) {
            As[ty][tx] = A[row * N + (k_step * 32 + tx)];
        } else {
            As[ty][tx] = 0.0;
        }

        if ((k_step * 32 + ty) < N && col < N) {
            Bs[ty][tx] = B[(k_step * 32 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < 32; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

int main() {
    const int N = 10000;
    const size_t matrix_size = (size_t)N * N * sizeof(double);

    std::cout << "Initializing " << N << "x" << N << " matrices (Double Precision F64)..." << std::endl;

    std::vector<double> h_A(N * N);
    std::vector<double> h_B(N * N);
    std::vector<double> h_C(N * N, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(2.0, 5.0);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, matrix_size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, matrix_size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, matrix_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrix_size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + 31) / 32, (N + 31) / 32);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Running Hand-Rolled CUDA Matrix Multiplication..." << std::endl;

    CUDA_CHECK(cudaEventRecord(start, 0));

    handRolledTiledMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double seconds = milliseconds / 1000.0;

    double total_ops = 2.0 * N * N * N - (double)N * N;
    double gflops = (total_ops / seconds) / 1e9;

    std::cout << "\nBenchmark Results" << std::endl;
    std::cout << "Time spent on calculation: " << seconds << " seconds" << std::endl;
    std::cout << "Achieved Performance: " << gflops << " GFLOPS" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
