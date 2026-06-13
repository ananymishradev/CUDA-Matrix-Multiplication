#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// Macro for checking CUDA errors safely
#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(-1); }

// ==============================================================================
// THE KERNEL: TIER 3 - Coarse-Grained Thread Tiling (Register Re-Use)
// ==============================================================================
__global__ void handRolledTiledMatMul_Opt(const float* A, const float* B, float* C, int N) {
    
    // We maintain the 32x32 memory footprint in Shared Memory
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Base row and col for this specific 32x32 block footprint
    int row_start = blockIdx.y * 32;
    int col_start = blockIdx.x * 32;

    // The specific global column this thread handles
    int col = col_start + tx;

    // --- REGISTER ACCUMULATION ---
    // Instead of 1 value, this thread holds 4 values in its ultra-fast registers.
    // It will calculate a 1x4 vertical slice of the output matrix.
    float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int num_tiles = (N + 31) / 32; 

    for (int k_step = 0; k_step < num_tiles; ++k_step) {
        
        // --- PHASE 1: COOPERATIVE LOAD ---
        // Because we only have 8 threads in the Y direction but need to load 32 rows 
        // into shared memory, each thread must load 4 elements.
        for (int i = 0; i < 4; ++i) {
            int load_y = ty * 4 + i; // The internal Y coordinate in the 32x32 tile
            
            // Load Matrix A
            int global_row_A = row_start + load_y;
            int global_col_A = k_step * 32 + tx;
            if (global_row_A < N && global_col_A < N) {
                As[load_y][tx] = A[global_row_A * N + global_col_A]; 
            } else {
                As[load_y][tx] = 0.0f; 
            }

            // Load Matrix B
            int global_row_B = k_step * 32 + load_y;
            int global_col_B = col_start + tx;
            if (global_row_B < N && global_col_B < N) {
                Bs[load_y][tx] = B[global_row_B * N + global_col_B];
            } else {
                Bs[load_y][tx] = 0.0f;
            }
        }

        __syncthreads();

        // --- PHASE 2: MATH & REGISTER RE-USE ---
        // Compute 4 elements for this thread. 
        for (int k = 0; k < 32; ++k) {
            // READ ONCE: Pull the Matrix B value from Shared Memory into a Register
            float b_val = Bs[k][tx];
            
            // MULTIPLY 4 TIMES: Use that single register value to compute 4 outputs.
            // This cuts our Shared Memory bandwidth requirements nearly in half!
            for (int i = 0; i < 4; ++i) {
                values[i] += As[ty * 4 + i][k] * b_val;
            }
        }

        __syncthreads();
    }

    // --- PHASE 3: FINAL MEMORY WRITE ---
    // Write all 4 calculated register values back to the global VRAM.
    for (int i = 0; i < 4; ++i) {
        int global_row = row_start + ty * 4 + i;
        if (global_row < N && col < N) {
            C[global_row * N + col] = values[i];
        }
    }
}

// ==============================================================================
// MAIN HOST FUNCTION
// ==============================================================================
int main() {
    const int N = 10000; 
    const size_t matrix_size = (size_t)N * N * sizeof(float);

    std::cout << "Initializing " << N << "x" << N << " matrices (Single Precision F32)..." << std::endl;

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(2.0f, 5.0f);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, matrix_size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, matrix_size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, matrix_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrix_size, cudaMemcpyHostToDevice));

    // --- EXECUTION CONFIGURATION UPGRADE ---
    // We are still covering a 32x32 block of the matrix, but we only assign 8 threads 
    // to the Y-axis because each thread does 4 units of work (8 x 4 = 32).
    dim3 threadsPerBlock(32, 8);
    dim3 blocksPerGrid((N + 31) / 32, (N + 31) / 32);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Running..." << std::endl;

    CUDA_CHECK(cudaEventRecord(start, 0));

    handRolledTiledMatMul_Opt<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

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