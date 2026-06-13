#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// Macro for checking CUDA errors safely
#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(-1); }
// ==============================================================================
// THE KERNEL: Hand-Rolled Tiled Matrix Multiplication 
// Based on Section 2.6 of "Comparing Matrix Multiplication Algorithms"
// ==============================================================================
__global__ void handRolledTiledMatMul(const float* A, const float* B, float* C, int N) {
    
    // --------------------------------------------------------------------------
    // PHASE 1: SHARED MEMORY ALLOCATION
    // The researchers explicitly chose a K=32 tile size. 
    // This creates 32x32 = 1024 threads per block, maximizing the SM occupancy.
    // We allocate two 2D arrays in 'Shared Memory'—a blazing fast, low-latency 
    // memory bank built directly into the GPU chip, shared by all 1024 threads.
    // --------------------------------------------------------------------------
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    // --------------------------------------------------------------------------
    // PHASE 2: THREAD & DATA MAPPING
    // Every thread needs to know its specific job. 
    // tx and ty: The thread's local X and Y coordinates inside its 32x32 block.
    // row and col: The exact global row and column of the massive N x N matrix 
    // that this specific thread is responsible for calculating.
    // --------------------------------------------------------------------------
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;

    // --------------------------------------------------------------------------
    // PHASE 3: REGISTER ACCUMULATOR
    // We store the running sum of the dot product in a local variable 'value'.
    // Local variables in CUDA are stored in the thread's "Registers"—the absolute 
    // fastest memory on the GPU. This avoids writing partial sums back to slow global RAM.
    // --------------------------------------------------------------------------
    float value = 0.0f;

    // Calculate how many 32x32 tiles it takes to cross the N x N matrix.
    // Adding 31 before dividing by 32 ensures we round up for odd-sized matrices.
    int num_tiles = (N + 31) / 32; 
    
    // --------------------------------------------------------------------------
    // PHASE 4: THE TILE SWIPE (Looping through the matrices tile by tile)
    // --------------------------------------------------------------------------
    for (int k_step = 0; k_step < num_tiles; ++k_step) {
        
        // --- MEMORY COALESCING & CONTROL DIVERGENCE (ZERO PADDING) ---
        // The threads work together to pull a tile from Matrix A and B into shared memory.
        // If the N x N matrix doesn't fit perfectly into 32x32 blocks, some threads on 
        // the edge will try to read out-of-bounds memory. The 'if' statements prevent this 
        // by loading 0.0 instead (Control Divergence). 
        // Because data is stored in Row-Major order, consecutive threads reading 
        // consecutive columns triggers "Memory Coalescing," turning 32 slow RAM 
        // requests into 1 massive, fast hardware burst.
        if (row < N && (k_step * 32 + tx) < N) {
            As[ty][tx] = A[row * N + (k_step * 32 + tx)]; 
        } else {
            As[ty][tx] = 0.0f; 
        }

        if ((k_step * 32 + ty) < N && col < N) {
            Bs[ty][tx] = B[(k_step * 32 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // --- HARDWARE SYNCHRONIZATION 1 ---
        // We cannot do any math until every single thread in the block has safely 
        // copied its assigned number into the shared cache. This forces fast threads 
        // to wait for slow threads.
        __syncthreads();

        // --- DOT PRODUCT COMPUTATION ---
        // Now that the tile is in the ultra-fast shared cache, the thread calculates
        // the dot product for its assigned cell. Notice it reads from As and Bs (cache),
        // completely bypassing the slow global GPU RAM.
        for (int k = 0; k < 32; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }

        // --- HARDWARE SYNCHRONIZATION 2 ---
        // Before the loop restarts to load the next tile, we must guarantee that 
        // all threads have finished using the current numbers. If we don't sync here, 
        // fast threads might overwrite the shared memory before slow threads finish their math.
        __syncthreads();
    }

    // --------------------------------------------------------------------------
    // PHASE 5: FINAL MEMORY WRITE
    // Once all tiles have been swiped and the final 'value' is computed, 
    // the thread writes the answer to Matrix C in the global GPU RAM.
    // The boundary check ensures ghost threads (from the zero-padding phase) 
    // do not corrupt memory outside the bounds of the actual matrix.
    // --------------------------------------------------------------------------
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// ==============================================================================
// MAIN HOST FUNCTION
// ==============================================================================
int main() {
    // Define matrix size (N x N) as proposed in the paper's max limit
    const int N = 10000; 
    const size_t matrix_size = (size_t)N * N * sizeof(float);

    std::cout << "Initializing " << N << "x" << N << " matrices (Single Precision F32)..." << std::endl;

    // Allocate Host Memory (RAM)
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N, 0.0f);

    // Populate matrices with uniform random numbers between 2.0 and 5.0 [cite: 331]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(2.0f, 5.0f);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    // Allocate Device Memory (GPU VRAM)
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, matrix_size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, matrix_size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, matrix_size));

    // Copy data from Host to Device (Excluding this from the FLOPS benchmark stopwatch) [cite: 334, 335]
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrix_size, cudaMemcpyHostToDevice));

    // Define CUDA Grid and Block Dimensions 
    // The paper strictly uses a 32x32 block size, yielding 1024 threads per block.
    dim3 threadsPerBlock(32, 32);
    // Calculate how many blocks are needed to cover the entire N x N grid
    dim3 blocksPerGrid((N + 31) / 32, (N + 31) / 32);

    // Setup CUDA timing events to measure ARITHMETIC calculation alone [cite: 336]
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Running Hand-Rolled CUDA Matrix Multiplication..." << std::endl;

    // Start Clock
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch the custom kernel
    handRolledTiledMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Stop Clock
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop)); // Ensure GPU completes calculation

    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double seconds = milliseconds / 1000.0;

    // Calculate total Floating-Point Operations using the paper's explicit formula: 2N^3 - N^2 [cite: 346, 347]
    double total_ops = 2.0 * N * N * N - (double)N * N;
    double gflops = (total_ops / seconds) / 1e9;

    std::cout << "\nBenchmark Results" << std::endl;
    std::cout << "Time spent on calculation: " << seconds << " seconds" << std::endl;
    std::cout << "Achieved Performance: " << gflops << " GFLOPS" << std::endl;

    // Clean up memory
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}