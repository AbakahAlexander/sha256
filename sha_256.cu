// cuda_sha256_benchmark.cu
// CUDA SHA-256 Hash Cracker Benchmark - With Performance Measurement

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>


#define THREADS_PER_BLOCK 256
#define BLOCKS 128
#define ITERATIONS 10  

// SHA-256 Constants, out in constant memory for fast read access by all threads
//derived from the fractional parts of the square roots of the first 64 primes
//formula: ⎣2³√n × 2³²⎦, where n is the prime number in the sequence
//found it in module 2^32 and conevrted it hex
//https://en.wikipedia.org/wiki/SHA-2#Pseudocode

__device__ __constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

//shift left by n bits and then OR with right shift by 32-n bits to rotate
__device__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

//transform 512 bits at a time of the entire message
//each block is divided into 16 32-bit words
__device__ void sha256_transform(const uint8_t *data, uint32_t *hash_out) {
    //hold the data to be transformed
    uint32_t w[64];

    //unroll the loop to reduce overhead. remove the idea of checking condition in loop
    //and then incrementing the counter
    //this will make the loop run faster
    #pragma unroll

    //process the 16 words in the block
    //one word at a time
    for (int i = 0; i < 16; i++) {
        w[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) |
               (data[i * 4 + 2] << 8) | data[i * 4 + 3];
    }

    //extend the first 16 words into the remaining 48 words
    //of the message schedule array
    //uses the XOR operation ^ to combine the words
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;


    //each 512-bit block is processed 64 times
    //each time the block is processed, the values of a-h are updated
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    hash_out[0] = a;
    hash_out[1] = b;
    hash_out[2] = c;
    hash_out[3] = d;
    hash_out[4] = e;
    hash_out[5] = f;
    hash_out[6] = g;
    hash_out[7] = h;
}

__global__ void sha256_kernel(uint8_t *inputs, uint32_t *hashes, int num_inputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_inputs) {
        sha256_transform(&inputs[idx * 64], &hashes[idx * 8]);
    }
}

int main() {
    int num_inputs = THREADS_PER_BLOCK * BLOCKS;
    size_t input_size = num_inputs * 64 * sizeof(uint8_t);
    size_t output_size = num_inputs * 8 * sizeof(uint32_t);

    uint8_t *h_inputs = (uint8_t*)malloc(input_size);
    uint32_t *h_hashes = (uint32_t*)malloc(output_size);

    for (int i = 0; i < num_inputs; i++) {
        memset(&h_inputs[i * 64], i % 256, 64);
    }

    uint8_t *d_inputs;
    uint32_t *d_hashes;

    cudaMalloc(&d_inputs, input_size);
    cudaMalloc(&d_hashes, output_size);

    cudaMemcpy(d_inputs, h_inputs, input_size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Timing variables for overall benchmark
    clock_t total_start, total_end;
    double total_elapsed_ms = 0.0;
    float kernel_elapsed_ms = 0.0;
    float avg_kernel_ms = 0.0;
    
    printf("Running SHA-256 benchmark with %d blocks x %d threads = %d total hashes\n", 
           BLOCKS, THREADS_PER_BLOCK, num_inputs);
    printf("Performing %d iterations for average results...\n\n", ITERATIONS);
    
    // Start total time measurement
    total_start = clock();
    
    // Run multiple iterations for better averaging
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Record kernel start time
        cudaEventRecord(start);
        
        // Launch the kernel
        sha256_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_inputs, d_hashes, num_inputs);
        
        // Record kernel end time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iter_elapsed_ms = 0.0f;
        cudaEventElapsedTime(&iter_elapsed_ms, start, stop);
        kernel_elapsed_ms += iter_elapsed_ms;
        
        // Check for errors
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            return -1;
        }
    }
    
    // Calculate average kernel time
    avg_kernel_ms = kernel_elapsed_ms / ITERATIONS;
    
    // Copy results back
    cudaMemcpy(h_hashes, d_hashes, output_size, cudaMemcpyDeviceToHost);
    
    // Calculate total elapsed time
    total_end = clock();
    total_elapsed_ms = (double)(total_end - total_start) * 1000.0 / CLOCKS_PER_SEC;
    
    // Display sample hashes
    printf("Sample hashes:\n");
    for (int i = 0; i < 3; i++) {
        printf("Hash %d: ", i);
        for (int j = 0; j < 8; j++) {
            printf("%08x ", h_hashes[i * 8 + j]);
        }
        printf("\n");
    }
    
    // Calculate performance
    double kernel_hashes_per_sec = (double)num_inputs * ITERATIONS * 1000.0 / kernel_elapsed_ms;
    double total_hashes_per_sec = (double)num_inputs * ITERATIONS * 1000.0 / total_elapsed_ms;
    
    // Display benchmark results
    printf("\nPerformance Results:\n");
    printf("-------------------\n");
    printf("Total hashes computed: %d\n", num_inputs * ITERATIONS);
    printf("GPU kernel time (avg): %.2f ms per %d hashes\n", avg_kernel_ms, num_inputs);
    printf("Total processing time: %.2f ms\n", total_elapsed_ms);
    printf("\nThroughput:\n");
    printf("  GPU kernel only:     %.2f million hashes/second\n", kernel_hashes_per_sec / 1000000.0);
    printf("  Including overhead:  %.2f million hashes/second\n", total_hashes_per_sec / 1000000.0);
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_inputs);
    cudaFree(d_hashes);
    free(h_inputs);
    free(h_hashes);

    return 0;
}
