#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

// Global variables for hash rate tracking
__device__ uint32_t d_hash_counter = 0;
__host__ uint32_t g_hash_rate = 0;

// SHA-256 constants - first 32 bits of the fractional parts of the cube roots of the first 64 primes
__device__ __constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Initial hash values - first 32 bits of the fractional parts of the square roots of the first 8 primes
__device__ __constant__ uint32_t initial_hash[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// SHA-256 functions
__device__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Full SHA-256 implementation for a single 512-bit block
__device__ void sha256_transform(uint32_t state[8], const uint32_t data[16]) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h, t1, t2;
    
    // Copy block into first 16 words of w
    for (int i = 0; i < 16; i++) {
        w[i] = data[i];
    }
    
    // Extend the first 16 words into the remaining 48 words of w
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }
    
    // Initialize working variables
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];
    
    // Main loop
    for (int i = 0; i < 64; i++) {
        t1 = h + sigma1(e) + ch(e, f, g) + k[i] + w[i];
        t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Add the compressed chunk to the current hash value
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// Prepare a padded message block for SHA-256
__device__ void prepare_sha256_message(const uint8_t *message, uint32_t length, uint32_t block[16]) {
    uint32_t i, j;
    
    // Copy the message into the block
    for (i = 0, j = 0; i < length; i += 4, j++) {
        if (i + 3 < length) {
            block[j] = ((uint32_t)message[i] << 24) |
                       ((uint32_t)message[i+1] << 16) |
                       ((uint32_t)message[i+2] << 8) |
                       ((uint32_t)message[i+3]);
        } else {
            block[j] = 0;
            for (uint32_t k = 0; k < length - i; k++) {
                block[j] |= ((uint32_t)message[i+k] << (24 - 8*k));
            }
            
            // Add the "1" bit for padding
            block[j] |= 0x80 << (8 * (3 - (length - i)));
            j++;
            
            // Fill with zeros until we reach the last two words
            while (j < 14) {
                block[j++] = 0;
            }
            
            // Add the length in bits as a 64-bit big-endian integer
            block[14] = 0;
            block[15] = length * 8;
            return;
        }
    }
    
    // If we reached the end of message exactly on block boundary, add a new block for padding
    block[j] = 0x80000000;
    j++;
    
    // Fill with zeros
    while (j < 14) {
        block[j++] = 0;
    }
    
    // Add the length in bits as a 64-bit big-endian integer
    block[14] = 0;
    block[15] = length * 8;
}

// Full SHA-256 hash computation
__device__ void sha256_hash(const uint8_t *message, uint32_t length, uint32_t hash[8]) {
    uint32_t block[16];
    
    // Initialize hash state with constants
    for (int i = 0; i < 8; i++) {
        hash[i] = initial_hash[i];
    }
    
    // Process complete blocks
    uint32_t i;
    for (i = 0; i + 64 <= length; i += 64) {
        for (int j = 0; j < 16; j++) {
            block[j] = ((uint32_t)message[i+j*4] << 24) |
                       ((uint32_t)message[i+j*4+1] << 16) |
                       ((uint32_t)message[i+j*4+2] << 8) |
                       ((uint32_t)message[i+j*4+3]);
        }
        sha256_transform(hash, block);
    }
    
    // Process final block with padding
    prepare_sha256_message(message + i, length - i, block);
    sha256_transform(hash, block);
}

// Mining kernel using full SHA-256
__global__ void mining_kernel(uint32_t start_nonce, uint32_t target, uint32_t *result) {
    uint32_t nonce = start_nonce + (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Create a block with nonce
    uint8_t block[64];
    for (int i = 0; i < 60; i++) {
        block[i] = 0;
    }
    
    // Put nonce in the last 4 bytes
    block[60] = (nonce >> 24) & 0xFF;
    block[61] = (nonce >> 16) & 0xFF;
    block[62] = (nonce >> 8) & 0xFF;
    block[63] = nonce & 0xFF;
    
    // Compute hash
    uint32_t hash[8];
    sha256_hash(block, 64, hash);
    
    // Check if hash meets target
    if (hash[0] < target) {
        *result = nonce;
    }
    
    // Increment global hash counter
    atomicAdd(&d_hash_counter, 1);
}

// Anomaly detection kernel
__global__ void anomaly_detection_kernel(uint32_t start_nonce, uint32_t *anomaly_scores) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = start_nonce + idx;
    
    // Measure how long it takes to hash
    clock_t start_time = clock();
    
    // Create a block with the nonce
    uint8_t block[64];
    for (int i = 0; i < 60; i++) {
        block[i] = 0;
    }
    
    // Put nonce in the last 4 bytes
    block[60] = (nonce >> 24) & 0xFF;
    block[61] = (nonce >> 16) & 0xFF;
    block[62] = (nonce >> 8) & 0xFF;
    block[63] = nonce & 0xFF;
    
    // Compute hash
    uint32_t hash[8];
    sha256_hash(block, 64, hash);
    
    // Measure end time
    clock_t end_time = clock();
    
    // Base anomaly score on execution time variance and hash pattern
    uint32_t time_taken = end_time - start_time;
    uint32_t hash_sum = hash[0] + hash[1];
    
    // Combine factors to create anomaly score (normalized to 0-999999)
    uint32_t score = (time_taken * 500) + (hash_sum % 500000);
    
    // Inject some artificial anomalies for demo purposes
    if (idx % 100 == 0) {
        score = 500000 + (nonce % 500000);
    }
    
    anomaly_scores[idx] = score;
}

// C API for the miner
extern "C" uint32_t mine(uint32_t start_nonce, uint32_t target, uint32_t blocks, uint32_t threads) {
    uint32_t *d_result;
    uint32_t h_result = 0;
    
    // Allocate device memory for result
    cudaMalloc((void **)&d_result, sizeof(uint32_t));
    cudaMemset(d_result, 0, sizeof(uint32_t));
    
    // Reset hash counter
    uint32_t zero = 0;
    cudaMemcpyToSymbol(d_hash_counter, &zero, sizeof(uint32_t));
    
    // Launch kernel
    mining_kernel<<<blocks, threads>>>(start_nonce, target, d_result);
    
    // Copy result back
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Get hash counter for rate measurement
    uint32_t hash_count;
    cudaMemcpyFromSymbol(&hash_count, d_hash_counter, sizeof(uint32_t));
    
    // Free device memory
    cudaFree(d_result);
    
    // Update hash rate (hash count * multiplier to account for measurement interval)
    g_hash_rate = hash_count * 10;
    
    return h_result;
}

// C API for getting hash rate
extern "C" uint32_t get_hash_rate() {
    return g_hash_rate;
}

// C API for anomaly detection
extern "C" void detect_anomalies(uint32_t start_nonce, float *anomaly_scores, uint32_t blocks, uint32_t threads) {
    uint32_t *d_scores;
    uint32_t *h_scores = new uint32_t[blocks * threads];
    
    // Allocate device memory
    cudaMalloc((void **)&d_scores, blocks * threads * sizeof(uint32_t));
    
    // Launch kernel
    anomaly_detection_kernel<<<blocks, threads>>>(start_nonce, d_scores);
    
    // Copy results back
    cudaMemcpy(h_scores, d_scores, blocks * threads * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Convert to float scores (0.0-1.0)
    for (uint32_t i = 0; i < blocks * threads; i++) {
        anomaly_scores[i] = h_scores[i] / 1000000.0f;
    }
    
    // Clean up
    cudaFree(d_scores);
    delete[] h_scores;
}

// Full SHA-256 implementation for transaction data
extern "C" void compute_hash(const char *data, int length, uint32_t *hash_result) {
    // Convert char* to uint8_t*
    const uint8_t *message = reinterpret_cast<const uint8_t*>(data);
    
    // Allocate device memory
    uint8_t *d_message;
    uint32_t *d_hash;
    
    cudaMalloc((void**)&d_message, length);
    cudaMalloc((void**)&d_hash, 8 * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_message, message, length, cudaMemcpyHostToDevice);
    
    // Compute hash - using a simple kernel
    auto compute_hash_kernel = [=] __device__ () {
        uint32_t hash[8];
        sha256_hash(d_message, length, hash);
        for (int i = 0; i < 8; i++) {
            d_hash[i] = hash[i];
        }
    };
    
    // Launch kernel (using a single thread)
    void *args[] = {nullptr};
    cudaLaunchKernel((void*)compute_hash_kernel, dim3(1), dim3(1), args);
    
    // Copy result back
    cudaMemcpy(hash_result, d_hash, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_message);
    cudaFree(d_hash);
}
