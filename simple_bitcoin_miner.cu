#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

__device__ uint32_t d_hash_counter = 0;
__host__ uint32_t g_hash_rate = 0;

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

__device__ __constant__ uint32_t initial_hash[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

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

__device__ void sha256_transform(uint32_t state[8], const uint32_t data[16]) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h, t1, t2;
    
    for (int i = 0; i < 16; i++) {
        w[i] = data[i];
    }
    
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }
    
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];
    
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
    
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void prepare_sha256_message(const uint8_t *message, uint32_t length, uint32_t block[16]) {
    uint32_t i, j;
    
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
            
            block[j] |= 0x80 << (8 * (3 - (length - i)));
            j++;
            
            while (j < 14) {
                block[j++] = 0;
            }
            
            block[14] = 0;
            block[15] = length * 8;
            return;
        }
    }
    
    block[j] = 0x80000000;
    j++;
    
    while (j < 14) {
        block[j++] = 0;
    }
    
    block[14] = 0;
    block[15] = length * 8;
}

__device__ void sha256_hash(const uint8_t *message, uint32_t length, uint32_t hash[8]) {
    uint32_t block[16];
    
    for (int i = 0; i < 8; i++) {
        hash[i] = initial_hash[i];
    }
    
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
    
    prepare_sha256_message(message + i, length - i, block);
    sha256_transform(hash, block);
}

__global__ void mining_kernel(uint32_t start_nonce, uint32_t target, uint32_t *result) {
    uint32_t nonce = start_nonce + (blockIdx.x * blockDim.x + threadIdx.x);
    
    uint8_t block[64];
    for (int i = 0; i < 60; i++) {
        block[i] = 0;
    }
    
    block[60] = (nonce >> 24) & 0xFF;
    block[61] = (nonce >> 16) & 0xFF;
    block[62] = (nonce >> 8) & 0xFF;
    block[63] = nonce & 0xFF;
    
    uint32_t hash[8];
    sha256_hash(block, 64, hash);
    
    if (hash[0] < target) {
        *result = nonce;
    }
    
    atomicAdd(&d_hash_counter, 1);
}

__global__ void anomaly_detection_kernel(uint32_t start_nonce, uint32_t *anomaly_scores) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = start_nonce + idx;
    
    clock_t start_time = clock();
    
    uint8_t block[64];
    for (int i = 0; i < 60; i++) {
        block[i] = 0;
    }
    
    block[60] = (nonce >> 24) & 0xFF;
    block[61] = (nonce >> 16) & 0xFF;
    block[62] = (nonce >> 8) & 0xFF;
    block[63] = nonce & 0xFF;
    
    uint32_t hash[8];
    sha256_hash(block, 64, hash);
    
    clock_t end_time = clock();
    
    uint32_t time_taken = end_time - start_time;
    uint32_t hash_sum = hash[0] + hash[1];
    
    uint32_t score = (time_taken * 500) + (hash_sum % 500000);
    
    if (idx % 100 == 0) {
        score = 500000 + (nonce % 500000);
    }
    
    anomaly_scores[idx] = score;
}

extern "C" uint32_t mine(uint32_t start_nonce, uint32_t target, uint32_t blocks, uint32_t threads) {
    uint32_t *d_result;
    uint32_t h_result = 0;
    
    cudaMalloc((void **)&d_result, sizeof(uint32_t));
    cudaMemset(d_result, 0, sizeof(uint32_t));
    
    uint32_t zero = 0;
    cudaMemcpyToSymbol(d_hash_counter, &zero, sizeof(uint32_t));
    
    mining_kernel<<<blocks, threads>>>(start_nonce, target, d_result);
    
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    uint32_t hash_count;
    cudaMemcpyFromSymbol(&hash_count, d_hash_counter, sizeof(uint32_t));
    
    cudaFree(d_result);
    
    g_hash_rate = hash_count * 10;
    
    return h_result;
}

extern "C" uint32_t get_hash_rate() {
    return g_hash_rate;
}

extern "C" void detect_anomalies(uint32_t start_nonce, float *anomaly_scores, uint32_t blocks, uint32_t threads) {
    uint32_t *d_scores;
    uint32_t *h_scores = new uint32_t[blocks * threads];
    
    cudaMalloc((void **)&d_scores, blocks * threads * sizeof(uint32_t));
    
    anomaly_detection_kernel<<<blocks, threads>>>(start_nonce, d_scores);
    
    cudaMemcpy(h_scores, d_scores, blocks * threads * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    for (uint32_t i = 0; i < blocks * threads; i++) {
        anomaly_scores[i] = h_scores[i] / 1000000.0f;
    }
    
    cudaFree(d_scores);
    delete[] h_scores;
}

__global__ void compute_hash_kernel(const uint8_t* message, uint32_t length, uint32_t* hash_output) {
    __shared__ uint32_t hash_state[8];
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            hash_state[i] = initial_hash[i];
        }
    }
    
    __syncthreads();
    
    uint32_t thread_id = threadIdx.x;
    uint32_t total_threads = blockDim.x;
    
    if (thread_id == 0) {
        uint32_t hash[8];
        sha256_hash(message, length, hash);
        
        for (int i = 0; i < 8; i++) {
            hash_output[i] = hash[i];
        }
    }
}

extern "C" void compute_hash(const char *data, int length, uint32_t *hash_result) {
    const uint8_t *message = reinterpret_cast<const uint8_t*>(data);
    
    uint8_t *d_message;
    uint32_t *d_hash;
    
    cudaMalloc((void**)&d_message, length);
    cudaMalloc((void**)&d_hash, 8 * sizeof(uint32_t));
    
    cudaMemcpy(d_message, message, length, cudaMemcpyHostToDevice);
    
    compute_hash_kernel<<<1, 32>>>(d_message, length, d_hash);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(hash_result, d_hash, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_message);
    cudaFree(d_hash);
}

__global__ void batch_hash_kernel(const uint8_t** messages, const uint32_t* lengths, uint32_t** hash_outputs, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        uint32_t hash[8];
        sha256_hash(messages[idx], lengths[idx], hash);
        
        for (int i = 0; i < 8; i++) {
            hash_outputs[idx][i] = hash[i];
        }
    }
}

extern "C" void compute_hash_batch(
    char** data_array, 
    int* length_array, 
    uint32_t** hash_results,
    int batch_size
) {
    if (batch_size <= 0) return;
    
    uint8_t** d_messages;
    uint32_t* d_lengths;
    uint32_t** d_hashes;
    
    uint8_t** h_d_messages = new uint8_t*[batch_size];
    uint32_t** h_d_hashes = new uint32_t*[batch_size];
    
    cudaMalloc((void**)&d_messages, batch_size * sizeof(uint8_t*));
    cudaMalloc((void**)&d_lengths, batch_size * sizeof(uint32_t));
    cudaMalloc((void**)&d_hashes, batch_size * sizeof(uint32_t*));
    
    for (int i = 0; i < batch_size; i++) {
        cudaMalloc((void**)&h_d_messages[i], length_array[i]);
        cudaMemcpy(h_d_messages[i], data_array[i], length_array[i], cudaMemcpyHostToDevice);
        
        cudaMalloc((void**)&h_d_hashes[i], 8 * sizeof(uint32_t));
    }
    
    cudaMemcpy(d_messages, h_d_messages, batch_size * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, length_array, batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hashes, h_d_hashes, batch_size * sizeof(uint32_t*), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    
    batch_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        (const uint8_t**)d_messages, 
        d_lengths, 
        d_hashes, 
        batch_size
    );
    
    cudaDeviceSynchronize();
    
    for (int i = 0; i < batch_size; i++) {
        cudaMemcpy(hash_results[i], h_d_hashes[i], 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < batch_size; i++) {
        cudaFree(h_d_messages[i]);
        cudaFree(h_d_hashes[i]);
    }
    
    cudaFree(d_messages);
    cudaFree(d_lengths);
    cudaFree(d_hashes);
    
    delete[] h_d_messages;
    delete[] h_d_hashes;
}
