// CUDA SHA-256 Password Cracking Implementation

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define THREADS_PER_BLOCK 256
#define BLOCKS 512
#define MAX_PASSWORD_LEN 8
#define CHARSET_SIZE 36  // a-z and 0-9

// SHA-256 constants in device constant memory
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

// Character set for password generation
__device__ __constant__ char charset[] = "abcdefghijklmnopqrstuvwxyz0123456789";

// The target hash we're trying to crack (stored in constant memory)
__device__ __constant__ uint32_t target_hash[8];

// Rotation function for SHA-256
__device__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// Host version of rotation function (for CPU)
__host__ uint32_t h_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 hash computation function
__device__ void sha256_transform(const uint8_t *data, uint32_t *hash_out) {
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) |
               (data[i * 4 + 2] << 8) | data[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

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

    hash_out[0] = 0x6a09e667 + a;
    hash_out[1] = 0xbb67ae85 + b;
    hash_out[2] = 0x3c6ef372 + c;
    hash_out[3] = 0xa54ff53a + d;
    hash_out[4] = 0x510e527f + e;
    hash_out[5] = 0x9b05688c + f;
    hash_out[6] = 0x1f83d9ab + g;
    hash_out[7] = 0x5be0cd19 + h;
}

// Host version of SHA-256 transform (for CPU)
__host__ void h_sha256_transform(const uint8_t *data, uint32_t *hash_out) {
    uint32_t w[64];
    
    for (int i = 0; i < 16; i++) {
        w[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) |
               (data[i * 4 + 2] << 8) | data[i * 4 + 3];
    }
    
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = h_rotr(w[i - 15], 7) ^ h_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = h_rotr(w[i - 2], 17) ^ h_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    // Host copy of k constants
    uint32_t host_k[64] = {
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

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = h_rotr(e, 6) ^ h_rotr(e, 11) ^ h_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + host_k[i] + w[i];
        uint32_t S0 = h_rotr(a, 2) ^ h_rotr(a, 13) ^ h_rotr(a, 22);
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

    hash_out[0] = 0x6a09e667 + a;
    hash_out[1] = 0xbb67ae85 + b;
    hash_out[2] = 0x3c6ef372 + c;
    hash_out[3] = 0xa54ff53a + d;
    hash_out[4] = 0x510e527f + e;
    hash_out[5] = 0x9b05688c + f;
    hash_out[6] = 0x1f83d9ab + g;
    hash_out[7] = 0x5be0cd19 + h;
}

// Prepare SHA-256 message block with padding
__device__ void prepare_input(const char *password, int length, uint8_t *input) {
    // Clear the input buffer first
    for (int i = 0; i < 64; i++) {
        input[i] = 0;
    }
    
    // Copy password into input buffer
    for (int i = 0; i < length; i++) {
        input[i] = password[i];
    }
    
    // Add the '1' bit after the message
    input[length] = 0x80;
    
    // Add the length in bits as a 64-bit big-endian integer at the end
    uint64_t bit_length = length * 8;
    input[63] = bit_length & 0xFF;
    input[62] = (bit_length >> 8) & 0xFF;
    input[61] = (bit_length >> 16) & 0xFF;
    input[60] = (bit_length >> 24) & 0xFF;
    input[59] = (bit_length >> 32) & 0xFF;
    input[58] = (bit_length >> 40) & 0xFF;
    input[57] = (bit_length >> 48) & 0xFF;
    input[56] = (bit_length >> 56) & 0xFF;
}

// Generate password for a given index
__device__ void generate_password(uint64_t index, char *password, int max_len, int *length) {
    // Determine password length (start with shorter passwords)
    *length = 1;
    uint64_t count = CHARSET_SIZE; // Number of 1-char passwords
    
    // Find the length for this index
    while (index >= count && *length < max_len) {
        index -= count;
        (*length)++;
        count *= CHARSET_SIZE;
    }
    
    // Generate the password characters
    for (int i = *length - 1; i >= 0; i--) {
        password[i] = charset[index % CHARSET_SIZE];
        index /= CHARSET_SIZE;
    }
    
    // Null terminator
    password[*length] = '\0';
}

// Check if two hashes match
__device__ bool compare_hashes(const uint32_t *hash1, const uint32_t *hash2) {
    for (int i = 0; i < 8; i++) {
        if (hash1[i] != hash2[i]) {
            return false;
        }
    }
    return true;
}

// Kernel for password cracking
__global__ void crack_password(uint64_t start_index, uint64_t *result_index, char *result_password, 
                              int max_password_len, bool *password_found) {
    uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t password_index = start_index + thread_id;
    
    // If a password is already found, exit early
    if (*password_found) return;
    
    // Buffer for the generated password
    char password[MAX_PASSWORD_LEN + 1];
    int length;
    
    // Generate the password for this thread
    generate_password(password_index, password, max_password_len, &length);
    
    // Prepare the hash input
    uint8_t input[64];
    prepare_input(password, length, input);
    
    // Compute the SHA-256 hash
    uint32_t hash[8];
    sha256_transform(input, hash);
    
    // Check if hash matches target
    if (compare_hashes(hash, target_hash)) {
        // Found a match! Store the result
        *password_found = true;
        *result_index = password_index;
        
        // Copy the password to result
        for (int i = 0; i <= length; i++) {
            result_password[i] = password[i];
        }
    }
}

// Compute SHA-256 hash of a string on the host
void compute_sha256(const char *input, uint32_t *hash) {
    // Initialize hash values
    hash[0] = 0x6a09e667;
    hash[1] = 0xbb67ae85;
    hash[2] = 0x3c6ef372;
    hash[3] = 0xa54ff53a;
    hash[4] = 0x510e527f;
    hash[5] = 0x9b05688c;
    hash[6] = 0x1f83d9ab;
    hash[7] = 0x5be0cd19;
    
    // Prepare the input block with padding
    uint8_t h_input[64] = {0};
    int len = strlen(input);
    memcpy(h_input, input, len);
    h_input[len] = 0x80;  // Append the '1' bit
    
    // Append the length in bits as a 64-bit big-endian integer at the end
    uint64_t bit_length = len * 8;
    h_input[63] = bit_length & 0xFF;
    h_input[62] = (bit_length >> 8) & 0xFF;
    h_input[61] = (bit_length >> 16) & 0xFF;
    h_input[60] = (bit_length >> 24) & 0xFF;
    h_input[59] = (bit_length >> 32) & 0xFF;
    h_input[58] = (bit_length >> 40) & 0xFF;
    h_input[57] = (bit_length >> 48) & 0xFF;
    h_input[56] = (bit_length >> 56) & 0xFF;
    
    // Call the host version of SHA-256 transform
    h_sha256_transform(h_input, hash);
}

int main(int argc, char **argv) {
    // Default password to crack
    const char *password_to_crack = "abc123";
    int max_len = 6;
    bool demo_mode = true;
    
    // Parse command line arguments
    if (argc > 1) {
        // If a hash is provided, use it as target
        if (argc >= 9 && strcmp(argv[1], "--hash") == 0) {
            demo_mode = false;
            uint32_t hash[8];
            for (int i = 0; i < 8; i++) {
                hash[i] = strtoul(argv[i + 2], NULL, 16);
            }
            cudaMemcpyToSymbol(target_hash, hash, sizeof(uint32_t) * 8);
        }
        // If a password is provided, hash it and then crack it
        else {
            password_to_crack = argv[1];
            if (argc > 2) {
                max_len = atoi(argv[2]);
            }
        }
    }
    
    // If in demo mode, compute the hash of the password
    if (demo_mode) {
        uint32_t hash[8];
        compute_sha256(password_to_crack, hash);
        
        printf("Target password: %s\n", password_to_crack);
        printf("SHA-256 hash: ");
        for (int i = 0; i < 8; i++) {
            printf("%08x ", hash[i]);
        }
        printf("\n\n");
        
        // Copy hash to device constant memory
        cudaMemcpyToSymbol(target_hash, hash, sizeof(uint32_t) * 8);
    }
    
    // Allocate device memory for results
    uint64_t *d_result_index;
    char *d_result_password;
    bool *d_password_found;
    
    cudaMalloc(&d_result_index, sizeof(uint64_t));
    cudaMalloc(&d_result_password, MAX_PASSWORD_LEN + 1);
    cudaMalloc(&d_password_found, sizeof(bool));
    
    // Initialize result values
    uint64_t h_result_index = 0;
    char h_result_password[MAX_PASSWORD_LEN + 1] = {0};
    bool h_password_found = false;
    
    cudaMemcpy(d_result_index, &h_result_index, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_password, h_result_password, MAX_PASSWORD_LEN + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_password_found, &h_password_found, sizeof(bool), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timer
    printf("Starting password crack (max length: %d)...\n", max_len);
    cudaEventRecord(start);
    
    // Calculate search space size (sum of CHARSET_SIZE^i for i=1 to max_len)
    uint64_t search_space = 0;
    uint64_t charset_size_power = 1;
    for (int i = 1; i <= max_len; i++) {
        charset_size_power *= CHARSET_SIZE;
        search_space += charset_size_power;
    }
    printf("Total search space: %lu passwords\n", search_space);
    
    // Batch size for each kernel launch
    uint64_t batch_size = THREADS_PER_BLOCK * BLOCKS;
    uint64_t num_batches = (search_space + batch_size - 1) / batch_size;
    
    // Launch kernel in batches
    uint64_t passwords_tried = 0;
    uint64_t batches_completed = 0;
    time_t start_time = time(NULL);
    time_t last_update = start_time;
    
    for (uint64_t i = 0; i < num_batches && !h_password_found; i++) {
        uint64_t start_index = i * batch_size;
        
        // Launch the kernel
        crack_password<<<BLOCKS, THREADS_PER_BLOCK>>>(start_index, d_result_index, d_result_password, max_len, d_password_found);
        
        // Check if password found every 10 batches or if last batch
        if ((i + 1) % 10 == 0 || i == num_batches - 1) {
            cudaMemcpy(&h_password_found, d_password_found, sizeof(bool), cudaMemcpyDeviceToHost);
        }
        
        // Update progress every second
        passwords_tried += batch_size;
        batches_completed++;
        time_t now = time(NULL);
        if (now != last_update || h_password_found) {
            last_update = now;
            double progress = (double)passwords_tried / search_space * 100.0;
            double elapsed = difftime(now, start_time);
            double speed = elapsed > 0 ? passwords_tried / elapsed / 1000000.0 : 0;
            
            printf("\rProgress: %.2f%% | Elapsed: %.0fs | Speed: %.2f M pwd/s | Batches: %lu/%lu", 
                   progress, elapsed, speed, batches_completed, num_batches);
            fflush(stdout);
        }
        
        // If found, break early
        if (h_password_found) break;
    }
    
    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    // Get the result if found
    if (h_password_found) {
        cudaMemcpy(&h_result_index, d_result_index, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_result_password, d_result_password, MAX_PASSWORD_LEN + 1, cudaMemcpyDeviceToHost);
        
        printf("\n\nPassword found: %s\n", h_result_password);
        printf("Password index: %lu\n", h_result_index);
        printf("Time taken: %.2f seconds\n", elapsed_ms / 1000.0);
        printf("Throughput: %.2f million passwords/second\n", 
               h_result_index / (elapsed_ms / 1000.0) / 1000000.0);
    } else {
        printf("\n\nPassword not found in the search space.\n");
    }
    
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_result_index);
    cudaFree(d_result_password);
    cudaFree(d_password_found);
    
    return 0;
}
