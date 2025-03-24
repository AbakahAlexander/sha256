NVCC = ./nvcc_wrapper.sh
NVCC_FLAGS = -O3

CXX = g++-11
CXXFLAGS = -std=c++17 -O3

all: sha256_benchmark sha256_crack simple_bitcoin_miner.so

sha256_benchmark: sha_256.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

sha256_crack: sha256_crack.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

simple_bitcoin_miner.so: simple_bitcoin_miner.cu
	$(NVCC) $(NVCC_FLAGS) --shared -o $@ $< -Xcompiler -fPIC

clean:
	rm -f sha256_benchmark sha256_crack simple_bitcoin_miner.so
