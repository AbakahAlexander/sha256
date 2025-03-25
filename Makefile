NVCC = ./nvcc_wrapper.sh
# More compatibility flags to handle older C++ standards
NVCC_FLAGS = -O3 --std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=0 -Xcompiler "-fno-stack-protector -fpermissive"

CXX = g++
CXXFLAGS = -std=c++11 -O3

# Default target is just the shared library needed for the web app
all: simple_bitcoin_miner.so

# Other targets are optional
optional: sha256_benchmark sha256_crack

sha256_benchmark: sha_256.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

sha256_crack: sha256_crack.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

simple_bitcoin_miner.so: simple_bitcoin_miner.cu
	$(NVCC) $(NVCC_FLAGS) --shared -o $@ $< -Xcompiler -fPIC

clean:
	rm -f sha256_benchmark sha256_crack simple_bitcoin_miner.so
