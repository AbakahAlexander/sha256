NVCC = ./nvcc_wrapper.sh
NVCC_FLAGS = -O3 --std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=0 -Xcompiler "-fno-stack-protector -fpermissive"

CXX = g++
CXXFLAGS = -std=c++11 -O3

all: simple_bitcoin_miner.so

simple_bitcoin_miner.so: simple_bitcoin_miner.cu
	$(NVCC) $(NVCC_FLAGS) --shared -o $@ $< -Xcompiler -fPIC

clean:
	rm -f simple_bitcoin_miner.so
