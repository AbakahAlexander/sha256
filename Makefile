NVCC = ./nvcc_wrapper.sh
NVCC_FLAGS = -O3 --std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=0 \
             --extended-lambda \
             -isystem /usr/lib/gcc/x86_64-linux-gnu/11/include \
             -isystem /usr/include/c++/11 \
             -isystem /usr/include/x86_64-linux-gnu/c++/11 \
             -Xcompiler "-fno-stack-protector -fpermissive" \
             --compiler-options "-fPIC"

CXX = g++
CXXFLAGS = -std=c++11 -O3

all: simple_bitcoin_miner.so

simple_bitcoin_miner.so: simple_bitcoin_miner.cu
	$(NVCC) $(NVCC_FLAGS) --shared -o $@ $<

clean:
	rm -f simple_bitcoin_miner.so
