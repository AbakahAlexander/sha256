#!/bin/bash

# This is a wrapper script for nvcc that ensures compilation works
# regardless of where CUDA is installed on the system

# Try to find nvcc in common locations
NVCC_PATHS=(
    "/usr/local/cuda/bin/nvcc"
    "/usr/bin/nvcc"
    "/opt/cuda/bin/nvcc"
    "/usr/lib/nvidia-cuda-toolkit/bin/nvcc"
)

NVCC=""

for path in "${NVCC_PATHS[@]}"; do
    if [ -x "$path" ]; then
        NVCC="$path"
        break
    fi
done

# If nvcc wasn't found, try using the PATH
if [ -z "$NVCC" ]; then
    if command -v nvcc &> /dev/null; then
        NVCC=$(command -v nvcc)
    else
        echo "Error: CUDA nvcc compiler not found."
        echo "Please ensure CUDA toolkit is installed and nvcc is in your PATH."
        exit 1
    fi
fi

# Execute nvcc with all arguments passed to this script
echo "Using CUDA compiler at: $NVCC"
$NVCC "$@"
