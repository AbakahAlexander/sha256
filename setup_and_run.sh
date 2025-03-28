#!/bin/bash
set -e  

echo "Setting up SHA-256 Demo Platform..."


chmod +x nvcc_wrapper.sh
echo "Made CUDA wrapper executable"


echo "Cleaning up previous builds..."
make clean || true  

echo "Building shared library..."
make simple_bitcoin_miner.so


if [ ! -f "simple_bitcoin_miner.so" ]; then
    echo "Failed to build shared library. Aborting."
    exit 1
fi


mkdir -p templates
mkdir -p static


PORT=8080
echo "Checking if port $PORT is in use..."


if command -v lsof >/dev/null 2>&1; then
    PROCESS_PID=$(lsof -ti:$PORT 2>/dev/null || echo "")
    if [ ! -z "$PROCESS_PID" ]; then
        echo "Port $PORT is in use. Attempting to kill process..."
        kill -9 $PROCESS_PID || echo "Failed to kill process, but continuing anyway"
        sleep 1
    fi
else
    echo "lsof not installed, skipping port check."
fi


echo "Starting web server on port $PORT..."
python3 web_server.py

if [ $? -ne 0 ]; then
    echo "Trying alternative Python command..."
    python web_server.py
fi
