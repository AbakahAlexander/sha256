#!/bin/bash

echo "Setting up SHA-256 Demo Platform..."

chmod +x nvcc_wrapper.sh
echo "Made CUDA wrapper executable"

make simple_bitcoin_miner.so
if [ $? -ne 0 ]; then
    echo "Failed to build shared library. Continuing anyway since we might be able to run in CPU-only mode."
fi

mkdir -p templates
mkdir -p static

echo "Starting web server on port 8080..."
python3 web_server.py

if [ $? -ne 0 ]; then
    echo "Trying alternative Python command..."
    python web_server.py
fi
