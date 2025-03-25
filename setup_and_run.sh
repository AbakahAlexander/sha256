#!/bin/bash

echo "Setting up SHA-256 Fraud Detection Demo..."

# Make the CUDA wrapper script executable
chmod +x nvcc_wrapper.sh
echo "Made CUDA wrapper executable"

# Build only the necessary shared library
make simple_bitcoin_miner.so
if [ $? -ne 0 ]; then
    echo "Failed to build shared library. Continuing anyway since we might be able to run in CPU-only mode."
fi

# Check if templates directory exists, create if not
mkdir -p templates
mkdir -p static

# Start the web server
echo "Starting web server on port 8080..."
python3 web_server.py

# If the server doesn't start, try with Python instead of Python3
if [ $? -ne 0 ]; then
    echo "Trying alternative Python command..."
    python web_server.py
fi
