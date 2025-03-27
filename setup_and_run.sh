#!/bin/bash
set -e  # Exit on any error

echo "Setting up SHA-256 Demo Platform..."

# Make sure executables are runnable
chmod +x nvcc_wrapper.sh
echo "Made CUDA wrapper executable"

# Clean previous builds
echo "Cleaning up previous builds..."
make clean || true  # Continue even if this fails

# Build the library
echo "Building shared library..."
make simple_bitcoin_miner.so

# Check if build was successful
if [ ! -f "simple_bitcoin_miner.so" ]; then
    echo "Failed to build shared library. Aborting."
    exit 1
fi

# Create directories if they don't exist
mkdir -p templates
mkdir -p static

# Check for port usage more simply
PORT=8080
echo "Checking if port $PORT is in use..."

# Don't exit if this fails - just report and continue
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

# Run the web server - THIS LINE WAS MISSING THE AMPERSAND TO RUN IN BACKGROUND
echo "Starting web server on port $PORT..."
python3 web_server.py

# This is a fallback and should only run if the previous command fails
if [ $? -ne 0 ]; then
    echo "Trying alternative Python command..."
    python web_server.py
fi
