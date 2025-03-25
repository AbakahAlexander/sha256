# SHA-256 GPU-Accelerated Demo Platform

A suite of demonstrations showcasing GPU-accelerated SHA-256 implementations, including financial fraud detection, AI-driven security monitoring, and Bitcoin mining.

## Key Features

- **Financial Fraud Detection**: Real-time anomaly detection in financial transactions using GPU-accelerated SHA-256 hashing
- **CryptoGuardian AI**: Security monitoring for cryptographic operations with threat detection
- **Bitcoin Mining Demo**: Simple demonstration of SHA-256 mining principles

## System Requirements

- CUDA-capable NVIDIA GPU
- CUDA Toolkit (tested with versions 11.x and 12.x)
- Python 3.6+
- Flask

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd sha256

# Setup and run
chmod +x setup_and_run.sh
./setup_and_run.sh
```

Then open your browser and navigate to:
```
http://localhost:8080
```

## Financial Fraud Detection

The fraud detection system uses GPU acceleration to analyze financial transactions in real-time, detecting anomalous patterns that could indicate fraudulent activity. The system:

- Simulates realistic financial transactions from different user profiles
- Uses SHA-256 to create digital fingerprints of each transaction
- Builds behavioral profiles for each user based on transaction history
- Detects unusual activity patterns through hash-based anomaly scoring

### Key Components

- **Transaction Simulator**: Generates realistic transaction data from different user profiles
- **Transaction Analyzer**: Processes transactions and detects anomalies
- **Interactive Dashboard**: Real-time visualization of transaction patterns and fraud alerts

## CryptoGuardian AI

An AI-driven security monitoring system for cryptographic operations. It detects potential threats such as:

- Timing attacks
- Side-channel vulnerabilities
- Brute force attempts
- Dictionary attacks

## CPU Fallback Mode

If CUDA is not available, the system will automatically fall back to CPU-based implementations for all functionality, ensuring compatibility across different environments.

## Project Structure

```
sha256/
├── simple_bitcoin_miner.cu     # CUDA implementation of core functionality
├── web_server.py               # Flask web server
├── transaction_simulator.py    # Transaction data generator
├── transaction_analyzer.py     # Transaction analysis engine
├── crypto_guardian.py          # AI security monitoring
├── Makefile                    # Build configuration
├── nvcc_wrapper.sh             # CUDA compiler wrapper
└── templates/                  # Web UI templates
    ├── index.html              # Main dashboard
    ├── fraud_detection.html    # Fraud detection UI
    └── guardian.html           # Security monitoring UI
```

## Development Notes

- The CUDA code uses simplified hash implementations to ensure compatibility across different CUDA versions
- Use `nvcc_wrapper.sh` for portable CUDA compilation across different setups
