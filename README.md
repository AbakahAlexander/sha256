# SHA-256 GPU-Accelerated Security Platform

A comprehensive security platform demonstrating the power of GPU-accelerated cryptographic operations for financial fraud detection and security monitoring.

## Project Overview

This platform showcases how GPU-accelerated SHA-256 hashing can be applied to real-world security challenges. It implements three core demonstrations:

1. **Financial Fraud Detection System** - Uses transaction fingerprinting and behavioral analysis
2. **CryptoGuardian AI Security Monitor** - Detects threats in cryptographic operations
3. **Bitcoin Mining Demonstration** - Illustrates SHA-256 proof-of-work principles

## System Architecture

The platform employs a hybrid architecture that leverages both GPU and CPU resources:

- **GPU Component**: CUDA-based implementation of SHA-256 hash functions, optimized for parallel computation
- **Web Server**: Flask-based backend that handles API requests and serves the UI
- **Data Processing Pipeline**: Multi-threaded system for real-time transaction processing
- **Visualization Layer**: Interactive dashboards for monitoring and analysis

## Technical Components

### GPU-Accelerated SHA-256 Implementation

The core cryptographic functionality is implemented in CUDA C++, providing:

- Simplified SHA-256 hashing algorithm optimized for throughput
- Memory-efficient transaction fingerprinting
- Constant-time operations to mitigate timing attacks
- Multi-threaded execution model with configurable block and thread counts
- CPU fallback mechanism when CUDA is unavailable

### Financial Fraud Detection System

#### Transaction Simulation Engine

- Generates realistic financial transaction data across different user profiles
- Models normal behavioral patterns for retail, business, and high-value accounts
- Simulates location, amount, and frequency patterns specific to each user
- Injects anomalous transactions at configurable rates for testing

#### Anomaly Detection Algorithm

The system uses a multi-faceted approach to detect fraudulent activity:

- **Hash-Based Fingerprinting**: Each transaction is hashed using SHA-256 to create a unique fingerprint
- **Historical Pattern Analysis**: User-specific behavior profiles are maintained
- **Location Anomaly Detection**: Identifies transactions from unusual geographic locations
- **Amount Anomaly Detection**: Flags unusually large transactions based on user history
- **Frequency Anomaly Detection**: Detects unusual transaction timing patterns
- **Hash Pattern Analysis**: Examines similarities between transaction hashes

The detection algorithm assigns weighted anomaly scores based on multiple factors:
- Location anomaly contributes 30% to the final score
- Amount anomaly contributes 30% to the final score
- Frequency anomaly contributes 25% to the final score
- Hash pattern anomaly contributes 15% to the final score

#### Real-Time Processing Pipeline

- Multi-threaded architecture for transaction handling
- Thread-safe queuing system for transaction buffering
- Concurrent analysis of transaction streams
- Efficient inter-thread communication with minimal locking

### CryptoGuardian AI Security Monitor

#### Threat Detection Methodology

The security monitor uses GPU-accelerated detection for:

- **Timing Attacks**: Detecting variations in cryptographic operation execution times
- **Side-Channel Attacks**: Identifying potential information leakage
- **Brute Force Attempts**: Recognizing patterns consistent with exhaustive search
- **Dictionary Attacks**: Detecting systematic password guessing

#### Anomaly Scoring System

- Real-time anomaly scoring based on cryptographic operation patterns
- Adaptive thresholding based on historical baseline metrics
- Severity classification (Low, Medium, High) for detected threats
- Trend analysis for evolving attack patterns

### Performance Optimizations

- Memory-efficient data structures for high-throughput processing
- Asynchronous transaction processing to prevent UI blocking
- Optimized CUDA kernel configurations for different GPU architectures
- Efficient data transfers between host and device memory

## Technical Specifications

- **GPU Acceleration**: CUDA-based implementation for all cryptographic operations
- **SHA-256 Implementation**: Simplified for compatibility across CUDA versions
- **API Architecture**: RESTful API for all monitoring and control functions
- **Visualization**: Chart.js-based real-time data visualization
- **Simulation**: Configurable parameters for transaction generation
- **Analytics**: Real-time metrics calculation for system performance

## Hardware Requirements

- CUDA-capable NVIDIA GPU (compute capability 3.0+)
- 4GB+ GPU memory recommended for optimal performance
- 8GB+ system RAM
- 100MB disk space

## Software Requirements

- CUDA Toolkit 11.0+ (compatible with 12.x)
- Python 3.6+
- Flask web framework
- Modern web browser with JavaScript enabled

## Setup and Usage

To run the SHA-256 GPU-Accelerated Security Platform:

1. Clone the repository
2. Run the setup script:
   ```
   chmod +x setup_and_run.sh
   ./setup_and_run.sh
   ```
3. Open your browser and navigate to `http://localhost:8080`
4. Explore the different demo sections through the navigation menu
