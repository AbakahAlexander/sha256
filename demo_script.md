
# Financial Fraud Detection Demo with GPU-Accelerated SHA-256

This demonstration showcases how GPU-accelerated SHA-256 hashing can be used to detect anomalous transactions that could indicate fraud in real-time.

## Setup Instructions

1. Compile the CUDA code:
   ```bash
   chmod +x nvcc_wrapper.sh
   make
   ```

2. Start the web server:
   ```bash
   python web_server.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8080/fraud-detection
   ```

## Demonstration Flow

### 1. System Overview

In this system, we use GPU-accelerated SHA-256 hashing to:

- Generate fingerprints of financial transactions
- Detect anomalies in transaction patterns
- Process transactions in real-time 
- Visualize fraud detection metrics

### 2. Transaction Simulation

- The system generates realistic financial transactions from different user profiles:
  - Retail customers (small, frequent transactions)
  - Business accounts (medium-sized, regular transactions)
  - High-value clients (large, infrequent transactions)

- Each profile has different normal behaviors and fraud risks

### 3. Fraud Detection with GPU-Accelerated Hashing

The system hashes each transaction's details using SHA-256 on the GPU and analyzes:

- **Location Anomalies**: Transactions from unusual locations
- **Amount Anomalies**: Unusually large transactions
- **Frequency Anomalies**: Multiple transactions in a short time period
- **Pattern Anomalies**: Unusual patterns in transaction hash distributions

### 4. Real-Time Monitoring

The dashboard shows:
- Live transaction stream
- Real-time anomaly scores
- Distribution of anomaly patterns
- Fraud alerts with risk scores

### 5. Performance Benefits of GPU Acceleration

- **Speed**: Process thousands of transactions per second
- **Scalability**: Handle increasing transaction volumes
- **Real-Time Detection**: Identify fraud as it happens, not after the fact

## Technical Details

- SHA-256 implementation is CUDA-accelerated
- Transaction fingerprints are analyzed using historical patterns
- Anomaly detection uses hash-based scoring
- System maintains user-specific hash patterns for behavioral profiling

## Security Implications

- Hash-based anomaly detection preserves privacy while detecting fraud
- GPU acceleration enables processing at financial institution scale
- Real-time alerts allow immediate intervention

## Next Steps

- Implement more sophisticated machine learning models on hash patterns
- Integrate with existing financial security systems
- Add support for cross-account transaction analysis
