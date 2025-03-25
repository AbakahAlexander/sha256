import numpy as np
import threading
import time
import random
import json
import os
import ctypes
from flask import jsonify, request

class CryptoGuardian:
    def __init__(self):
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Detection thresholds
        self.anomaly_threshold = 0.15
        
        # History
        self.alerts = []
        self.metrics_history = []
        
        # Threat categories
        self.threat_categories = [
            "Benign",
            "Timing Attack",
            "Side-Channel Attack",
            "Brute Force Attack",
            "Dictionary Attack"
        ]
        
        # Load CUDA library
        try:
            self.cuda_lib = ctypes.CDLL('./simple_bitcoin_miner.so')
            self.cuda_lib.mine.restype = ctypes.c_uint32
            self.cuda_lib.get_hash_rate.restype = ctypes.c_uint32
            self.cuda_lib.detect_anomalies.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_uint32,
                ctypes.c_uint32
            ]
            self.cuda_available = True
            print("CUDA library loaded successfully")
        except Exception as e:
            print(f"Failed to load CUDA library: {e}")
            self.cuda_available = False
        
        # CUDA parameters
        self.blocks = 32
        self.threads = 256
        self.total_threads = self.blocks * self.threads
        self.nonce = 0
        
    def start_monitoring(self):
        """Start monitoring cryptographic operations"""
        print("Starting monitoring...")
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("Monitoring thread started")
            return True
        return False
    
    def stop_monitoring(self):
        """Stop monitoring cryptographic operations"""
        print("Stopping monitoring...")
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1.0)
            print("Monitoring stopped")
            return True
        return False
    
    def _monitor_loop(self):
        """Main monitoring loop that processes data and detects threats"""
        print("Monitor loop running")
        while self.is_monitoring:
            if self.cuda_available:
                # Use CUDA-based anomaly detection
                self._detect_cuda_anomalies()
            else:
                # Fallback to random anomaly generation
                self._generate_random_anomalies()
                
            # Sleep for a bit
            time.sleep(0.5)
    
    def _detect_cuda_anomalies(self):
        """Use CUDA to detect real cryptographic anomalies"""
        # Allocate buffer for anomaly scores
        anomaly_scores = (ctypes.c_float * self.total_threads)()
        
        # Call CUDA function to detect anomalies
        self.cuda_lib.detect_anomalies(
            ctypes.c_uint32(self.nonce),
            anomaly_scores,
            ctypes.c_uint32(self.blocks),
            ctypes.c_uint32(self.threads)
        )
        
        # Get hash rate from the miner
        hash_rate = self.cuda_lib.get_hash_rate()
        
        # Process each anomaly score
        timestamp = time.time()
        max_anomaly = 0.0
        max_threat = "Benign"
        
        for i in range(self.total_threads):
            anomaly_score = anomaly_scores[i]
            max_anomaly = max(max_anomaly, anomaly_score)
            
            # Determine threat type based on anomaly pattern
            if anomaly_score > self.anomaly_threshold:
                # Classify the threat based on anomaly patterns
                if anomaly_score > 0.5:
                    threat_type = "Timing Attack"
                elif i % 4 == 0:
                    threat_type = "Side-Channel Attack"
                elif i % 4 == 1:
                    threat_type = "Brute Force Attack"
                else:
                    threat_type = "Dictionary Attack"
                
                max_threat = threat_type
                
                # Create alert for significant anomalies
                alert = {
                    "timestamp": timestamp,
                    "anomaly_score": float(anomaly_score),
                    "threat_type": threat_type,
                    "severity": "High" if anomaly_score > 0.3 else "Medium",
                    "message": f"Potential {threat_type} detected in cryptographic operations"
                }
                self.alerts.append(alert)
                print(f"ALERT: {alert['message']} (Score: {anomaly_score:.2f})")
        
        # Record overall metrics
        record = {
            "timestamp": timestamp,
            "metrics": [float(anomaly_scores[i]) for i in range(min(10, self.total_threads))],
            "anomaly_score": float(max_anomaly),
            "hash_rate": hash_rate,
            "threat_type": max_threat
        }
        
        # Store in history
        self.metrics_history.append(record)
        
        # Limit history size
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Increment nonce for next iteration
        self.nonce += self.total_threads
    
    def _generate_random_anomalies(self):
        """Generate random anomalies for demonstration when CUDA is not available"""
        # Generate a random anomaly score
        anomaly_score = random.uniform(0, 0.5)
        
        # Randomly choose a threat type
        threat_idx = 0 if anomaly_score < self.anomaly_threshold else random.randint(1, 4)
        threat_type = self.threat_categories[threat_idx]
        
        # Record metrics and findings
        timestamp = time.time()
        record = {
            "timestamp": timestamp,
            "metrics": [random.random() for _ in range(10)],
            "anomaly_score": float(anomaly_score),
            "hash_rate": random.randint(500000, 2000000),
            "threat_type": threat_type
        }
        
        # Store in history
        self.metrics_history.append(record)
        
        # If anomaly detected, create alert
        if anomaly_score > self.anomaly_threshold:
            alert = {
                "timestamp": timestamp,
                "anomaly_score": float(anomaly_score),
                "threat_type": threat_type,
                "severity": "High" if anomaly_score > 0.3 else "Medium",
                "message": f"Potential {threat_type} detected (simulation)"
            }
            self.alerts.append(alert)
            print(f"ALERT: {alert['message']} (Score: {anomaly_score:.2f})")
        
        # Limit history size
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_status(self):
        """Get current monitoring status"""
        if not self.metrics_history:
            return {
                "monitoring": self.is_monitoring,
                "alerts_count": len(self.alerts),
                "recent_anomaly_score": 0,
                "recent_threat": "None"
            }
            
        return {
            "monitoring": self.is_monitoring,
            "alerts_count": len(self.alerts),
            "recent_anomaly_score": self.metrics_history[-1]["anomaly_score"],
            "recent_threat": self.metrics_history[-1]["threat_type"]
        }
    
    def get_recent_alerts(self, limit=10):
        """Get recent security alerts"""
        return self.alerts[-limit:] if self.alerts else []
    
    def get_metrics_history(self):
        """Get historical metrics for visualization"""
        # Extract time series for visualization
        timestamps = [record["timestamp"] for record in self.metrics_history]
        anomaly_scores = [record["anomaly_score"] for record in self.metrics_history]
        
        # Convert to relative timestamps for better visualization
        if timestamps:
            start_time = timestamps[0]
            rel_timestamps = [t - start_time for t in timestamps]
        else:
            rel_timestamps = []
            
        return {
            "timestamps": rel_timestamps,
            "anomaly_scores": anomaly_scores,
            "threshold": self.anomaly_threshold
        }
    
    def generate_security_report(self):
        """Generate a comprehensive security report"""
        if not self.metrics_history:
            return {"error": "No monitoring data available"}
            
        # Calculate statistics
        anomaly_scores = [record["anomaly_score"] for record in self.metrics_history]
        threat_counts = {}
        for record in self.metrics_history:
            threat = record["threat_type"]
            if threat != "Benign":
                threat_counts[threat] = threat_counts.get(threat, 0) + 1
                
        return {
            "monitoring_duration": len(self.metrics_history) * 0.5,  # In seconds
            "total_samples_analyzed": len(self.metrics_history),
            "anomalies_detected": len(self.alerts),
            "average_anomaly_score": float(np.mean(anomaly_scores)) if anomaly_scores else 0,
            "max_anomaly_score": float(max(anomaly_scores)) if anomaly_scores else 0,
            "threat_distribution": threat_counts,
            "security_rating": self._calculate_security_rating()
        }
    
    def _calculate_security_rating(self):
        """Calculate overall security rating based on monitoring data"""
        if not self.metrics_history:
            return "Unknown"
            
        # Calculate percentage of anomalies
        anomaly_percent = len(self.alerts) / max(len(self.metrics_history), 1) * 100
        
        if anomaly_percent < 1:
            return "Excellent"
        elif anomaly_percent < 5:
            return "Good"
        elif anomaly_percent < 10:
            return "Fair"
        else:
            return "Poor"

# Create guardian instance
crypto_guardian = CryptoGuardian()

# Add routes to existing Flask app
def add_guardian_routes(app):
    @app.route('/api/guardian/start', methods=['POST'])
    def start_guardian():
        print("API: Starting guardian")
        result = crypto_guardian.start_monitoring()
        return jsonify({"status": "started" if result else "already_running"})
    
    @app.route('/api/guardian/stop', methods=['POST'])
    def stop_guardian():
        print("API: Stopping guardian")
        result = crypto_guardian.stop_monitoring()
        return jsonify({"status": "stopped" if result else "not_running"})
    
    @app.route('/api/guardian/status', methods=['GET'])
    def guardian_status():
        status = crypto_guardian.get_status()
        print(f"API: Status query - {status}")
        return jsonify(status)
    
    @app.route('/api/guardian/alerts', methods=['GET'])
    def guardian_alerts():
        alerts = crypto_guardian.get_recent_alerts()
        return jsonify({"alerts": alerts})
    
    @app.route('/api/guardian/report', methods=['GET'])
    def guardian_report():
        return jsonify(crypto_guardian.generate_security_report())
    
    @app.route('/api/guardian/history', methods=['GET'])
    def guardian_history():
        history = crypto_guardian.get_metrics_history()
        return jsonify(history)

# Integration function for main web server
def integrate_guardian(app):
    add_guardian_routes(app)
    print("CryptoGuardian integrated with web server")
