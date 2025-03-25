import numpy as np
import ctypes
import hashlib
import threading
import time
from datetime import datetime
import json
import queue
from collections import defaultdict

class TransactionAnalyzer:
    def __init__(self):
        # Analysis settings
        self.is_analyzing = False
        self.analyzer_thread = None
        self.input_queue = queue.Queue(maxsize=1000)
        self.anomaly_threshold = 0.2
        self.analysis_results = []
        self.max_results = 1000  # Keep at most 1000 results in history
        
        # User behavior history (for profiling)
        self.user_history = defaultdict(list)
        self.location_hash_patterns = {}
        self.amount_hash_patterns = {}
        
        # Try to load the CUDA library for SHA-256 computation
        try:
            self.cuda_lib = ctypes.CDLL('./simple_bitcoin_miner.so')
            self.cuda_lib.compute_hash.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
            self.cuda_available = True
            print("CUDA library loaded for transaction analysis")
        except Exception as e:
            print(f"Failed to load CUDA library for transaction analysis: {e}")
            self.cuda_available = False
        
        # Statistics
        self.total_analyzed = 0
        self.total_anomalies = 0
        self.start_time = None
        
    def _compute_transaction_hash(self, transaction):
        """Compute SHA-256 hash of a transaction"""
        # If CUDA is available, use GPU acceleration
        if self.cuda_available:
            # Convert transaction to JSON string
            transaction_str = json.dumps(transaction, sort_keys=True).encode('utf-8')
            
            # Allocate buffer for hash result (8 uint32 values)
            hash_result = (ctypes.c_uint32 * 8)()
            
            # Call CUDA function to compute hash
            self.cuda_lib.compute_hash(
                transaction_str,
                len(transaction_str),
                hash_result
            )
            
            # Convert hash result to hexadecimal string
            hash_hex = ''.join(f'{val:08x}' for val in hash_result)
            return hash_hex
        else:
            # Fallback to CPU-based hashing
            transaction_str = json.dumps(transaction, sort_keys=True).encode('utf-8')
            return hashlib.sha256(transaction_str).hexdigest()
    
    def _compute_feature_hash(self, user_id, feature_value):
        """Compute hash of specific user feature"""
        combined = f"{user_id}:{feature_value}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()[:16]  # Use shorter hash for features
    
    def _analyze_transaction(self, transaction):
        """Analyze a single transaction for anomalies"""
        user_id = transaction['user_id']
        user_history = self.user_history[user_id]
        
        # Compute transaction hash
        transaction_hash = self._compute_transaction_hash(transaction)
        
        # Compute feature-specific hashes
        location_hash = self._compute_feature_hash(user_id, transaction['location'])
        amount_hash = self._compute_feature_hash(user_id, str(int(transaction['amount'])))
        
        # Initialize anomaly score components
        location_anomaly = 0.0
        amount_anomaly = 0.0
        frequency_anomaly = 0.0
        pattern_anomaly = 0.0
        
        # Check location anomaly
        if user_id in self.location_hash_patterns:
            typical_locations = self.location_hash_patterns[user_id]
            if location_hash not in typical_locations:
                location_anomaly = 0.7
        
        # Check amount anomaly
        if user_id in self.amount_hash_patterns:
            amount_profile = self.amount_hash_patterns[user_id]
            amount_found = False
            for known_amount in amount_profile:
                # Compare first few digits of hash (amount range)
                if amount_hash[:4] == known_amount[:4]:
                    amount_found = True
                    break
            if not amount_found:
                amount_anomaly = 0.6
        
        # Check transaction frequency
        if len(user_history) >= 2:
            # Get timestamps of last few transactions
            last_timestamp = datetime.fromisoformat(transaction['timestamp'])
            prev_timestamp = datetime.fromisoformat(user_history[-1]['timestamp'])
            
            # Check if transactions are too close together
            time_diff = (last_timestamp - prev_timestamp).total_seconds()
            if time_diff < 60:  # Less than a minute apart
                frequency_anomaly = 0.5 + (0.5 * (60 - time_diff) / 60)
        
        # Look for unusual hash patterns
        if len(user_history) >= 3:
            # Check hash collision or unusual patterns
            last_hashes = [self._compute_transaction_hash(tx) for tx in user_history[-3:]]
            
            # Check for unusual hash similarity
            similarities = []
            for h in last_hashes:
                # Count matching characters with current hash
                similarity = sum(a == b for a, b in zip(transaction_hash, h)) / len(transaction_hash)
                similarities.append(similarity)
            
            avg_similarity = sum(similarities) / len(similarities)
            
            # Very high or low similarity could indicate fraud
            if avg_similarity > 0.8 or avg_similarity < 0.2:
                pattern_anomaly = 0.4
        
        # Combine anomaly scores
        # Weight them based on significance (can be tuned)
        weighted_score = (
            location_anomaly * 0.3 +
            amount_anomaly * 0.3 +
            frequency_anomaly * 0.25 +
            pattern_anomaly * 0.15
        )
        
        # Create analysis result
        analysis = {
            'transaction_id': transaction['transaction_id'],
            'timestamp': transaction['timestamp'],
            'user_id': user_id,
            'transaction_hash': transaction_hash,
            'anomaly_score': weighted_score,
            'components': {
                'location_anomaly': location_anomaly,
                'amount_anomaly': amount_anomaly,
                'frequency_anomaly': frequency_anomaly,
                'pattern_anomaly': pattern_anomaly
            },
            'is_anomalous': weighted_score > self.anomaly_threshold,
            'true_positive': transaction.get('is_fraudulent', False) and weighted_score > self.anomaly_threshold,
            'false_negative': transaction.get('is_fraudulent', False) and weighted_score <= self.anomaly_threshold,
            'false_positive': not transaction.get('is_fraudulent', False) and weighted_score > self.anomaly_threshold
        }
        
        # Update user history with this transaction
        user_history.append(transaction)
        if len(user_history) > 20:  # Keep only last 20 transactions per user
            user_history.pop(0)
            
        # Update location patterns
        if user_id not in self.location_hash_patterns:
            self.location_hash_patterns[user_id] = set()
        self.location_hash_patterns[user_id].add(location_hash)
        
        # Update amount patterns
        if user_id not in self.amount_hash_patterns:
            self.amount_hash_patterns[user_id] = set()
        self.amount_hash_patterns[user_id].add(amount_hash)
        
        # Count this analysis
        self.total_analyzed += 1
        if analysis['is_anomalous']:
            self.total_anomalies += 1
            
        return analysis
    
    def _analyzer_loop(self):
        """Main analysis loop"""
        self.start_time = datetime.now()
        
        while self.is_analyzing:
            try:
                # Get a transaction from the queue
                transaction = self.input_queue.get(timeout=1.0)
                
                # Analyze the transaction
                analysis_result = self._analyze_transaction(transaction)
                
                # Add result to history
                self.analysis_results.append(analysis_result)
                if len(self.analysis_results) > self.max_results:
                    self.analysis_results.pop(0)
                
                # Mark task as done
                self.input_queue.task_done()
                
            except queue.Empty:
                # No transactions in queue, just continue
                pass
            except Exception as e:
                print(f"Error in transaction analysis: {e}")
    
    def start_analyzer(self):
        """Start the transaction analyzer"""
        if not self.is_analyzing:
            self.is_analyzing = True
            self.analyzer_thread = threading.Thread(target=self._analyzer_loop)
            self.analyzer_thread.daemon = True
            self.analyzer_thread.start()
            return True
        return False
    
    def stop_analyzer(self):
        """Stop the transaction analyzer"""
        if self.is_analyzing:
            self.is_analyzing = False
            if self.analyzer_thread:
                self.analyzer_thread.join(timeout=2.0)
            return True
        return False
    
    def add_transaction(self, transaction):
        """Add a transaction to the analysis queue"""
        try:
            self.input_queue.put(transaction, block=False)
            return True
        except queue.Full:
            return False
    
    def get_status(self):
        """Get current analyzer status"""
        now = datetime.now()
        elapsed_time = (now - self.start_time).total_seconds() if self.start_time else 0
        
        anomaly_rate = (self.total_anomalies / max(1, self.total_analyzed)) * 100
        
        # Calculate detection metrics
        true_positives = sum(1 for r in self.analysis_results if r.get('true_positive', False))
        false_negatives = sum(1 for r in self.analysis_results if r.get('false_negative', False))
        false_positives = sum(1 for r in self.analysis_results if r.get('false_positive', False))
        
        # Calculate precision and recall if possible
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1_score = 2 * precision * recall / max(0.001, precision + recall)
        
        return {
            "running": self.is_analyzing,
            "total_analyzed": self.total_analyzed,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": anomaly_rate,
            "elapsed_time": elapsed_time,
            "transactions_per_second": self.total_analyzed / max(1, elapsed_time),
            "queue_size": self.input_queue.qsize(),
            "detection_metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }
        }
    
    def get_recent_anomalies(self, limit=10):
        """Get the most recent anomalies"""
        anomalies = [r for r in self.analysis_results if r['is_anomalous']]
        return anomalies[-limit:] if anomalies else []
    
    def get_all_results(self, limit=100):
        """Get all analysis results"""
        return self.analysis_results[-limit:] if self.analysis_results else []

# Global instance for use across modules
transaction_analyzer = TransactionAnalyzer()

# For testing
if __name__ == "__main__":
    import time
    from transaction_simulator import TransactionSimulator
    
    # Create instances
    simulator = TransactionSimulator()
    analyzer = TransactionAnalyzer()
    
    # Start both
    simulator.start_simulation()
    analyzer.start_analyzer()
    
    # Process some transactions
    print("Processing transactions for 10 seconds...")
    start_time = time.time()
    while time.time() - start_time < 10:
        transaction = simulator.get_transaction(timeout=0.5)
        if transaction:
            analyzer.add_transaction(transaction)
    
    # Get results
    anomalies = analyzer.get_recent_anomalies()
    print(f"\nFound {len(anomalies)} anomalies:")
    for a in anomalies:
        print(f"Transaction {a['transaction_id']}: Score {a['anomaly_score']:.2f}")
    
    # Get status
    status = analyzer.get_status()
    print(f"\nAnalyzer status:")
    print(f"- Transactions analyzed: {status['total_analyzed']}")
    print(f"- Anomalies detected: {status['total_anomalies']}")
    print(f"- Anomaly rate: {status['anomaly_rate']:.2f}%")
    print(f"- Detection precision: {status['detection_metrics']['precision']:.2f}")
    print(f"- Detection recall: {status['detection_metrics']['recall']:.2f}")
    
    # Stop both
    simulator.stop_simulation()
    analyzer.stop_analyzer()
    print("\nStopped simulator and analyzer")
