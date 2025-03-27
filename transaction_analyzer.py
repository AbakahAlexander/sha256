import numpy as np
import ctypes
import hashlib
import threading
import time
import random
from datetime import datetime
import json
import queue
from collections import defaultdict

class TransactionAnalyzer:
    def __init__(self):
        self.is_analyzing = False
        self.analyzer_thread = None
        self.input_queue = queue.Queue(maxsize=1000)
        self.anomaly_threshold = 0.2
        self.analysis_results = []
        self.max_results = 1000
        
        self.user_history = defaultdict(list)
        self.location_hash_patterns = {}
        self.amount_hash_patterns = {}
        
        try:
            self.cuda_lib = ctypes.CDLL('./simple_bitcoin_miner.so')
            self.cuda_lib.compute_hash.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
            
            self.cuda_lib.compute_hash_batch.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_int
            ]
            
            self.cuda_available = True
            print("CUDA library loaded for transaction analysis with batch processing")
        except Exception as e:
            print(f"Failed to load CUDA library for transaction analysis: {e}")
            self.cuda_available = False
        
        self.total_analyzed = 0
        self.total_anomalies = 0
        self.start_time = None
    
    def _compute_transaction_hash_batch(self, transactions):
        if not hasattr(self.cuda_lib, 'compute_hash_batch'):
            return [self._compute_transaction_hash(tx) for tx in transactions]
        
        batch_size = len(transactions)
        json_data = [json.dumps(tx, sort_keys=True).encode('utf-8') for tx in transactions]
        
        data_array = (ctypes.c_char_p * batch_size)()
        length_array = (ctypes.c_int * batch_size)()
        
        hash_results = []
        hash_result_ptrs = (ctypes.POINTER(ctypes.c_uint32) * batch_size)()
        
        for i in range(batch_size):
            data_array[i] = json_data[i]
            length_array[i] = len(json_data[i])
            
            hash_result = (ctypes.c_uint32 * 8)()
            hash_results.append(hash_result)
            hash_result_ptrs[i] = hash_result
        
        try:
            self.cuda_lib.compute_hash_batch(
                data_array,
                length_array,
                hash_result_ptrs,
                batch_size
            )
            
            return [''.join(f'{val:08x}' for val in result) for result in hash_results]
        except Exception as e:
            print(f"Batch hashing error: {e}")
            return [self._compute_transaction_hash(tx) for tx in transactions]
        
    def _compute_transaction_hash(self, transaction):
        
        transaction_str = json.dumps(transaction, sort_keys=True).encode('utf-8')
        
        hash_result = (ctypes.c_uint32 * 8)()
        
        self.cuda_lib.compute_hash(
            transaction_str,
            len(transaction_str),
            hash_result
        )
        
        hash_hex = ''.join(f'{val:08x}' for val in hash_result)
        return hash_hex
        
    
    def _compute_feature_hash(self, user_id, feature_value):
        combined = f"{user_id}:{feature_value}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()[:16]
    
    def _analyze_transaction(self, transaction):
        user_id = transaction['user_id']
        user_history = self.user_history[user_id]
        
        transaction_hash = self._compute_transaction_hash(transaction)
        
        location_hash = self._compute_feature_hash(user_id, transaction['location'])
        amount_hash = self._compute_feature_hash(user_id, str(int(transaction['amount'])))
        
        location_anomaly = 0.0
        amount_anomaly = 0.0
        frequency_anomaly = 0.0
        pattern_anomaly = 0.0
        
        if user_id in self.location_hash_patterns:
            typical_locations = self.location_hash_patterns[user_id]
            if location_hash not in typical_locations:
                location_anomaly = 0.6 + random.random() * 0.3
        
        if user_id in self.amount_hash_patterns:
            amount_profile = self.amount_hash_patterns[user_id]
            amount_found = False
            for known_amount in amount_profile:
                if amount_hash[:4] == known_amount[:4]:
                    amount_found = True
                    break
            if not amount_found:
                amount_anomaly = 0.55 + random.random() * 0.4
        
        if len(user_history) >= 2:
            last_timestamp = datetime.fromisoformat(transaction['timestamp'])
            prev_timestamp = datetime.fromisoformat(user_history[-1]['timestamp'])
            
            time_diff = (last_timestamp - prev_timestamp).total_seconds()
            if time_diff < 60:
                if time_diff < 10:
                    frequency_anomaly = 0.8 + (random.random() * 0.15)
                else:
                    frequency_anomaly = 0.4 + (0.4 * (60 - time_diff) / 60)
        
        if len(user_history) >= 3:
            tx_batch = user_history[-3:] + [transaction]
            all_hashes = self._compute_transaction_hash_batch(tx_batch)
            
            transaction_hash = all_hashes[-1]
            last_hashes = all_hashes[:-1]
            
            similarities = []
            for h in last_hashes:
                similarity = sum(a == b for a, b in zip(transaction_hash, h)) / len(transaction_hash)
                similarities.append(similarity)
            
            avg_similarity = sum(similarities) / len(similarities)
            
            if avg_similarity > 0.8:
                pattern_anomaly = 0.7 + (random.random() * 0.2)
            elif avg_similarity < 0.2:
                pattern_anomaly = 0.6 + (random.random() * 0.3)
            else:
                pattern_anomaly = random.random() * 0.2
        
        weighted_score = (
            location_anomaly * 0.3 +
            amount_anomaly * 0.3 +
            frequency_anomaly * 0.25 +
            pattern_anomaly * 0.15
        )
        
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
        
        user_history.append(transaction)
        if len(user_history) > 20:
            user_history.pop(0)
            
        if user_id not in self.location_hash_patterns:
            self.location_hash_patterns[user_id] = set()
        self.location_hash_patterns[user_id].add(location_hash)
        
        if user_id not in self.amount_hash_patterns:
            self.amount_hash_patterns[user_id] = set()
        self.amount_hash_patterns[user_id].add(amount_hash)
        
        self.total_analyzed += 1
        if analysis['is_anomalous']:
            self.total_anomalies += 1
            
        return analysis
    
    def _analyze_transactions_batch(self, transactions):
        if not transactions:
            return []
        
        all_transaction_hashes = self._compute_transaction_hash_batch(transactions)
        
        user_to_txs = defaultdict(list)
        user_to_indices = defaultdict(list)
        for i, tx in enumerate(transactions):
            user_id = tx['user_id']
            user_to_txs[user_id].append(tx)
            user_to_indices[user_id].append(i)
        
        location_hashes = []
        amount_hashes = []
        for tx in transactions:
            user_id = tx['user_id']
            location_hashes.append(self._compute_feature_hash(user_id, tx['location']))
            amount_hashes.append(self._compute_feature_hash(user_id, str(int(tx['amount']))))
        
        history_txs = []
        history_user_map = {}
        for user_id, user_txs in user_to_txs.items():
            user_history = self.user_history[user_id]
            if len(user_history) >= 3:
                for hist_tx in user_history[-3:]:
                    history_txs.append(hist_tx)
                    history_user_map[len(history_txs)-1] = user_id
        
        history_hashes = {}
        if history_txs:
            all_history_hashes = self._compute_transaction_hash_batch(history_txs)
            for i, hist_hash in enumerate(all_history_hashes):
                user_id = history_user_map[i]
                if user_id not in history_hashes:
                    history_hashes[user_id] = []
                history_hashes[user_id].append(hist_hash)
        
        results = []
        
        for i, transaction in enumerate(transactions):
            user_id = transaction['user_id']
            user_history = self.user_history[user_id]
            transaction_hash = all_transaction_hashes[i]
            
            location_anomaly = 0.0
            amount_anomaly = 0.0
            frequency_anomaly = 0.0
            pattern_anomaly = 0.0
            
            if user_id in self.location_hash_patterns:
                if location_hashes[i] not in self.location_hash_patterns[user_id]:
                    location_anomaly = 0.6 + random.random() * 0.3
            
            if user_id in self.amount_hash_patterns:
                amount_found = False
                for known_amount in self.amount_hash_patterns[user_id]:
                    if amount_hashes[i][:4] == known_amount[:4]:
                        amount_found = True
                        break
                if not amount_found:
                    amount_anomaly = 0.55 + random.random() * 0.4
            
            if len(user_history) >= 2:
                last_timestamp = datetime.fromisoformat(transaction['timestamp'])
                prev_timestamp = datetime.fromisoformat(user_history[-1]['timestamp'])
                
                time_diff = (last_timestamp - prev_timestamp).total_seconds()
                if time_diff < 60:
                    if time_diff < 10:
                        frequency_anomaly = 0.8 + (random.random() * 0.15)
                    else:
                        frequency_anomaly = 0.4 + (0.4 * (60 - time_diff) / 60)
            
            if user_id in history_hashes:
                user_hist_hashes = history_hashes[user_id]
                
                similarities = [
                    sum(a == b for a, b in zip(transaction_hash, h)) / len(transaction_hash)
                    for h in user_hist_hashes
                ]
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    if avg_similarity > 0.8:
                        pattern_anomaly = 0.7 + (random.random() * 0.2)
                    elif avg_similarity < 0.2:
                        pattern_anomaly = 0.6 + (random.random() * 0.3)
                    else:
                        pattern_anomaly = random.random() * 0.2
            
            weighted_score = (
                location_anomaly * 0.3 +
                amount_anomaly * 0.3 +
                frequency_anomaly * 0.25 +
                pattern_anomaly * 0.15
            )
            
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
            
            user_history.append(transaction)
            if len(user_history) > 20:
                user_history.pop(0)
                
            if user_id not in self.location_hash_patterns:
                self.location_hash_patterns[user_id] = set()
            self.location_hash_patterns[user_id].add(location_hashes[i])
            
            if user_id not in self.amount_hash_patterns:
                self.amount_hash_patterns[user_id] = set()
            self.amount_hash_patterns[user_id].add(amount_hashes[i])
            
            self.total_analyzed += 1
            if analysis['is_anomalous']:
                self.total_anomalies += 1
                
            results.append(analysis)
        
        return results
    
    def _analyzer_loop(self):
        self.start_time = datetime.now()
        
        while self.is_analyzing:
            try:
                transaction = self.input_queue.get(timeout=1.0)
                
                analysis_result = self._analyze_transaction(transaction)
                
                self.analysis_results.append(analysis_result)
                if len(self.analysis_results) > self.max_results:
                    self.analysis_results.pop(0)
                
                self.input_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in transaction analysis: {e}")
    
    def _analyzer_loop_batch(self):
        self.start_time = datetime.now()
        batch_size = 512
        
        while self.is_analyzing:
            try:
                transaction_batch = []
                start_extract = time.time()
                
                while len(transaction_batch) < batch_size:
                    try:
                        transaction = self.input_queue.get_nowait()
                        transaction_batch.append(transaction)
                        self.input_queue.task_done()
                    except queue.Empty:
                        break
                
                if transaction_batch:
                    batch_count = len(transaction_batch)
                    print(f"Processing batch of {batch_count} transactions")
                    
                    start_process = time.time()
                    analysis_results = self._analyze_transactions_batch(transaction_batch)
                    process_time = time.time() - start_process
                    
                    self.analysis_results.extend(analysis_results)
                    
                    if len(self.analysis_results) > self.max_results:
                        self.analysis_results = self.analysis_results[-self.max_results:]
                    
                    total_time = time.time() - start_extract
                    print(f"Batch metrics: {batch_count} transactions, " 
                          f"Processing: {process_time:.3f}s, Total: {total_time:.3f}s, "
                          f"Rate: {batch_count/total_time:.1f} tx/s")
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"Error in batch transaction analysis: {e}")
                import traceback
                traceback.print_exc()

    def start_analyzer(self):
        if not self.is_analyzing:
            self.is_analyzing = True
            self.analyzer_thread = threading.Thread(target=self._analyzer_loop_batch)
            self.analyzer_thread.daemon = True
            self.analyzer_thread.start()
            return True
        return False
    
    def stop_analyzer(self):
        if self.is_analyzing:
            self.is_analyzing = False
            if self.analyzer_thread:
                self.analyzer_thread.join(timeout=2.0)
            return True
        return False
    
    def add_transaction(self, transaction):
        try:
            self.input_queue.put(transaction, block=False)
            return True
        except queue.Full:
            return False
    
    def get_status(self):
        now = datetime.now()
        elapsed_time = (now - self.start_time).total_seconds() if self.start_time else 0
        
        anomaly_rate = (self.total_anomalies / max(1, self.total_analyzed)) * 100
        true_positives = sum(1 for r in self.analysis_results if r.get('true_positive', False))
        false_negatives = sum(1 for r in self.analysis_results if r.get('false_negative', False))
        false_positives = sum(1 for r in self.analysis_results if r.get('false_positive', False))
        
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
        anomalies = [r for r in self.analysis_results if r['is_anomalous']]
        return anomalies[-limit:] if anomalies else []
    
    def get_all_results(self, limit=100):
        return self.analysis_results[-limit:] if self.analysis_results else []

transaction_analyzer = TransactionAnalyzer()

if __name__ == "__main__":
    from transaction_simulator import TransactionSimulator
    import time

    simulator = TransactionSimulator()
    analyzer = TransactionAnalyzer()
    
    simulator.start_simulation()
    analyzer.start_analyzer()
    
    print("Processing transactions for 10 seconds...")
    start_time = time.time()
    while time.time() - start_time < 10:
        transaction = simulator.get_transaction(timeout=0.5)
        if transaction:
            analyzer.add_transaction(transaction)
    
    print("\nFound anomalies:")
    anomalies = analyzer.get_recent_anomalies()
    for a in anomalies:
        print(f"Transaction {a['transaction_id']}: Score {a['anomaly_score']:.2f}")
    
    print("\nStopped simulator and analyzer")
    analyzer.stop_analyzer()
    simulator.stop_simulation()
    
    print("\nAnalyzer status:")
    status = analyzer.get_status()
    print(f"- Transactions analyzed: {status['total_analyzed']}")
    print(f"- Anomalies detected: {status['total_anomalies']}")
    print(f"- Anomaly rate: {status['anomaly_rate']:.2f}%")
    print(f"- Detection precision: {status['detection_metrics']['precision']:.2f}")
    print(f"- Detection recall: {status['detection_metrics']['recall']:.2f}")
    print(f"- F1 score: {status['detection_metrics']['f1_score']:.2f}")