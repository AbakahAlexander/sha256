import random
import time
import json
import threading
import uuid
from datetime import datetime, timedelta
import numpy as np
import queue

class TransactionSimulator:
    def __init__(self):
        self.is_running = False
        self.simulation_thread = None
        self.transaction_queue = queue.Queue(maxsize=1000)
        self.transactions_history = []
        self.max_history = 1000
        
        self.user_profiles = {
            'retail': {
                'count': 50,
                'transaction_frequency': (5, 20),
                'amount_range': (10, 500),
                'locations': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
                'fraud_probability': 0.005
            },
            'business': {
                'count': 20,
                'transaction_frequency': (30, 120),
                'amount_range': (500, 10000),
                'locations': ['San Francisco', 'Seattle', 'Boston', 'Miami', 'Dallas'],
                'fraud_probability': 0.01
            },
            'high_value': {
                'count': 10,
                'transaction_frequency': (120, 480),
                'amount_range': (5000, 50000),
                'locations': ['New York', 'San Francisco', 'Chicago', 'Miami', 'Las Vegas'],
                'fraud_probability': 0.02
            }
        }
        
        self.users = self._generate_users()
        
        self.fraud_patterns = [
            self._unusual_location_fraud,
            self._unusual_amount_fraud,
            self._unusual_frequency_fraud
        ]
        
        self.total_transactions = 0
        self.total_fraudulent = 0
        self.start_time = None
        
    def _generate_users(self):
        users = {}
        
        for profile_type, profile in self.user_profiles.items():
            for i in range(profile['count']):
                user_id = f"{profile_type}_{i+1:03d}"
                
                users[user_id] = {
                    'user_id': user_id,
                    'profile_type': profile_type,
                    'typical_amounts': np.random.normal(
                        (profile['amount_range'][0] + profile['amount_range'][1]) / 2,
                        (profile['amount_range'][1] - profile['amount_range'][0]) / 6,
                        10
                    ).tolist(),
                    'typical_locations': random.sample(profile['locations'], 
                                                     min(3, len(profile['locations']))),
                    'last_transaction': None,
                    'transaction_count': 0,
                    'fraud_probability': profile['fraud_probability'],
                    'transaction_frequency': random.uniform(
                        profile['transaction_frequency'][0],
                        profile['transaction_frequency'][1]
                    )
                }
        
        return users
    
    def _generate_normal_transaction(self, user_id):
        user = self.users[user_id]
        profile = self.user_profiles[user['profile_type']]
        
        amount = random.choice(user['typical_amounts']) * random.uniform(0.8, 1.2)
        amount = max(profile['amount_range'][0], min(profile['amount_range'][1], amount))
        amount = round(amount, 2)
        
        location = random.choice(user['typical_locations'])
        timestamp = datetime.now()
        
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'user_id': user_id,
            'amount': amount,
            'location': location,
            'timestamp': timestamp.isoformat(),
            'type': random.choice(['payment', 'transfer', 'withdrawal', 'deposit']),
            'is_fraudulent': False,
            'fraud_type': None,
            'confidence': 1.0
        }
        
        user['last_transaction'] = timestamp
        user['transaction_count'] += 1
        
        return transaction
    
    def _unusual_location_fraud(self, user_id):
        transaction = self._generate_normal_transaction(user_id)
        
        user = self.users[user_id]
        profile = self.user_profiles[user['profile_type']]
        unusual_locations = [loc for loc in profile['locations'] if loc not in user['typical_locations']]
        
        if unusual_locations:
            transaction['location'] = random.choice(unusual_locations)
        else:
            transaction['location'] = "International"
            
        transaction['is_fraudulent'] = True
        transaction['fraud_type'] = 'unusual_location'
        transaction['confidence'] = random.uniform(0.7, 0.9)
        
        return transaction
    
    def _unusual_amount_fraud(self, user_id):
        transaction = self._generate_normal_transaction(user_id)
        
        user = self.users[user_id]
        profile = self.user_profiles[user['profile_type']]
        max_normal = profile['amount_range'][1]
        
        transaction['amount'] = max_normal * random.uniform(2, 5)
        transaction['amount'] = round(transaction['amount'], 2)
        transaction['is_fraudulent'] = True
        transaction['fraud_type'] = 'unusual_amount'
        transaction['confidence'] = random.uniform(0.8, 0.95)
        
        return transaction
    
    def _unusual_frequency_fraud(self, user_id):
        transaction = self._generate_normal_transaction(user_id)
        
        transaction['is_fraudulent'] = True
        transaction['fraud_type'] = 'unusual_frequency'
        transaction['confidence'] = random.uniform(0.6, 0.85)
        
        return transaction
    
    def _generate_transaction(self):
        user_id = random.choice(list(self.users.keys()))
        user = self.users[user_id]
        
        is_fraud = random.random() < user['fraud_probability']
        
        if is_fraud:
            fraud_generator = random.choice(self.fraud_patterns)
            transaction = fraud_generator(user_id)
            self.total_fraudulent += 1
        else:
            transaction = self._generate_normal_transaction(user_id)
        
        self.total_transactions += 1
        return transaction
    
    def _simulation_loop(self):
        self.start_time = datetime.now()
        
        batch_size = 50
        
        while self.is_running:
            try:
                start_time = time.time()
                transactions = []
                
                for _ in range(batch_size):
                    transactions.append(self._generate_transaction())
                
                for tx in transactions:
                    try:
                        self.transaction_queue.put_nowait(tx)
                        self.transactions_history.append(tx)
                    except queue.Full:
                        break
                
                while len(self.transactions_history) > self.max_history:
                    self.transactions_history.pop(0)
                
                elapsed = time.time() - start_time
                target_time = batch_size * 0.01
                sleep_time = max(0.001, target_time - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error in transaction simulation: {e}")
                time.sleep(0.1)
    
    def start_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.simulation_thread = threading.Thread(target=self._simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            return True
        return False
    
    def stop_simulation(self):
        if self.is_running:
            self.is_running = False
            if self.simulation_thread:
                self.simulation_thread.join(timeout=2.0)
            return True
        return False
    
    def get_transaction(self, block=True, timeout=None):
        try:
            return self.transaction_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_status(self):
        now = datetime.now()
        elapsed_time = (now - self.start_time).total_seconds() if self.start_time else 0
        
        fraudulent_rate = (self.total_fraudulent / max(1, self.total_transactions)) * 100
        
        return {
            "running": self.is_running,
            "total_transactions": self.total_transactions,
            "total_fraudulent": self.total_fraudulent,
            "fraudulent_rate": fraudulent_rate,
            "elapsed_time": elapsed_time,
            "transactions_per_second": self.total_transactions / max(1, elapsed_time),
            "queue_size": self.transaction_queue.qsize(),
        }
    
    def get_recent_transactions(self, limit=10):
        return self.transactions_history[-limit:] if self.transactions_history else []

transaction_simulator = TransactionSimulator()

if __name__ == "__main__":
    simulator = TransactionSimulator()
    simulator.start_simulation()
    
    print("Generating 10 sample transactions:")
    for i in range(10):
        transaction = simulator.get_transaction()
        print(f"{i+1}. User: {transaction['user_id']}, Amount: ${transaction['amount']:.2f}, " +
              f"Location: {transaction['location']}, Fraud: {transaction['is_fraudulent']}")
    
    simulator.stop_simulation()
    print("\nSimulation stopped")
