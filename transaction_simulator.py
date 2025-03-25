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
        # Transaction generation settings
        self.is_running = False
        self.simulation_thread = None
        self.transaction_queue = queue.Queue(maxsize=1000)
        self.transactions_history = []
        self.max_history = 1000  # Keep at most 1000 transactions in history
        
        # User profiles - normal behavior patterns for different user types
        self.user_profiles = {
            'retail': {
                'count': 50,
                'transaction_frequency': (5, 20),  # minutes between transactions
                'amount_range': (10, 500),
                'locations': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
                'fraud_probability': 0.005
            },
            'business': {
                'count': 20,
                'transaction_frequency': (30, 120),  # minutes between transactions
                'amount_range': (500, 10000),
                'locations': ['San Francisco', 'Seattle', 'Boston', 'Miami', 'Dallas'],
                'fraud_probability': 0.01
            },
            'high_value': {
                'count': 10,
                'transaction_frequency': (120, 480),  # minutes between transactions
                'amount_range': (5000, 50000),
                'locations': ['New York', 'San Francisco', 'Chicago', 'Miami', 'Las Vegas'],
                'fraud_probability': 0.02
            }
        }
        
        # Generate user database
        self.users = self._generate_users()
        
        # Fraud patterns
        self.fraud_patterns = [
            self._unusual_location_fraud,
            self._unusual_amount_fraud,
            self._unusual_frequency_fraud
        ]
        
        # Statistics
        self.total_transactions = 0
        self.total_fraudulent = 0
        self.start_time = None
        
    def _generate_users(self):
        """Generate simulated users with their profiles"""
        users = {}
        
        for profile_type, profile in self.user_profiles.items():
            for i in range(profile['count']):
                user_id = f"{profile_type}_{i+1:03d}"
                
                # Create user profile
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
        """Generate a normal transaction for a user"""
        user = self.users[user_id]
        profile = self.user_profiles[user['profile_type']]
        
        # Generate transaction data
        amount = random.choice(user['typical_amounts']) * random.uniform(0.8, 1.2)
        amount = max(profile['amount_range'][0], min(profile['amount_range'][1], amount))
        amount = round(amount, 2)
        
        location = random.choice(user['typical_locations'])
        timestamp = datetime.now()
        
        # Create transaction object
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
        
        # Update user's last transaction info
        user['last_transaction'] = timestamp
        user['transaction_count'] += 1
        
        return transaction
    
    def _unusual_location_fraud(self, user_id):
        """Generate a transaction with an unusual location"""
        transaction = self._generate_normal_transaction(user_id)
        
        # Choose an unusual location
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
        """Generate a transaction with an unusually high amount"""
        transaction = self._generate_normal_transaction(user_id)
        
        # Set an unusual amount - significantly higher than typical
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
        """Generate multiple transactions in a short time period"""
        transaction = self._generate_normal_transaction(user_id)
        
        transaction['is_fraudulent'] = True
        transaction['fraud_type'] = 'unusual_frequency'
        transaction['confidence'] = random.uniform(0.6, 0.85)
        
        return transaction
    
    def _generate_transaction(self):
        """Generate a single transaction, potentially fraudulent"""
        # Choose a random user
        user_id = random.choice(list(self.users.keys()))
        user = self.users[user_id]
        
        # Decide if this will be a fraudulent transaction
        is_fraud = random.random() < user['fraud_probability']
        
        if is_fraud:
            # Choose a fraud pattern
            fraud_generator = random.choice(self.fraud_patterns)
            transaction = fraud_generator(user_id)
            self.total_fraudulent += 1
        else:
            # Generate normal transaction
            transaction = self._generate_normal_transaction(user_id)
        
        self.total_transactions += 1
        return transaction
    
    def _simulation_loop(self):
        """Main simulation loop"""
        self.start_time = datetime.now()
        
        while self.is_running:
            # Generate transactions at a random rate
            try:
                # Generate a transaction and add it to the queue
                transaction = self._generate_transaction()
                self.transaction_queue.put(transaction, block=False)
                
                # Also add to history
                self.transactions_history.append(transaction)
                if len(self.transactions_history) > self.max_history:
                    self.transactions_history.pop(0)
                
                # Adjust sleep time based on desired transaction rate
                # More sophisticated simulators might use a Poisson process
                sleep_time = random.uniform(0.1, 0.5)  # 2-10 transactions per second
                time.sleep(sleep_time)
                
            except queue.Full:
                # Queue is full, wait for consumer to catch up
                time.sleep(1)
    
    def start_simulation(self):
        """Start the transaction simulation"""
        if not self.is_running:
            self.is_running = True
            self.simulation_thread = threading.Thread(target=self._simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            return True
        return False
    
    def stop_simulation(self):
        """Stop the transaction simulation"""
        if self.is_running:
            self.is_running = False
            if self.simulation_thread:
                self.simulation_thread.join(timeout=2.0)
            return True
        return False
    
    def get_transaction(self, block=True, timeout=None):
        """Get the next transaction from the queue"""
        try:
            return self.transaction_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_status(self):
        """Get current simulation status"""
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
        """Get the most recent transactions"""
        return self.transactions_history[-limit:] if self.transactions_history else []

# Global instance for use across modules
transaction_simulator = TransactionSimulator()

# For testing
if __name__ == "__main__":
    simulator = TransactionSimulator()
    simulator.start_simulation()
    
    # Generate and print 10 transactions
    print("Generating 10 sample transactions:")
    for i in range(10):
        transaction = simulator.get_transaction()
        print(f"{i+1}. User: {transaction['user_id']}, Amount: ${transaction['amount']:.2f}, " +
              f"Location: {transaction['location']}, Fraud: {transaction['is_fraudulent']}")
    
    simulator.stop_simulation()
    print("\nSimulation stopped")
