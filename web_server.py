from flask import Flask, render_template, jsonify, request
import ctypes
import os
import time
import threading
import sys
import json
from transaction_simulator import transaction_simulator
from transaction_analyzer import transaction_analyzer

app = Flask(__name__)

# Load the shared library
cuda_lib = ctypes.CDLL('./simple_bitcoin_miner.so')
cuda_lib.mine.restype = ctypes.c_uint32
cuda_lib.get_hash_rate.restype = ctypes.c_uint32

# Global variables
mining_active = False
mining_thread = None
start_time = 0
hash_rate_history = []
found_solutions = []

# Transaction processing thread
transaction_process_thread = None
transaction_processing = False

def mining_task():
    global mining_active, hash_rate_history
    
    start_nonce = 0
    target = 0x1000000
    
    while mining_active:
        result = cuda_lib.mine(start_nonce, target, 512, 256)
        
        hash_rate = cuda_lib.get_hash_rate()
        hash_rate_history.append(hash_rate)
        
        if len(hash_rate_history) > 60:
            hash_rate_history.pop(0)
            
        if result != 0:
            found_solutions.append({
                'nonce': result,
                'time': time.time() - start_time
            })
            print(f"Found solution: {result}")
            
        start_nonce += 512 * 256
        
        time.sleep(0.1)

def transaction_processing_task():
    """Process transactions from simulator to analyzer"""
    global transaction_processing
    
    print("Transaction processing task started")
    while transaction_processing:
        # Get transaction from simulator and add to analyzer
        transaction = transaction_simulator.get_transaction(block=True, timeout=1.0)
        if transaction:
            transaction_analyzer.add_transaction(transaction)
        time.sleep(0.01)  

@app.route('/')
def index():
    print("Serving index page")
    return render_template('index.html')

@app.route('/guardian')
def guardian():
    print("Serving guardian page")
    return render_template('guardian.html')

@app.route('/fraud-detection')
def fraud_detection():
    print("Serving fraud detection page")
    return render_template('fraud_detection.html')

@app.route('/api/start', methods=['POST'])
def start_mining():
    global mining_active, mining_thread, start_time, hash_rate_history, found_solutions
    
    if not mining_active:
        mining_active = True
        start_time = time.time()
        hash_rate_history = []
        found_solutions = []
        mining_thread = threading.Thread(target=mining_task)
        mining_thread.start()
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "already running"})

@app.route('/api/stop', methods=['POST'])
def stop_mining():
    global mining_active
    
    if mining_active:
        mining_active = False
        if mining_thread:
            mining_thread.join(timeout=2.0)
        return jsonify({"status": "stopped"})
    else:
        return jsonify({"status": "not running"})

@app.route('/api/status', methods=['GET'])
def get_status():
    global mining_active, start_time, hash_rate_history, found_solutions
    
    elapsed = time.time() - start_time if mining_active else 0
    
    return jsonify({
        "mining": mining_active,
        "elapsed": elapsed,
        "hashRate": hash_rate_history[-1] if hash_rate_history else 0,
        "hashRateHistory": hash_rate_history,
        "solutions": found_solutions
    })

# Fraud Detection API Routes

@app.route('/api/fraud-detection/start-simulator', methods=['POST'])
def start_transaction_simulator():
    """Start the transaction simulator"""
    result = transaction_simulator.start_simulation()
  
    if result and transaction_analyzer.is_analyzing:
        start_transaction_processing()
    return jsonify({"status": "started" if result else "already_running"})

@app.route('/api/fraud-detection/stop-simulator', methods=['POST'])
def stop_transaction_simulator():
    """Stop the transaction simulator"""
    result = transaction_simulator.stop_simulation()
    
    if result:
        stop_transaction_processing()
    return jsonify({"status": "stopped" if result else "not_running"})

@app.route('/api/fraud-detection/start-analyzer', methods=['POST'])
def start_transaction_analyzer():
    """Start the transaction analyzer"""
    result = transaction_analyzer.start_analyzer()
 
    if result and transaction_simulator.is_running:
        start_transaction_processing()
    return jsonify({"status": "started" if result else "already_running"})

@app.route('/api/fraud-detection/stop-analyzer', methods=['POST'])
def stop_transaction_analyzer():
    """Stop the transaction analyzer"""
    result = transaction_analyzer.stop_analyzer()
   
    if result:
        stop_transaction_processing()
    return jsonify({"status": "stopped" if result else "not_running"})

def start_transaction_processing():
    """Start the transaction processing thread"""
    global transaction_process_thread, transaction_processing
    if not transaction_processing:
        transaction_processing = True
        transaction_process_thread = threading.Thread(target=transaction_processing_task)
        transaction_process_thread.daemon = True
        transaction_process_thread.start()
        return True
    return False

def stop_transaction_processing():
    """Stop the transaction processing thread"""
    global transaction_processing, transaction_process_thread
    if transaction_processing and not (transaction_simulator.is_running and transaction_analyzer.is_analyzing):
        transaction_processing = False
        if transaction_process_thread:
            transaction_process_thread.join(timeout=2.0)
        return True
    return False

@app.route('/api/fraud-detection/simulator-status', methods=['GET'])
def get_simulator_status():
    """Get current status of the transaction simulator"""
    return jsonify(transaction_simulator.get_status())

@app.route('/api/fraud-detection/analyzer-status', methods=['GET'])
def get_analyzer_status():
    """Get current status of the transaction analyzer"""
    return jsonify(transaction_analyzer.get_status())

@app.route('/api/fraud-detection/recent-transactions', methods=['GET'])
def get_recent_transactions():
    """Get most recent transactions"""
    limit = int(request.args.get('limit', 10))
    return jsonify({"transactions": transaction_simulator.get_recent_transactions(limit)})

@app.route('/api/fraud-detection/recent-analyses', methods=['GET'])
def get_recent_analyses():
    """Get most recent analysis results"""
    limit = int(request.args.get('limit', 20))
    return jsonify({"analyses": transaction_analyzer.get_all_results(limit)})

@app.route('/api/fraud-detection/recent-anomalies', methods=['GET'])
def get_recent_anomalies():
    """Get most recent anomalies"""
    limit = int(request.args.get('limit', 10))
    return jsonify({"anomalies": transaction_analyzer.get_recent_anomalies(limit)})

@app.route('/api/fraud-detection/set-threshold', methods=['POST'])
def set_anomaly_threshold():
    """Set the anomaly threshold"""
    data = request.get_json()
    if 'threshold' in data:
        threshold = float(data['threshold'])
        transaction_analyzer.anomaly_threshold = threshold
        return jsonify({"status": "success", "threshold": threshold})
    return jsonify({"status": "error", "message": "No threshold provided"})

@app.route('/api/fraud-detection/transactions-by-id', methods=['POST'])
def get_transactions_by_id():
    """Get transactions by their IDs"""
    data = request.get_json()
    if 'ids' in data and isinstance(data['ids'], list):
      
        matching_transactions = []
        for tx in transaction_simulator.transactions_history:
            if tx['transaction_id'] in data['ids']:
                matching_transactions.append(tx)
        
        return jsonify({"transactions": matching_transactions})
    return jsonify({"status": "error", "message": "Invalid ID list"})

if __name__ == '__main__':
    
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="en">
...HTML template content (omitted for brevity)...
</html>''')
    
   
    try:
        print("Importing CryptoGuardian...")
        from crypto_guardian import integrate_guardian
        integrate_guardian(app)
        print("CryptoGuardian imported and integrated successfully")
    except Exception as e:
        print(f"Error importing CryptoGuardian: {e}", file=sys.stderr)
        print("Guardian functionality will not be available")
    

    ports = [8080, 8081, 8082, 5000]
    
    for port in ports:
        try:
            print(f"Trying to start Flask server on port {port}...")
            app.run(host='0.0.0.0', port=port, debug=True)
            break 
        except OSError as e:
            print(f"Port {port} is not available: {e}")
            if port == ports[-1]:
                print("All port attempts failed. Please free up a port manually.")
                sys.exit(1)
            continue
