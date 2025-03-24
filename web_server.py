from flask import Flask, render_template, jsonify, request
import ctypes
import os
import time
import threading
import sys

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

@app.route('/')
def index():
    print("Serving index page")
    return render_template('index.html')

@app.route('/guardian')
def guardian():
    print("Serving guardian page")
    return render_template('guardian.html')

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

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Ensure template files exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="en">
...HTML template content (omitted for brevity)...
</html>''')
    
    # Try to import and integrate CryptoGuardian
    try:
        print("Importing CryptoGuardian...")
        from crypto_guardian import integrate_guardian
        integrate_guardian(app)
        print("CryptoGuardian imported and integrated successfully")
    except Exception as e:
        print(f"Error importing CryptoGuardian: {e}", file=sys.stderr)
        print("Guardian functionality will not be available")
    
    # Start the Flask server with debugging enabled
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=8080, debug=True)
