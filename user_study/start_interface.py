from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import sys
import logging
from multiprocessing import Process, Queue

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # allow communication on the same IP (main use case for this package)
app.static_folder = 'static'

input_queue = Queue()
output_queue = Queue()

@app.route('/study')
def index():
    return render_template('index.html')

@socketio.on('communication')
def handle_message(message):
    # print(f"received to {message['name']}: {message['action_type']}")
    input_queue.put(message)
    response = output_queue.get()
    emit('query', response, broadcast=True)



if __name__ == '__main__':
    from preference_engine import worker

    worker_process = Process(target=worker, args=(input_queue, output_queue))
    worker_process.start()
    
    try:
        # TODO: add argparse for host and port
        socketio.run(app, host='0.0.0.0', port=8001)
    finally:
        input_queue.put('STOP')
        worker_process.join()
