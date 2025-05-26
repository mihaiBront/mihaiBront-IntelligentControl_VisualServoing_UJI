from flask import Flask, render_template, jsonify
import os
import json
import time
from threading import Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)

# Global variables to store training state
training_state = {
    'current_model': None,
    'metadata': {},
    'lock': Lock()
}

class TrainingMonitor(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('_metadata.json'):
            with training_state['lock']:
                try:
                    with open(event.src_path, 'r') as f:
                        training_state['metadata'] = json.load(f)
                except Exception as e:
                    print(f"Error reading metadata file: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/training-status')
def training_status():
    with training_state['lock']:
        return jsonify(training_state['metadata'])

def start_monitoring():
    # Create training_models directory if it doesn't exist
    os.makedirs('training_models', exist_ok=True)
    
    # Set up file system monitoring
    event_handler = TrainingMonitor()
    observer = Observer()
    observer.schedule(event_handler, path='training_models', recursive=False)
    observer.start()
    return observer

if __name__ == '__main__':
    observer = start_monitoring()
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        observer.stop()
        observer.join() 