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

@app.route('/api/architecture/<model_name>')
def get_architecture(model_name):
    """Serve architecture diagram content for a specific model"""
    try:
        architecture_file = f'training_models/{model_name.lower()}_architecture.md'
        if os.path.exists(architecture_file):
            with open(architecture_file, 'r') as f:
                content = f.read()
            return jsonify({'content': content, 'model': model_name})
        else:
            return jsonify({'error': f'Architecture file not found for model: {model_name}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-history')
def get_model_history():
    """Get list of all available models with their metadata"""
    try:
        models = []
        training_models_dir = 'training_models'
        
        if not os.path.exists(training_models_dir):
            return jsonify({'models': []})
        
        # Find all metadata files
        for filename in os.listdir(training_models_dir):
            if filename.endswith('_metadata.json'):
                model_name = filename.replace('_metadata.json', '')
                filepath = os.path.join(training_models_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract key information
                    history = metadata.get('training_history', {})
                    epochs = history.get('epochs', [])
                    train_losses = history.get('train_losses', [])
                    val_losses = history.get('val_losses', [])
                    
                    # Determine status
                    status = 'completed'
                    if history.get('stopped_early', False):
                        status = 'early_stopped'
                    elif len(epochs) == 0:
                        status = 'not_started'
                    
                    # Check if this is the currently training model
                    with training_state['lock']:
                        current_metadata = training_state.get('metadata', {})
                        if current_metadata.get('model_name') == model_name:
                            current_history = current_metadata.get('training_history', {})
                            if len(current_history.get('epochs', [])) > 0 and not current_history.get('stopped_early', False):
                                status = 'training'
                    
                    model_info = {
                        'name': model_name,
                        'status': status,
                        'epochs': len(epochs),
                        'best_loss': history.get('best_val_loss'),
                        'final_loss': val_losses[-1] if val_losses else None,
                        'last_modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(filepath)))
                    }
                    
                    models.append(model_info)
                    
                except Exception as e:
                    print(f"Error reading metadata for {model_name}: {e}")
                    continue
        
        # Sort by last modified (most recent first)
        models.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return jsonify({'models': models})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-data/<model_name>')
def get_model_data(model_name):
    """Get complete training data for a specific model"""
    try:
        metadata_file = f'training_models/{model_name}_metadata.json'
        
        if not os.path.exists(metadata_file):
            return jsonify({'error': f'Model data not found: {model_name}'}), 404
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return jsonify(metadata)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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