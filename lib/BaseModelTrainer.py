import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from lib.DataLoading import prepare_dataloaders, train_epoch, validate
from model_training.config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, VAL_SPLIT, DEVICE, RANDOM_SEED,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA
)
from sklearn.preprocessing import StandardScaler

class BaseModelTrainer(ABC):
    def __init__(self, model_name, model_config):
        """
        Base trainer class for all IBVS models
        
        Args:
            model_name (str): Name of the model (e.g., 'fnn', 'lstm')
            model_config (dict): Model-specific configuration
        """
        self.model_name = model_name
        self.config = model_config
        self.model = None
        
        # Scalers for feature and target normalization
        self.feature_scaler = None
        self.target_scaler = None
        
        # Create training_models directory
        os.makedirs('training_models', exist_ok=True)
        
        # Initialize training metadata
        self.training_metadata = {
            'model_name': model_name,
            'model_config': model_config,
            'training_config': {
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'learning_rate': LEARNING_RATE,
                'val_split': VAL_SPLIT,
                'device': DEVICE,
                'random_seed': RANDOM_SEED,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                'early_stopping_min_delta': EARLY_STOPPING_MIN_DELTA
            },
            'training_history': {
                'train_losses': [],
                'val_losses': [],
                'epochs': [],
                'best_val_loss': float('inf'),
                'best_epoch': 0,
                'stopped_early': False,
                'early_stopping_epoch': None
            },
            'performance': {
                'epoch_durations': [],  # Time per epoch in seconds
                'samples_per_second': [],  # Training throughput
                'learning_rates': [],  # Learning rate progression
                'memory_usage': [],  # GPU/CPU memory usage in MB
                'avg_epoch_time': 0,  # Average time per epoch
                'total_samples_processed': 0  # Total training samples processed
            },
            'timing': {
                'start_time': None,
                'end_time': None,
                'total_duration': None
            },
            'normalization': {
                'feature_scaler_path': None,
                'target_scaler_path': None
            }
        }
    
    def save_scalers(self):
        """Save the feature and target scalers"""
        if self.feature_scaler is not None:
            feature_scaler_path = f'training_models/{self.model_name}_feature_scaler.pkl'
            with open(feature_scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            self.training_metadata['normalization']['feature_scaler_path'] = feature_scaler_path
            print(f"\n✓ Saved feature scaler to {feature_scaler_path}")
            
        if self.target_scaler is not None:
            target_scaler_path = f'training_models/{self.model_name}_target_scaler.pkl'
            with open(target_scaler_path, 'wb') as f:
                pickle.dump(self.target_scaler, f)
            self.training_metadata['normalization']['target_scaler_path'] = target_scaler_path
            print(f"✓ Saved target scaler to {target_scaler_path}")
    
    def load_scalers(self):
        """Load the feature and target scalers"""
        feature_scaler_path = self.training_metadata['normalization'].get('feature_scaler_path')
        target_scaler_path = self.training_metadata['normalization'].get('target_scaler_path')
        
        if feature_scaler_path and os.path.exists(feature_scaler_path):
            with open(feature_scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)
            print(f"\n✓ Loaded feature scaler from {feature_scaler_path}")
            
        if target_scaler_path and os.path.exists(target_scaler_path):
            with open(target_scaler_path, 'rb') as f:
                self.target_scaler = pickle.load(f)
            print(f"✓ Loaded target scaler from {target_scaler_path}")
    
    def normalize_features(self, features):
        """Normalize input features"""
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            return self.feature_scaler.fit_transform(features)
        return self.feature_scaler.transform(features)
    
    def normalize_targets(self, targets):
        """Normalize target values"""
        if self.target_scaler is None:
            self.target_scaler = StandardScaler()
            return self.target_scaler.fit_transform(targets)
        return self.target_scaler.transform(targets)
    
    def denormalize_predictions(self, predictions):
        """Denormalize model predictions back to original scale"""
        if self.target_scaler is not None:
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            return self.target_scaler.inverse_transform(predictions)
        return predictions
    
    @abstractmethod
    def create_model(self):
        """Create and return the model instance"""
        pass
    
    def get_dataloaders(self, data_path):
        """
        Prepare data loaders with model-specific settings
        """
        sequence_length = self.config.get('sequence_length', 1)
        train_loader, val_loader = prepare_dataloaders(
            data_path,
            batch_size=BATCH_SIZE,
            sequence_length=sequence_length,
            val_split=VAL_SPLIT,
            feature_scaler=self.feature_scaler,
            target_scaler=self.target_scaler
        )
        
        # Get the scalers from the dataset if they were created
        if hasattr(train_loader.dataset, 'dataset'):
            # Handle random split wrapper
            dataset = train_loader.dataset.dataset
        else:
            dataset = train_loader.dataset
            
        if self.feature_scaler is None:
            self.feature_scaler = dataset.feature_scaler
        if self.target_scaler is None:
            self.target_scaler = dataset.target_scaler
            
        return train_loader, val_loader
    
    def save_metadata(self):
        """Save training metadata to JSON file"""
        metadata_path = f'training_models/{self.model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=4)
        print(f"\n✓ Saved training metadata to {metadata_path}")
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, epochs, save_path=None):
        """
        Plot training and validation losses
        
        Args:
            train_losses (list): List of training losses
            val_losses (list): List of validation losses
            epochs (list): List of epoch numbers
            save_path (str, optional): If provided, save the plot to this path
        """
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        # Add best validation loss point
        best_epoch_idx = val_losses.index(min(val_losses))
        best_val_loss = val_losses[best_epoch_idx]
        plt.plot(epochs[best_epoch_idx], best_val_loss, 'g*', 
                label=f'Best Val Loss: {best_val_loss:.6f}',
                markersize=15)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"\n✓ Saved training plot to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def train(self, data_path):
        """
        Train the model with early stopping and performance tracking
        """
        # Set random seed and start timing
        torch.manual_seed(RANDOM_SEED)
        self.training_metadata['timing']['start_time'] = time.time()
        
        # Create model
        self.model = self.create_model().to(DEVICE)
        
        # Prepare data
        train_loader, val_loader = self.get_dataloaders(data_path)
        
        # Loss and optimizer with weight decay
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        
        # Learning rate scheduler - more aggressive to combat overfitting
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=5
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        print(f"\nStarting training for max {NUM_EPOCHS} epochs...")
        print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
        print("=" * 50)
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            print("-" * 20)
            
            # Start epoch timing
            epoch_start_time = time.time()
            
            # Training and validation with progress bars
            train_loss = train_epoch(self.model, train_loader, criterion, optimizer, DEVICE)
            val_loss = validate(self.model, val_loader, criterion, DEVICE)
            
            # Calculate epoch performance metrics
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            # Calculate samples per second (training samples only)
            train_samples = len(train_loader.dataset)
            samples_per_second = train_samples / epoch_duration if epoch_duration > 0 else 0
            
            # Get memory usage
            memory_usage = 0
            try:
                if torch.cuda.is_available() and DEVICE == 'cuda':
                    memory_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                else:
                    import psutil
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            except:
                memory_usage = 0  # Fallback if memory tracking fails
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update training history
            self.training_metadata['training_history']['train_losses'].append(train_loss)
            self.training_metadata['training_history']['val_losses'].append(val_loss)
            self.training_metadata['training_history']['epochs'].append(epoch + 1)
            
            # Update performance metrics
            self.training_metadata['performance']['epoch_durations'].append(epoch_duration)
            self.training_metadata['performance']['samples_per_second'].append(samples_per_second)
            self.training_metadata['performance']['learning_rates'].append(current_lr)
            self.training_metadata['performance']['memory_usage'].append(memory_usage)
            self.training_metadata['performance']['total_samples_processed'] += train_samples
            
            # Calculate average epoch time
            avg_epoch_time = sum(self.training_metadata['performance']['epoch_durations']) / len(self.training_metadata['performance']['epoch_durations'])
            self.training_metadata['performance']['avg_epoch_time'] = avg_epoch_time
            
            # Save metadata after each epoch for real-time monitoring
            self.save_metadata()
            
            # Check for improvement
            if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = val_loss
                self.training_metadata['training_history']['best_val_loss'] = val_loss
                self.training_metadata['training_history']['best_epoch'] = epoch + 1
                torch.save(self.model.state_dict(), f'training_models/{self.model_name}_best.pth')
                print("\n✓ Saved new best model!")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                    self.training_metadata['training_history']['stopped_early'] = True
                    self.training_metadata['training_history']['early_stopping_epoch'] = epoch + 1
                    break
            
            # Epoch summary with performance metrics
            print(f"\nEpoch Summary:")
            print(f"Train Loss:     {train_loss:.6f}")
            print(f"Val Loss:       {val_loss:.6f}")
            print(f"Best Val:       {best_val_loss:.6f}")
            print(f"Learning Rate:  {current_lr:.2e}")
            print(f"Epoch Time:     {epoch_duration:.2f}s")
            print(f"Samples/sec:    {samples_per_second:.0f}")
            print(f"Memory Usage:   {memory_usage:.1f} MB")
            print(f"Patience:       {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            print("=" * 50)
        
        # Record end time and duration
        self.training_metadata['timing']['end_time'] = time.time()
        self.training_metadata['timing']['total_duration'] = (
            self.training_metadata['timing']['end_time'] - 
            self.training_metadata['timing']['start_time']
        )
        
        # Save scalers and metadata
        self.save_scalers()
        self.save_metadata()
        
        # Plot and save training history
        plot_path = f'training_models/{self.model_name}_training_history.png'
        self.plot_training_history(
            self.training_metadata['training_history']['train_losses'],
            self.training_metadata['training_history']['val_losses'],
            self.training_metadata['training_history']['epochs'],
            save_path=plot_path
        )
        
        return best_val_loss
