import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from lib.DataLoading import prepare_dataloaders, train_epoch, validate
from model_training.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, VAL_SPLIT, DEVICE, RANDOM_SEED

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
                'random_seed': RANDOM_SEED
            },
            'training_history': {
                'train_losses': [],
                'val_losses': [],
                'epochs': [],
                'best_val_loss': float('inf'),
                'best_epoch': 0
            },
            'timing': {
                'start_time': None,
                'end_time': None,
                'total_duration': None
            }
        }
    
    @abstractmethod
    def create_model(self):
        """Create and return the model instance"""
        pass
    
    def get_dataloaders(self, data_path):
        """
        Prepare data loaders with model-specific settings
        """
        sequence_length = self.config.get('sequence_length', 1)
        return prepare_dataloaders(
            data_path,
            batch_size=BATCH_SIZE,
            sequence_length=sequence_length,
            val_split=VAL_SPLIT
        )
    
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
        Train the model
        """
        # Set random seed and start timing
        torch.manual_seed(RANDOM_SEED)
        self.training_metadata['timing']['start_time'] = time.time()
        
        # Create model
        self.model = self.create_model().to(DEVICE)
        
        # Prepare data
        train_loader, val_loader = self.get_dataloaders(data_path)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Training loop
        best_val_loss = float('inf')
        print(f"\nStarting training for {NUM_EPOCHS} epochs...")
        print("=" * 50)
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            print("-" * 20)
            
            # Training and validation with progress bars
            train_loss = train_epoch(self.model, train_loader, criterion, optimizer, DEVICE)
            val_loss = validate(self.model, val_loader, criterion, DEVICE)
            
            # Update training history
            self.training_metadata['training_history']['train_losses'].append(train_loss)
            self.training_metadata['training_history']['val_losses'].append(val_loss)
            self.training_metadata['training_history']['epochs'].append(epoch + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.training_metadata['training_history']['best_val_loss'] = val_loss
                self.training_metadata['training_history']['best_epoch'] = epoch + 1
                torch.save(self.model.state_dict(), f'training_models/{self.model_name}_best.pth')
                print("\n✓ Saved new best model!")
            
            # Epoch summary
            print(f"\nEpoch Summary:")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss:   {val_loss:.6f}")
            print(f"Best Val:   {best_val_loss:.6f}")
            print("=" * 50)
        
        # Record end time and duration
        self.training_metadata['timing']['end_time'] = time.time()
        self.training_metadata['timing']['total_duration'] = (
            self.training_metadata['timing']['end_time'] - 
            self.training_metadata['timing']['start_time']
        )
        
        # Save final metadata
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
