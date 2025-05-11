import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from lib.DataLoading import prepare_dataloaders, train_epoch, validate
from config import FNN_CONFIG, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, VAL_SPLIT, DEVICE, RANDOM_SEED
import os

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(FeedforwardNN, self).__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_fnn(data_path):
    """
    Train the Feedforward Neural Network
    """
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    
    # Create model
    model = FeedforwardNN(
        input_size=FNN_CONFIG['input_size'],
        hidden_sizes=FNN_CONFIG['hidden_sizes'],
        output_size=FNN_CONFIG['output_size'],
        dropout_rate=FNN_CONFIG['dropout_rate']
    ).to(DEVICE)
    
    # Prepare data
    train_loader, val_loader = prepare_dataloaders(
        data_path,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("=" * 50)
    
    # Create training_models directory if it doesn't exist
    os.makedirs('training_models', exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 20)
        
        # Training and validation with progress bars
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'training_models/fnn_best.pth')
            print("\n✓ Saved new best model!")
        
        # Epoch summary
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Best Val:   {best_val_loss:.6f}")
        print("=" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FNN for IBVS control')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    train_fnn(args.data_path) 