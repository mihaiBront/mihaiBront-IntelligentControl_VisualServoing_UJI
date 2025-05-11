import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from lib.DataLoading import prepare_dataloaders, train_epoch, validate
from config import LSTM_CONFIG, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, VAL_SPLIT, DEVICE, RANDOM_SEED
import os

class LSTMController(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMController, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

def train_lstm(data_path):
    """
    Train the LSTM Controller
    """
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    
    # Create model
    model = LSTMController(
        input_size=LSTM_CONFIG['input_size'],
        hidden_size=LSTM_CONFIG['hidden_size'],
        num_layers=LSTM_CONFIG['num_layers'],
        output_size=LSTM_CONFIG['output_size'],
        dropout_rate=LSTM_CONFIG['dropout_rate']
    ).to(DEVICE)
    
    # Prepare data
    train_loader, val_loader = prepare_dataloaders(
        data_path,
        batch_size=BATCH_SIZE,
        sequence_length=LSTM_CONFIG['sequence_length'],
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
            torch.save(model.state_dict(), 'training_models/lstm_best.pth')
            print("\n✓ Saved new best model!")
        
        # Epoch summary
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Best Val:   {best_val_loss:.6f}")
        print("=" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM for IBVS control')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    train_lstm(args.data_path) 