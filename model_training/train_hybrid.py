import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from lib.DataLoading import prepare_dataloaders, train_epoch, validate
from config import HYBRID_CONFIG, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, VAL_SPLIT, DEVICE, RANDOM_SEED
import os

class HybridController(nn.Module):
    def __init__(self, input_size, cnn_channels, lstm_hidden_size, lstm_num_layers,
                 output_size, dropout_rate, sequence_length):
        super(HybridController, self).__init__()
        
        # Reshape input for CNN
        self.sequence_length = sequence_length
        
        # CNN layers for spatial feature extraction
        cnn_layers = []
        in_channels = 1  # Single channel input
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output size
        cnn_output_size = (input_size // (2 ** len(cnn_channels))) * cnn_channels[-1]
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )
        
        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch, sequence, features) -> (batch * sequence, 1, features)
        x = x.view(-1, 1, x.size(-1))
        
        # Apply CNN
        x = self.cnn(x)
        
        # Reshape for LSTM: (batch * sequence, channels, features) -> (batch, sequence, features)
        x = x.view(batch_size, self.sequence_length, -1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output
        x = lstm_out[:, -1, :]
        
        # Final fully connected layers
        return self.fc(x)

def train_hybrid(data_path):
    """
    Train the Hybrid Controller
    """
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    
    # Create model
    model = HybridController(
        input_size=HYBRID_CONFIG['input_size'],
        cnn_channels=HYBRID_CONFIG['cnn_channels'],
        lstm_hidden_size=HYBRID_CONFIG['lstm_hidden_size'],
        lstm_num_layers=HYBRID_CONFIG['lstm_num_layers'],
        output_size=HYBRID_CONFIG['output_size'],
        dropout_rate=HYBRID_CONFIG['dropout_rate'],
        sequence_length=HYBRID_CONFIG['sequence_length']
    ).to(DEVICE)
    
    # Prepare data
    train_loader, val_loader = prepare_dataloaders(
        data_path,
        batch_size=BATCH_SIZE,
        sequence_length=HYBRID_CONFIG['sequence_length'],
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
            torch.save(model.state_dict(), 'training_models/hybrid_best.pth')
            print("\n✓ Saved new best model!")
        
        # Epoch summary
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Best Val:   {best_val_loss:.6f}")
        print("=" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-LSTM for IBVS control')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    train_hybrid(args.data_path) 