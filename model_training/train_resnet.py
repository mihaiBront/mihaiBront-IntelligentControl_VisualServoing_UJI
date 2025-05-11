import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from lib.DataLoading import prepare_dataloaders, train_epoch, validate
from config import RESNET_CONFIG, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, VAL_SPLIT, DEVICE, RANDOM_SEED
import os

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection

class ResNetController(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, output_size, dropout_rate):
        super(ResNetController, self).__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        return self.output_proj(x)

def train_resnet(data_path):
    """
    Train the ResNet Controller
    """
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    
    # Create model
    model = ResNetController(
        input_size=RESNET_CONFIG['input_size'],
        hidden_size=RESNET_CONFIG['hidden_size'],
        num_blocks=RESNET_CONFIG['num_blocks'],
        output_size=RESNET_CONFIG['output_size'],
        dropout_rate=RESNET_CONFIG['dropout_rate']
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
            torch.save(model.state_dict(), 'training_models/resnet_best.pth')
            print("\n✓ Saved new best model!")
        
        # Epoch summary
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Best Val:   {best_val_loss:.6f}")
        print("=" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet for IBVS control')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    train_resnet(args.data_path) 