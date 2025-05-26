import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm

class IBVSDataset(Dataset):
    def __init__(self, data_path, sequence_length=1, feature_cols=None, target_cols=None, feature_scaler=None, target_scaler=None):
        """
        Args:
            data_path (str): Path to the CSV file with IBVS data
            sequence_length (int): Length of sequence for RNN-based models
            feature_cols (list): List of column names for features
            target_cols (list): List of column names for targets
            feature_scaler (StandardScaler): Pre-fitted scaler for features
            target_scaler (StandardScaler): Pre-fitted scaler for targets
        """
        self.sequence_length = sequence_length
        
        # Load the data and clean column names
        self.data = pd.read_csv(data_path)
        self.data.columns = self.data.columns.str.strip()
        
        # Print available columns to help with debugging
        print("\nAvailable columns in dataset (after cleaning):")
        for col in self.data.columns:
            print(f"- {col}")
        print()
        
        # Set default feature and target columns based on known naming scheme
        if feature_cols is None:
            feature_cols = [f'current_feature_{i}' for i in range(8)]
            print(f"Using default feature columns: {feature_cols}")
            
        if target_cols is None:
            target_cols = ['vx', 'vy', 'vz']  # Only linear velocities since angular are essentially zero
            print(f"Using default target columns: {target_cols}")
            print("Note: Angular velocities (wx, wy, wz) excluded as they contain only numerical noise")
        
        # Verify columns exist
        missing_features = [col for col in feature_cols if col not in self.data.columns]
        missing_targets = [col for col in target_cols if col not in self.data.columns]
        
        if missing_features or missing_targets:
            error_msg = "Missing columns in dataset:\n"
            if missing_features:
                error_msg += f"Feature columns: {missing_features}\n"
            if missing_targets:
                error_msg += f"Target columns: {missing_targets}\n"
            error_msg += f"\nAvailable columns: {list(self.data.columns)}"
            raise ValueError(error_msg)
        
        # Separate features and targets
        self.features = self.data[feature_cols].values
        self.targets = self.data[target_cols].values
        
        # Use provided scalers or create new ones
        self.feature_scaler = feature_scaler if feature_scaler is not None else StandardScaler()
        self.target_scaler = target_scaler if target_scaler is not None else StandardScaler()
        
        # Fit and transform if scalers are new
        if feature_scaler is None:
            self.features = self.feature_scaler.fit_transform(self.features)
        else:
            self.features = self.feature_scaler.transform(self.features)
            
        if target_scaler is None:
            self.targets = self.target_scaler.fit_transform(self.targets)
        else:
            self.targets = self.target_scaler.transform(self.targets)
        
        print(f"\nDataset loaded successfully:")
        print(f"- Number of samples: {len(self.data)}")
        print(f"- Feature shape: {self.features.shape}")
        print(f"- Target shape: {self.targets.shape}")
        print()
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        if self.sequence_length > 1:
            # For RNN-based models
            feature_seq = self.features[idx:idx + self.sequence_length]
            target = self.targets[idx + self.sequence_length - 1]
            return torch.FloatTensor(feature_seq), torch.FloatTensor(target)
        else:
            # For non-RNN models
            return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.targets[idx])

def prepare_dataloaders(data_path, batch_size=32, sequence_length=1, val_split=0.2, feature_scaler=None, target_scaler=None):
    """
    Prepare train and validation dataloaders
    
    Args:
        data_path (str): Path to the CSV file with IBVS data
        batch_size (int): Batch size for training
        sequence_length (int): Length of sequence for RNN-based models
        val_split (float): Fraction of data to use for validation
        feature_scaler (StandardScaler): Pre-fitted scaler for features
        target_scaler (StandardScaler): Pre-fitted scaler for targets
    """
    # Create dataset
    dataset = IBVSDataset(
        data_path,
        sequence_length=sequence_length,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler
    )
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch with progress bar
    """
    model.train()
    total_loss = 0
    
    # Create progress bar
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """
    Validate the model with progress bar
    """
    model.eval()
    total_loss = 0
    
    # Create progress bar
    pbar = tqdm(val_loader, desc='Validating', leave=False)
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(val_loader)

#PENDING IMPLEMENTATION