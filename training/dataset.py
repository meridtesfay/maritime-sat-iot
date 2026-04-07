"""
PyTorch Dataset and DataLoader utilities
Handles class imbalance with SMOTE and stratified sampling
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import json

class SatelliteLinkDataset(Dataset):
    def __init__(self, data_dir='data/processed', split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load data
        self.X = np.load(self.data_dir / f'{split}_X.npy')
        self.y = np.load(self.data_dir / f'{split}_y.npy')
        
        # Load metadata
        with open(self.data_dir / 'metadata.json') as f:
            self.metadata = json.load(f)
        
        print(f"{split} dataset: {len(self.X)} samples")
        print(f"  Class distribution: {np.bincount(self.y)}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])[0]
        return x, y
    
    def get_class_weights(self):
        """Compute class weights for balanced loss"""
        class_counts = np.bincount(self.y)
        total = len(self.y)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)

def get_dataloaders(data_dir='data/processed', batch_size=64, 
                   use_weighted_sampler=True, num_workers=4):
    """Create train/val/test dataloaders"""
    
    # Create datasets
    train_dataset = SatelliteLinkDataset(data_dir, 'train')
    val_dataset = SatelliteLinkDataset(data_dir, 'val')
    test_dataset = SatelliteLinkDataset(data_dir, 'test')
    
    # Weighted sampler for training (handles imbalance)
    if use_weighted_sampler:
        class_counts = np.bincount(train_dataset.y)
        weights = 1.0 / class_counts[train_dataset.y]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.get_class_weights()

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(batch_size=32)
    
    print("\nDataLoader test:")
    for x, y in train_loader:
        print(f"Batch shape: {x.shape}, Labels: {y.shape}")
        print(f"Label distribution in batch: {torch.bincount(y)}")
        break
    
    print(f"\nClass weights: {class_weights}")
