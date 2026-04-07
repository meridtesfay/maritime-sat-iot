"""
Training script for all models
Supports distributed training on HPC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm

import sys
sys.path.append('.')
from training.dataset import get_dataloaders
from training.focal_loss import FocalLoss
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.hybrid_model import HybridModel

def get_model(model_name, num_classes=3):
    """Initialize model by name"""
    models = {
        'cnn': CNNModel(num_classes=num_classes),
        'lstm': LSTMModel(num_classes=num_classes),
        'transformer': TransformerModel(num_classes=num_classes),
        'hybrid': HybridModel(num_classes=num_classes),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name]

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Validation'):
            x, y = x.to(device), y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Compute per-class metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    per_class_acc = []
    for c in range(3):
        mask = all_targets == c
        if mask.sum() > 0:
            acc = (all_preds[mask] == c).mean()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)
    
    return total_loss / len(loader), 100. * correct / total, per_class_acc

def train_model(model_name, epochs=50, batch_size=64, lr=0.001, 
                device='cuda', save_dir='checkpoints'):
    """Main training loop"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}\n")
    
    # Setup
    save_dir = Path(save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=batch_size,
        use_weighted_sampler=True
    )
    
    # Model
    model = get_model(model_name).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with class weights and focal loss
    criterion = FocalLoss(
        alpha=class_weights.to(device),
        gamma=2.0
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, per_class_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Per-class Acc: Good={per_class_acc[0]*100:.1f}%, "
              f"Degraded={per_class_acc[1]*100:.1f}%, "
              f"Outage={per_class_acc[2]*100:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'per_class_acc': per_class_acc
            }, save_dir / 'best_model.pth')
            print(f"Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Save final model
    torch.save(model.state_dict(), save_dir / 'final_model.pth')
    
    # Save history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(save_dir / 'best_model.pth')['model_state_dict'])
    test_loss, test_acc, test_per_class = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"Test Acc: {test_acc:.2f}%")
    print(f"Per-class: Good={test_per_class[0]*100:.1f}%, "
          f"Degraded={test_per_class[1]*100:.1f}%, "
          f"Outage={test_per_class[2]*100:.1f}%")
    
    results = {
        'model': model_name,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_per_class_acc': [float(x) for x in test_per_class]
    }
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hybrid',
                       choices=['cnn', 'lstm', 'transformer', 'hybrid'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    results = train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
