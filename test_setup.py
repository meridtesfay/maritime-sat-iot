"""
Quick test - verify all components work
Run this before submitting to HPC
"""

import torch
import sys
sys.path.append('.')

print("Testing data loading...")
from training.dataset import get_dataloaders
train_loader, val_loader, test_loader, class_weights = get_dataloaders(batch_size=32, num_workers=0)
print(f"✓ Data loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
print(f"✓ Class weights: {class_weights}")

print("\nTesting models...")
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.hybrid_model import HybridModel

models = {
    'CNN': CNNModel(),
    'LSTM': LSTMModel(),
    'Transformer': TransformerModel(),
    'Hybrid': HybridModel()
}

x_test = torch.randn(8, 24, 15)

for name, model in models.items():
    out = model(x_test)
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ {name:12s} - output shape: {out.shape}, params: {params:,}")

print("\nTesting focal loss...")
from training.focal_loss import FocalLoss
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
loss = criterion(out, torch.randint(0, 3, (8,)))
print(f"✓ Focal loss computed: {loss.item():.4f}")

print("\nTesting training loop (1 batch)...")
model = models['Hybrid']
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for x, y in train_loader:
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(f"✓ Training step - loss: {loss.item():.4f}")
    break

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nReady for HPC training!")
print("\nQuick start:")
print("  Local:  python training/train.py --model hybrid --epochs 10")
print("  HPC:    sbatch training/slurm_train.sh")
