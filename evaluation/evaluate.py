"""
Comprehensive evaluation script
Computes all metrics including balanced accuracy, macro F1, per-class recall
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report, matthews_corrcoef
)
from pathlib import Path
import json
import argparse

import sys
sys.path.append('.')
from training.dataset import get_dataloaders
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.hybrid_model import HybridModel

def get_model(model_name, checkpoint_path):
    """Load trained model"""
    models = {
        'cnn': CNNModel(),
        'lstm': LSTMModel(),
        'transformer': TransformerModel(),
        'hybrid': HybridModel(),
    }
    
    model = models[model_name]
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def evaluate_model(model, loader, device='cuda'):
    """Get predictions"""
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics"""
    
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(acc),
        'balanced_accuracy': float(balanced_acc),
        'mcc': float(mcc),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class': {
            'good': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1': float(f1[0]),
                'support': int(support[0])
            },
            'degraded': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1': float(f1[1]),
                'support': int(support[1])
            },
            'outage': {
                'precision': float(precision[2]),
                'recall': float(recall[2]),
                'f1': float(f1[2]),
                'support': int(support[2])
            }
        },
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def print_metrics(metrics, model_name):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Overall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro F1:          {metrics['macro_f1']:.3f}")
    print(f"  MCC:               {metrics['mcc']:.3f}")
    
    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"\n  {class_name.capitalize()}:")
        print(f"    Precision: {class_metrics['precision']:.3f}")
        print(f"    Recall:    {class_metrics['recall']:.3f}")
        print(f"    F1:        {class_metrics['f1']:.3f}")
        print(f"    Support:   {class_metrics['support']}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print("           Pred:")
    print("             Good  Deg  Out")
    for i, row_name in enumerate(['Good', 'Deg', 'Out']):
        print(f"  True {row_name:4s} {cm[i,0]:4d}  {cm[i,1]:4d}  {cm[i,2]:4d}")
    
    # Key metric for maritime safety
    outage_recall = metrics['per_class']['outage']['recall']
    print(f"\n⭐ CRITICAL: Outage Detection Rate = {outage_recall*100:.1f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['cnn', 'lstm', 'transformer', 'hybrid'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load model
    if args.checkpoint is None:
        checkpoint_path = Path('checkpoints') / args.model / 'best_model.pth'
    else:
        checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading model from {checkpoint_path}")
    model = get_model(args.model, checkpoint_path)
    
    # Load test data
    _, _, test_loader, _ = get_dataloaders(batch_size=64)
    
    # Evaluate
    print("Evaluating on test set...")
    y_pred, y_true, y_prob = evaluate_model(model, test_loader, args.device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Print results
    print_metrics(metrics, args.model)
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f'{args.model}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_dir / f'{args.model}_metrics.json'}")

if __name__ == "__main__":
    main()
