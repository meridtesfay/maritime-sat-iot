"""
Generate realistic satellite link quality data with proper class distribution
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

def generate_realistic_link_data(num_samples=15000):
    """Generate data with realistic class imbalance"""
    
    print(f"Generating {num_samples} samples...")
    
    # Target distribution: 85% good, 12% degraded, 3% outage
    n_good = int(num_samples * 0.85)
    n_degraded = int(num_samples * 0.12)
    n_outage = num_samples - n_good - n_degraded
    
    records = []
    
    # Good links (high SNR, low rain)
    for i in range(n_good):
        rain = np.random.exponential(0.5) if np.random.rand() < 0.1 else 0
        snr = np.random.normal(18, 2)
        records.append({'rain_rate_mm_hr': rain, 'snr_db': snr, 'quality': 0})
    
    # Degraded links (medium SNR, light-moderate rain)
    for i in range(n_degraded):
        rain = np.random.uniform(2, 10)
        snr = np.random.normal(13, 1.5)
        records.append({'rain_rate_mm_hr': rain, 'snr_db': snr, 'quality': 1})
    
    # Outage (low SNR, heavy rain)
    for i in range(n_outage):
        rain = np.random.uniform(10, 30)
        snr = np.random.normal(8, 2)
        records.append({'rain_rate_mm_hr': rain, 'snr_db': snr, 'quality': 2})
    
    df = pd.DataFrame(records)
    
    # Add full feature set
    df['latitude'] = np.random.uniform(0, 60, num_samples)
    df['longitude'] = np.random.uniform(-80, 140, num_samples)
    df['speed_knots'] = np.random.uniform(10, 20, num_samples)
    df['heading'] = np.random.uniform(0, 360, num_samples)
    df['temperature_c'] = np.random.normal(15, 10, num_samples)
    df['humidity_percent'] = 70 - df['temperature_c'] * 0.5 + np.random.randn(num_samples) * 10
    df['humidity_percent'] = np.clip(df['humidity_percent'], 20, 100)
    df['pressure_hpa'] = np.random.normal(1013, 10, num_samples)
    df['wind_speed_ms'] = np.random.uniform(2, 15, num_samples)
    df['hour'] = np.random.randint(0, 24, num_samples)
    df['day_of_year'] = np.random.randint(1, 365, num_samples)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sat_elevation'] = np.random.uniform(20, 80, num_samples)
    
    # Rain attenuation
    k, alpha = 0.0188, 1.217
    df['rain_atten_db'] = k * (df['rain_rate_mm_hr'] ** alpha)
    df['atmospheric_loss_db'] = 0.5 + 0.005 * df['humidity_percent']
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Class distribution:")
    print(df['quality'].value_counts().sort_index())
    
    return df

def create_sequences_from_data(df, seq_len=24):
    """Create sequences from flat data"""
    
    feature_cols = [
        'latitude', 'longitude', 'speed_knots', 'heading',
        'temperature_c', 'humidity_percent', 'pressure_hpa',
        'wind_speed_ms', 'rain_rate_mm_hr',
        'hour_sin', 'hour_cos', 'day_of_year',
        'sat_elevation', 'rain_atten_db', 'atmospheric_loss_db'
    ]
    
    # Create synthetic sequences
    sequences = []
    labels = []
    
    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i+seq_len][feature_cols].values
        label = df.iloc[i+seq_len]['quality']
        sequences.append(seq)
        labels.append(label)
    
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

def main():
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    df = generate_realistic_link_data(num_samples=18000)
    
    # Create sequences
    print("\nCreating sequences...")
    X, y = create_sequences_from_data(df, seq_len=24)
    print(f"Sequences: {X.shape}")
    print(f"Labels: {y.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Split
    n = len(X)
    idx = np.random.permutation(n)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_X, train_y = X[idx[:train_end]], y[idx[:train_end]]
    val_X, val_y = X[idx[train_end:val_end]], y[idx[train_end:val_end]]
    test_X, test_y = X[idx[val_end:]], y[idx[val_end:]]
    
    print(f"\nTrain: {len(train_X)} - {np.bincount(train_y)}")
    print(f"Val: {len(val_X)} - {np.bincount(val_y)}")
    print(f"Test: {len(test_X)} - {np.bincount(test_y)}")
    
    # Save
    np.save(output_dir / 'train_X.npy', train_X)
    np.save(output_dir / 'train_y.npy', train_y)
    np.save(output_dir / 'val_X.npy', val_X)
    np.save(output_dir / 'val_y.npy', val_y)
    np.save(output_dir / 'test_X.npy', test_X)
    np.save(output_dir / 'test_y.npy', test_y)
    
    metadata = {
        'sequence_length': 24,
        'num_features': 15,
        'num_classes': 3,
        'train_samples': int(len(train_X)),
        'val_samples': int(len(val_X)),
        'test_samples': int(len(test_X)),
        'class_names': ['Good', 'Degraded', 'Outage']
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nData saved to {output_dir}")

if __name__ == "__main__":
    main()
