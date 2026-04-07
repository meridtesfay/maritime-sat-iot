"""
Data Preprocessing Pipeline
Fuses AIS vessel data with weather grid, creates features, generates link quality labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from scipy.spatial import cKDTree

class DataPreprocessor:
    def __init__(self, ais_path, weather_path, output_dir='data/processed'):
        self.ais_path = Path(ais_path)
        self.weather_path = Path(weather_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        print("Loading AIS data...")
        self.ais_df = pd.read_csv(self.ais_path)
        self.ais_df['timestamp'] = pd.to_datetime(self.ais_df['timestamp'])
        
        print("Loading weather data...")
        self.weather_df = pd.read_csv(self.weather_path)
        self.weather_df['timestamp'] = pd.to_datetime(self.weather_df['timestamp'])
        
        print(f"AIS records: {len(self.ais_df):,}")
        print(f"Weather records: {len(self.weather_df):,}")
        
    def match_weather_to_vessels(self):
        """Spatial-temporal matching of weather to vessel positions"""
        print("\nMatching weather to vessel positions...")
        
        # Round timestamps to nearest 3 hours (weather resolution)
        self.ais_df['timestamp_rounded'] = self.ais_df['timestamp'].dt.round('3h')
        
        matched_records = []
        
        for timestamp in self.ais_df['timestamp_rounded'].unique():
            # Get vessels at this time
            vessels_at_t = self.ais_df[self.ais_df['timestamp_rounded'] == timestamp]
            
            # Get weather at this time
            weather_at_t = self.weather_df[self.weather_df['timestamp'] == timestamp]
            
            if len(weather_at_t) == 0:
                continue
            
            # Build KD-tree for spatial matching
            weather_coords = weather_at_t[['latitude', 'longitude']].values
            tree = cKDTree(weather_coords)
            
            # Find nearest weather point for each vessel
            vessel_coords = vessels_at_t[['latitude', 'longitude']].values
            distances, indices = tree.query(vessel_coords)
            
            # Merge
            for i, (_, vessel) in enumerate(vessels_at_t.iterrows()):
                weather = weather_at_t.iloc[indices[i]]
                
                record = {
                    'mmsi': vessel['mmsi'],
                    'timestamp': vessel['timestamp'],
                    'latitude': vessel['latitude'],
                    'longitude': vessel['longitude'],
                    'speed_knots': vessel['speed_knots'],
                    'heading': vessel['heading'],
                    'temperature_c': weather['temperature_c'],
                    'humidity_percent': weather['humidity_percent'],
                    'pressure_hpa': weather['pressure_hpa'],
                    'wind_speed_ms': weather['wind_speed_ms'],
                    'rain_rate_mm_hr': weather['rain_rate_mm_hr'],
                }
                matched_records.append(record)
        
        self.matched_df = pd.DataFrame(matched_records)
        print(f"Matched records: {len(self.matched_df):,}")
        
    def create_features(self):
        """Engineer features for ML models"""
        print("\nCreating features...")
        
        df = self.matched_df.copy()
        
        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Satellite geometry (simplified - assume Iridium-like constellation)
        # Elevation angle depends on latitude (higher at equator)
        df['sat_elevation'] = 30 + 20 * np.cos(np.radians(df['latitude']))
        df['sat_elevation'] = np.clip(df['sat_elevation'], 10, 85)
        
        # Rain attenuation (ITU-R P.838 simplified for 12 GHz)
        k = 0.0188
        alpha = 1.217
        df['rain_atten_db'] = k * (df['rain_rate_mm_hr'] ** alpha)
        
        # Path loss (simplified - distance-based)
        # Assume satellite at 780 km altitude (Iridium orbit)
        sat_altitude_km = 780
        elevation_rad = np.radians(df['sat_elevation'])
        slant_range_km = sat_altitude_km / np.sin(elevation_rad)
        
        frequency_ghz = 12.0
        df['free_space_loss_db'] = 20 * np.log10(slant_range_km * 1000) + \
                                   20 * np.log10(frequency_ghz * 1e9) - 147.55
        
        # Atmospheric absorption (humidity-based)
        df['atmospheric_loss_db'] = 0.5 + 0.005 * df['humidity_percent']
        
        # Total attenuation
        df['total_atten_db'] = df['rain_atten_db'] + df['atmospheric_loss_db']
        
        # Received SNR (assume EIRP=50 dBW, antenna gain=15 dBi, noise=-110 dBm)
        link_budget_db = 50 + 15 - df['free_space_loss_db'] + 140  # Convert to SNR
        df['received_snr_db'] = link_budget_db - df['total_atten_db']
        
        # Add noise
        df['received_snr_db'] += np.random.randn(len(df)) * 0.5
        
        # Link quality classification (adjusted thresholds)
        df['link_quality'] = pd.cut(
            df['received_snr_db'],
            bins=[-np.inf, 12, 17, np.inf],
            labels=[2, 1, 0]  # 0: Good, 1: Degraded, 2: Outage
        ).astype(int)
        
        self.feature_df = df
        
        print(f"\nLink quality distribution:")
        print(df['link_quality'].value_counts().sort_index())
        print(f"  0 (Good): {(df['link_quality']==0).sum()} ({100*(df['link_quality']==0).mean():.1f}%)")
        print(f"  1 (Degraded): {(df['link_quality']==1).sum()} ({100*(df['link_quality']==1).mean():.1f}%)")
        print(f"  2 (Outage): {(df['link_quality']==2).sum()} ({100*(df['link_quality']==2).mean():.1f}%)")
        
    def create_sequences(self, sequence_length=24):
        """Create sliding window sequences"""
        print(f"\nCreating sequences (window={sequence_length})...")
        
        sequences = []
        labels = []
        
        # Group by vessel
        for mmsi in self.feature_df['mmsi'].unique():
            vessel_data = self.feature_df[self.feature_df['mmsi'] == mmsi].sort_values('timestamp')
            
            if len(vessel_data) < sequence_length + 1:
                continue
            
            # Feature columns
            feature_cols = [
                'latitude', 'longitude', 'speed_knots', 'heading',
                'temperature_c', 'humidity_percent', 'pressure_hpa',
                'wind_speed_ms', 'rain_rate_mm_hr',
                'hour_sin', 'hour_cos', 'day_of_year',
                'sat_elevation', 'rain_atten_db', 'atmospheric_loss_db'
            ]
            
            features = vessel_data[feature_cols].values
            targets = vessel_data['link_quality'].values
            
            # Sliding window
            for i in range(len(features) - sequence_length):
                seq = features[i:i+sequence_length]
                label = targets[i+sequence_length]
                
                sequences.append(seq)
                labels.append(label)
        
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        print(f"Total sequences: {len(sequences):,}")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return sequences, labels
    
    def split_data(self, sequences, labels, train_frac=0.7, val_frac=0.15):
        """Train/val/test split"""
        print("\nSplitting data...")
        
        n = len(sequences)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        splits = {
            'train': (sequences[train_idx], labels[train_idx]),
            'val': (sequences[val_idx], labels[val_idx]),
            'test': (sequences[test_idx], labels[test_idx])
        }
        
        for split_name, (X, y) in splits.items():
            print(f"{split_name}: {len(X):,} samples")
            print(f"  Class dist: {np.bincount(y)}")
        
        return splits
    
    def save_processed_data(self, splits):
        """Save to disk"""
        print("\nSaving processed data...")
        
        np.save(self.output_dir / 'train_X.npy', splits['train'][0])
        np.save(self.output_dir / 'train_y.npy', splits['train'][1])
        np.save(self.output_dir / 'val_X.npy', splits['val'][0])
        np.save(self.output_dir / 'val_y.npy', splits['val'][1])
        np.save(self.output_dir / 'test_X.npy', splits['test'][0])
        np.save(self.output_dir / 'test_y.npy', splits['test'][1])
        
        # Metadata
        metadata = {
            'sequence_length': int(splits['train'][0].shape[1]),
            'num_features': int(splits['train'][0].shape[2]),
            'num_classes': 3,
            'train_samples': int(len(splits['train'][0])),
            'val_samples': int(len(splits['val'][0])),
            'test_samples': int(len(splits['test'][0])),
            'feature_names': [
                'latitude', 'longitude', 'speed_knots', 'heading',
                'temperature_c', 'humidity_percent', 'pressure_hpa',
                'wind_speed_ms', 'rain_rate_mm_hr',
                'hour_sin', 'hour_cos', 'day_of_year',
                'sat_elevation', 'rain_atten_db', 'atmospheric_loss_db'
            ],
            'class_names': ['Good', 'Degraded', 'Outage']
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Data saved to {self.output_dir}")
        
    def run_pipeline(self):
        """Execute full pipeline"""
        self.load_data()
        self.match_weather_to_vessels()
        self.create_features()
        sequences, labels = self.create_sequences(sequence_length=24)
        splits = self.split_data(sequences, labels)
        self.save_processed_data(splits)
        print("\nPreprocessing complete!")

if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        ais_path='data/raw/ais/ais_data.csv',
        weather_path='data/raw/weather/weather_data.csv'
    )
    preprocessor.run_pipeline()
