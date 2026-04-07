"""
AIS Vessel Data Downloader

Downloads Automatic Identification System (AIS) data for maritime vessels.
Uses public datasets from Marine Cadastre and other sources.

Author: Merid Tesfay
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from pathlib import Path
import json

class AISDataDownloader:
    """
    Download and process AIS vessel tracking data.
    
    For this demo, we generate realistic synthetic AIS data based on
    actual maritime traffic patterns. In production, replace with API calls
    to Marine Cadastre, AIS Hub, or similar services.
    """
    
    def __init__(self, output_dir='data/raw/ais'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_realistic_ais_data(self, num_vessels=100, days=30):
        """
        Generate realistic AIS data based on actual maritime patterns.
        
        Args:
            num_vessels: Number of vessels to simulate
            days: Number of days of data
            
        Returns:
            DataFrame with AIS records
        """
        print(f"Generating AIS data for {num_vessels} vessels over {days} days...")
        
        # Define major shipping routes
        routes = {
            'north_atlantic': {
                'start': (40.0, -70.0),  # New York area
                'end': (51.5, -0.1),      # English Channel
                'vessel_count': 30
            },
            'mediterranean': {
                'start': (36.0, -5.0),    # Gibraltar
                'end': (31.2, 34.8),      # Suez
                'vessel_count': 25
            },
            'pacific': {
                'start': (35.7, 139.7),   # Tokyo
                'end': (37.8, -122.4),    # San Francisco
                'vessel_count': 25
            },
            'indian_ocean': {
                'start': (1.3, 103.8),    # Singapore
                'end': (21.3, 39.2),      # Jeddah
                'vessel_count': 20
            }
        }
        
        all_records = []
        vessel_id = 1
        
        for route_name, route_info in routes.items():
            for v in range(route_info['vessel_count']):
                if vessel_id > num_vessels:
                    break
                    
                # Generate vessel trajectory
                records = self._generate_vessel_trajectory(
                    vessel_id=vessel_id,
                    start_pos=route_info['start'],
                    end_pos=route_info['end'],
                    days=days
                )
                all_records.extend(records)
                vessel_id += 1
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        print(f"Generated {len(df):,} AIS records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Vessels: {df['mmsi'].nunique()}")
        
        return df
    
    def _generate_vessel_trajectory(self, vessel_id, start_pos, end_pos, days):
        """Generate realistic vessel trajectory between two points."""
        
        # Vessel parameters
        avg_speed_knots = np.random.uniform(12, 18)  # Typical cargo ship
        
        # Calculate route
        start_lat, start_lon = start_pos
        end_lat, end_lon = end_pos
        
        # Estimate number of waypoints (one per 4 hours)
        hours = days * 24
        waypoints = hours // 4
        
        # Generate trajectory with some randomness (currents, weather avoidance)
        lat_points = np.linspace(start_lat, end_lat, waypoints)
        lon_points = np.linspace(start_lon, end_lon, waypoints)
        
        # Add random variation (up to 0.5 degrees)
        lat_points += np.random.randn(waypoints) * 0.5
        lon_points += np.random.randn(waypoints) * 0.5
        
        # Generate records
        records = []
        start_time = datetime(2024, 1, 1)
        
        for i, (lat, lon) in enumerate(zip(lat_points, lon_points)):
            timestamp = start_time + timedelta(hours=i*4)
            
            # Calculate heading to next waypoint
            if i < len(lat_points) - 1:
                dlat = lat_points[i+1] - lat
                dlon = lon_points[i+1] - lon
                heading = np.arctan2(dlon, dlat) * 180 / np.pi
                heading = (heading + 360) % 360
            else:
                heading = records[-1]['heading'] if records else 0
            
            # Speed variation
            speed = avg_speed_knots + np.random.randn() * 2
            speed = max(8, min(22, speed))  # Clamp to realistic range
            
            record = {
                'mmsi': f'{vessel_id:09d}',  # Maritime Mobile Service Identity
                'timestamp': timestamp,
                'latitude': lat,
                'longitude': lon,
                'speed_knots': speed,
                'heading': heading,
                'vessel_type': 'cargo',  # Simplified
                'length': np.random.randint(150, 300),  # meters
            }
            records.append(record)
        
        return records
    
    def save_data(self, df, filename='ais_data.csv'):
        """Save AIS data to CSV."""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"\nAIS data saved to {filepath}")
        
        # Save metadata
        metadata = {
            'records': len(df),
            'vessels': int(df['mmsi'].nunique()),
            'start_date': str(df['timestamp'].min()),
            'end_date': str(df['timestamp'].max()),
            'lat_range': [float(df['latitude'].min()), float(df['latitude'].max())],
            'lon_range': [float(df['longitude'].min()), float(df['longitude'].max())],
            'source': 'Synthetic AIS data (realistic patterns)',
            'generated': str(datetime.now())
        }
        
        meta_path = self.output_dir / 'ais_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {meta_path}")
        
        return filepath

def main():
    """Main execution."""
    downloader = AISDataDownloader()
    
    # Generate data
    df = downloader.generate_realistic_ais_data(
        num_vessels=100,
        days=30
    )
    
    # Save
    downloader.save_data(df)
    
    # Display summary
    print("\n" + "="*60)
    print("AIS DATA SUMMARY")
    print("="*60)
    print(df.head(10))
    print(f"\nTotal records: {len(df):,}")
    print(f"Vessels: {df['mmsi'].nunique()}")
    print(f"Avg speed: {df['speed_knots'].mean():.1f} knots")
    print(f"Geographic coverage:")
    print(f"  Latitude: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
    print(f"  Longitude: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")

if __name__ == "__main__":
    main()
