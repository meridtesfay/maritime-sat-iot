"""
Weather Data Downloader (ERA5)

Downloads weather reanalysis data from Copernicus Climate Data Store.
For demo purposes, generates realistic synthetic weather data.

Author: Merid Tesfay
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

class WeatherDataDownloader:
    """
    Download and process ERA5 weather reanalysis data.
    
    In production, this would use cdsapi to fetch real ERA5 data.
    For demo, we generate realistic weather patterns.
    """
    
    def __init__(self, output_dir='data/raw/weather'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_weather_grid(self, lat_range=(0, 60), lon_range=(-80, 140), 
                             resolution=1.0, days=30):
        """
        Generate gridded weather data.
        
        Args:
            lat_range: (min_lat, max_lat)
            lon_range: (min_lon, max_lon)
            resolution: Grid spacing in degrees
            days: Number of days
            
        Returns:
            DataFrame with weather grid points
        """
        print(f"Generating weather grid data...")
        print(f"Resolution: {resolution}° (~{resolution*111:.0f} km)")
        
        # Create spatial grid
        lats = np.arange(lat_range[0], lat_range[1], resolution)
        lons = np.arange(lon_range[0], lon_range[1], resolution)
        
        # Time points (every 3 hours)
        hours = days * 24
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=h) 
                     for h in range(0, hours, 3)]
        
        records = []
        
        for timestamp in timestamps:
            # Day of year for seasonal variation
            doy = timestamp.timetuple().tm_yday
            hour = timestamp.hour
            
            for lat in lats:
                for lon in lons:
                    # Generate weather parameters
                    weather = self._generate_weather_point(lat, lon, doy, hour)
                    weather['latitude'] = lat
                    weather['longitude'] = lon
                    weather['timestamp'] = timestamp
                    records.append(weather)
            
            if len(records) % 10000 == 0:
                print(f"Generated {len(records):,} weather records...")
        
        df = pd.DataFrame(records)
        
        print(f"\nTotal weather records: {len(df):,}")
        print(f"Grid points: {len(lats)} x {len(lons)} = {len(lats)*len(lons):,}")
        print(f"Time points: {len(timestamps)}")
        
        return df
    
    def _generate_weather_point(self, lat, lon, day_of_year, hour):
        """Generate realistic weather parameters for a grid point."""
        
        # Temperature (°C) - seasonal and diurnal variation
        base_temp = 15 + 15 * np.cos(2 * np.pi * (day_of_year - 172) / 365)  # Seasonal
        base_temp -= abs(lat) * 0.5  # Latitude effect
        diurnal = 5 * np.sin(2 * np.pi * (hour - 6) / 24)  # Daily cycle
        temp = base_temp + diurnal + np.random.randn() * 2
        
        # Humidity (%) - anticorrelated with temperature
        humidity = 70 - 0.5 * temp + np.random.randn() * 10
        humidity = np.clip(humidity, 20, 100)
        
        # Pressure (hPa)
        pressure = 1013 + np.random.randn() * 10
        
        # Wind speed (m/s)
        # Higher winds at mid-latitudes and over oceans
        base_wind = 5 + abs(lat - 30) * 0.1
        wind_speed = base_wind + abs(np.random.randn() * 3)
        
        # Wind direction (degrees)
        wind_direction = np.random.uniform(0, 360)
        
        # Precipitation/Rain rate (mm/hr)
        # Use Markov chain: 85% no rain, 12% light, 3% heavy
        rain_state = np.random.choice([0, 1, 2], p=[0.85, 0.12, 0.03])
        if rain_state == 0:
            rain_rate = 0.0
        elif rain_state == 1:
            rain_rate = np.random.uniform(0.5, 5.0)
        else:
            rain_rate = np.random.uniform(5.0, 25.0)
        
        # Cloud cover (fraction 0-1)
        if rain_rate > 0:
            cloud_cover = np.random.uniform(0.7, 1.0)
        else:
            cloud_cover = np.random.uniform(0.0, 0.5)
        
        return {
            'temperature_c': temp,
            'humidity_percent': humidity,
            'pressure_hpa': pressure,
            'wind_speed_ms': wind_speed,
            'wind_direction_deg': wind_direction,
            'rain_rate_mm_hr': rain_rate,
            'cloud_cover': cloud_cover
        }
    
    def save_data(self, df, filename='weather_data.csv'):
        """Save weather data."""
        filepath = self.output_dir / filename
        
        # Save as CSV (could use NetCDF for real ERA5)
        df.to_csv(filepath, index=False)
        print(f"\nWeather data saved to {filepath}")
        print(f"File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Metadata
        metadata = {
            'records': len(df),
            'lat_range': [float(df['latitude'].min()), float(df['latitude'].max())],
            'lon_range': [float(df['longitude'].min()), float(df['longitude'].max())],
            'time_range': [str(df['timestamp'].min()), str(df['timestamp'].max())],
            'variables': list(df.columns),
            'source': 'Synthetic weather data (ERA5-like patterns)',
            'generated': str(datetime.now())
        }
        
        meta_path = self.output_dir / 'weather_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath

def main():
    """Main execution."""
    downloader = WeatherDataDownloader()
    
    # Generate weather grid
    df = downloader.generate_weather_grid(
        lat_range=(0, 60),
        lon_range=(-80, 140),
        resolution=2.0,  # 2 degrees (~220 km)
        days=30
    )
    
    # Save
    downloader.save_data(df)
    
    # Summary
    print("\n" + "="*60)
    print("WEATHER DATA SUMMARY")
    print("="*60)
    print(df.head(10))
    print(f"\nStatistics:")
    print(df.describe())
    
    print(f"\nRain events: {(df['rain_rate_mm_hr'] > 0).sum():,} "
          f"({100*(df['rain_rate_mm_hr'] > 0).mean():.1f}%)")

if __name__ == "__main__":
    main()
