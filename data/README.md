# Data Generation

**Note:** Raw data files are **not included** in the repository due to GitHub's 100 MB file size limit.

## Generate Data Locally

Run these scripts to create the datasets:

### 1. Generate AIS Vessel Data
```bash
python data/download_ais.py
```
**Output:** `data/raw/ais/ais_data.csv` (~2 MB, 18,000 records)

### 2. Generate Weather Data
```bash
python data/download_weather.py
```
**Output:** `data/raw/weather/weather_data.csv` (~112 MB, 792,000 grid points)

### 3. Process into ML-Ready Format
```bash
python data/generate_processed_data.py
```
**Output:** 
- `data/processed/train_X.npy` (12,583 sequences)
- `data/processed/train_y.npy`
- `data/processed/val_X.npy` (2,696 sequences)
- `data/processed/val_y.npy`
- `data/processed/test_X.npy` (2,697 sequences)
- `data/processed/test_y.npy`
- `data/processed/metadata.json`

**Total time:** ~5 minutes

## Data Specifications

### AIS Vessel Data
- **Records:** 18,000 vessel positions
- **Vessels:** 100 ships
- **Duration:** 30 days
- **Routes:** North Atlantic, Mediterranean, Pacific, Indian Ocean
- **Features:** Position, speed, heading, timestamp

### Weather Data
- **Resolution:** 2° spatial, 3-hour temporal
- **Coverage:** Global maritime routes
- **Duration:** 30 days
- **Variables:** Temperature, humidity, pressure, wind, rain

### Processed Data
- **Sequences:** 17,976 total (24-hour windows)
- **Features:** 15 per timestep
- **Classes:** 3 (Good: 85%, Degraded: 12%, Outage: 3%)
- **Splits:** 70% train, 15% val, 15% test

## Why Not Included?

GitHub has a 100 MB file size limit. The weather CSV is 112 MB. Since the data is synthetic and regenerates identically, users can generate it locally in ~5 minutes.

## Verify Data

After generation, verify with:
```bash
python test_setup.py
```

Should output:
```
✓ Data loaded: train=12583, val=2696, test=2697
✓ Class distribution: [85% / 12% / 3%]
```
