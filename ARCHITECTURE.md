# Maritime Satellite IoT Link Prediction - System Architecture

**Author:** Merid Tesfay  
**Project:** ML for Satellite Communication Link Quality Prediction  
**Purpose:** Aalborg University AI:GeoComm Lab PhD Application

---

## Executive Summary

This project implements a multi-modal deep learning system for predicting satellite IoT link quality for maritime vessels. The system addresses key research challenges in satellite communications:

1. **Spatiotemporal dependencies** in vessel movement and weather evolution
2. **Extreme class imbalance** in link quality states (similar to rare disease detection)
3. **Multimodal data fusion** across spatial, temporal, and environmental domains
4. **Geographic distribution shift** across different ocean regions

**Key Innovation:** Physics-informed neural network architecture that combines domain knowledge (satellite geometry, atmospheric propagation) with data-driven learning.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MARITIME SAT-IoT LINK PREDICTION SYSTEM                   │
│                                                                               │
│  Input: Vessel Position + Weather + Satellite Geometry → Output: Link Quality│
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────── DATA ACQUISITION LAYER ────────────────────────────┐
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐     │
│  │   AIS Data   │      │ Weather Data │      │  Satellite Orbits    │     │
│  │              │      │              │      │                      │     │
│  │ • Position   │      │ • Temperature│      │ • TLE Data (Iridium) │     │
│  │ • Speed      │      │ • Humidity   │      │ • Coverage Maps      │     │
│  │ • Heading    │      │ • Pressure   │      │ • Elevation Angles   │     │
│  │ • Timestamp  │      │ • Wind Speed │      │ • Doppler Shift      │     │
│  │              │      │ • Rain Rate  │      │                      │     │
│  └──────────────┘      └──────────────┘      └──────────────────────┘     │
│       │                      │                         │                    │
│       │                      │                         │                    │
└───────┼──────────────────────┼─────────────────────────┼────────────────────┘
        │                      │                         │
        └──────────────────────┴─────────────────────────┘
                               │
                               ▼
┌──────────────────────── DATA PREPROCESSING ─────────────────────────────────┐
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │               MULTIMODAL DATA FUSION PIPELINE                   │        │
│  │                                                                  │        │
│  │  1. Temporal Alignment (synchronize to common timestamps)       │        │
│  │  2. Spatial Interpolation (grid-based weather matching)         │        │
│  │  3. Feature Engineering:                                         │        │
│  │     - Satellite elevation angle                                 │        │
│  │     - Rain attenuation (ITU-R model)                            │        │
│  │     - Vessel velocity vector                                    │        │
│  │     - Sea state (derived from wind)                             │        │
│  │     - Ionospheric TEC (time-of-day proxy)                       │        │
│  │  4. Sequence Creation (24-hour sliding windows)                 │        │
│  │  5. Normalization & Scaling                                     │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                               │                                              │
└───────────────────────────────┼──────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────── MODEL ARCHITECTURE LAYER ────────────────────────────┐
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                    INPUT FEATURE TENSOR                       │          │
│  │         Shape: [Batch, Sequence_Length=24, Features=15]       │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                               │                                              │
│                ┌──────────────┴──────────────┐                              │
│                ▼                             ▼                              │
│  ┌─────────────────────────┐   ┌─────────────────────────┐                │
│  │   SPATIAL BRANCH (CNN)  │   │ TEMPORAL BRANCH (LSTM)  │                │
│  │                         │   │                         │                │
│  │  Conv1D(15→64, k=3)    │   │  BiLSTM(15→128)        │                │
│  │  BatchNorm + ReLU       │   │  Dropout(0.3)          │                │
│  │  Conv1D(64→128, k=3)   │   │  BiLSTM(128→128)       │                │
│  │  MaxPool(k=2)          │   │  Dropout(0.3)          │                │
│  │  Conv1D(128→256, k=3)  │   │                         │                │
│  │  GlobalAvgPool         │   │  Output: [B, 24, 256]  │                │
│  │                         │   │                         │                │
│  │  Output: [B, 256]      │   └─────────────────────────┘                │
│  └─────────────────────────┘                │                              │
│                │                             │                              │
│                └─────────────┬───────────────┘                              │
│                              ▼                                              │
│               ┌──────────────────────────────┐                             │
│               │ MULTI-HEAD ATTENTION FUSION  │                             │
│               │                              │                             │
│               │  Q = Linear(spatial_feat)   │                             │
│               │  K = Linear(temporal_feat)  │                             │
│               │  V = Linear(temporal_feat)  │                             │
│               │                              │                             │
│               │  Attention = Softmax(QK^T/√d)│                             │
│               │  Output = Attention · V      │                             │
│               └──────────────────────────────┘                             │
│                              │                                              │
│                              ▼                                              │
│               ┌──────────────────────────────┐                             │
│               │   PHYSICS-INFORMED LAYER     │                             │
│               │                              │                             │
│               │  • Satellite elevation       │                             │
│               │  • Path loss calculation     │                             │
│               │  • Rain attenuation          │                             │
│               │  (Embedded as additional     │                             │
│               │   features with constraints) │                             │
│               └──────────────────────────────┘                             │
│                              │                                              │
│                              ▼                                              │
│               ┌──────────────────────────────┐                             │
│               │   PREDICTION HEAD            │                             │
│               │                              │                             │
│               │  FC(512 → 256) + ReLU       │                             │
│               │  Dropout(0.3)               │                             │
│               │  FC(256 → 128) + ReLU       │                             │
│               │  FC(128 → 3)                │                             │
│               │  Softmax (3 classes)        │                             │
│               │                              │                             │
│               │  Classes:                    │                             │
│               │  0: Good Link (SNR > 10 dB) │                             │
│               │  1: Degraded (5-10 dB)      │                             │
│               │  2: Outage (< 5 dB)         │                             │
│               └──────────────────────────────┘                             │
│                              │                                              │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────── LOSS FUNCTION ─────────────────────────────────────┐
│                                                                              │
│  HYBRID LOSS = α·Focal_Loss + β·Physics_Consistency_Loss                   │
│                                                                              │
│  1. Focal Loss (addresses class imbalance):                                 │
│     FL(p_t) = -α_t(1-p_t)^γ log(p_t)                                       │
│     • Focuses on hard examples                                              │
│     • Down-weights easy negatives                                           │
│     • γ=2.0, α=[0.25, 0.50, 0.75] for [Good, Degraded, Outage]           │
│                                                                              │
│  2. Physics Consistency Loss:                                               │
│     L_physics = ||predicted_SNR - physics_model_SNR||²                     │
│     • Ensures predictions respect physical bounds                           │
│     • Regularizes model with domain knowledge                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

```

---

## Data Flow Pipeline

```
┌─────────────┐
│ Raw AIS     │  Marine Cadastre API
│ Vessel Data │  • 1M+ vessel positions/day
└──────┬──────┘  • Global coverage
       │
       ▼
┌─────────────────────┐
│ Data Filtering      │
│ • Maritime routes   │
│ • Valid positions   │
│ • Timestamp range   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐      ┌──────────────┐
│ Weather Data Fetch  │◄─────┤ ERA5 Climate │
│ • Interpolate to    │      │ Reanalysis   │
│   vessel position   │      └──────────────┘
│ • Extract:          │
│   - Temperature     │
│   - Humidity        │      ┌──────────────┐
│   - Pressure        │◄─────┤ CelesTrak    │
│   - Wind            │      │ TLE Data     │
│   - Precipitation   │      └──────────────┘
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Feature Engineering │
│                     │
│ Spatial Features:   │
│ • Latitude          │
│ • Longitude         │
│ • Distance to coast │
│ • Sea depth         │
│                     │
│ Temporal Features:  │
│ • Hour of day       │
│ • Day of year       │
│ • Vessel speed      │
│ • Heading           │
│                     │
│ Satellite Features: │
│ • Elevation angle   │
│ • Azimuth           │
│ • Doppler shift     │
│ • Path loss         │
│                     │
│ Derived Features:   │
│ • Rain attenuation  │
│ • Sea state         │
│ • Link margin       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Sequence Creation   │
│ • 24-hour windows   │
│ • 1-hour stride     │
│ • Train/Val/Test    │
│   (70/15/15)        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Data Augmentation   │
│ (for minority class)│
│                     │
│ • SMOTE-NC          │
│ • Temporal jitter   │
│ • Gaussian noise    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Batch Loading       │
│ PyTorch DataLoader  │
└─────────────────────┘
```

---

## Model Comparison Framework

We implement 7 different architectures to demonstrate comprehensive ML knowledge:

| Model | Architecture | Purpose | Key Innovation |
|-------|--------------|---------|----------------|
| **1. Physics Baseline** | ITU-R + Friis | Establish lower bound | Domain knowledge only |
| **2. CNN** | 1D ResNet | Spatial patterns | Geographic link quality maps |
| **3. BiLSTM** | 2-layer BiLSTM | Temporal sequences | Movement trajectory learning |
| **4. Transformer** | Multi-head attention | Long-range dependencies | Self-attention over time/space |
| **5. CNN-LSTM Hybrid** | CNN→LSTM pipeline | Spatiotemporal fusion | Best of both worlds |
| **6. Attention Fusion** | CNN+LSTM+Attention | Multimodal integration | Learned fusion weights |
| **7. Physics-Informed NN** | Hybrid+Physics loss | Constrained learning | Respects physical laws |

---

## Research Challenges Addressed

### 1. **Class Imbalance Problem**

**Challenge:** Link quality distribution
- Good: 85% (majority)
- Degraded: 12%
- Outage: 3% (rare but critical!)

**Solutions Implemented:**
```python
# 1. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=[0.25, 0.50, 0.75]):
        # Down-weight easy examples
        # Focus learning on hard/rare cases
        
# 2. Class Weighting
class_weights = compute_class_weight('balanced', classes, y_train)

# 3. SMOTE Oversampling
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. Stratified Sampling
train_loader = DataLoader(dataset, sampler=StratifiedSampler(...))
```

**Analogous to:** Medical imaging rare disease detection (your thesis!)

### 2. **Spatiotemporal Dependencies**

**Challenge:** Link quality depends on:
- Where vessel is (spatial)
- Where vessel was (temporal)
- Where weather system is moving (spatiotemporal)

**Solutions:**
```python
# Hybrid CNN-LSTM captures both
spatial_features = CNN_branch(input)    # Geographic patterns
temporal_features = LSTM_branch(input)  # Movement trajectories
fused = Attention(spatial_features, temporal_features)
```

**Analogous to:** Medical 3D+time sequences (4D CT scans)

### 3. **Multimodal Fusion**

**Challenge:** Integrate heterogeneous data sources
- Discrete: Vessel positions
- Continuous: Weather fields
- Geometric: Satellite orbits

**Solution:**
```python
class MultimodalFusion(nn.Module):
    def __init__(self):
        self.spatial_encoder = CNN()
        self.temporal_encoder = LSTM()
        self.attention = MultiHeadAttention(heads=4)
        
    def forward(self, spatial, temporal):
        # Learn which modality matters when
        weights = self.attention(spatial, temporal)
        return weighted_fusion(spatial, temporal, weights)
```

**Analogous to:** Medical imaging + clinical data fusion

### 4. **Physics-Informed Learning**

**Challenge:** Predictions must respect physical constraints
- SNR cannot be negative
- Path loss increases with distance
- Rain attenuation follows ITU-R model

**Solution:**
```python
def physics_consistency_loss(predicted_snr, elevation, rain_rate):
    # Calculate physics-based SNR
    physics_snr = friis_equation(elevation) - itu_rain_atten(rain_rate)
    
    # Penalize violations
    consistency_loss = mse_loss(predicted_snr, physics_snr)
    
    # Total loss
    return focal_loss + λ * consistency_loss
```

**Novel Contribution:** Embedding domain knowledge as soft constraints

---

## Training Strategy (HPC-Optimized)

### SLURM Configuration
```bash
#!/bin/bash
#SBATCH --job-name=maritime_sat_iot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Distributed training ready
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    train.py --config configs/hybrid_model.yaml
```

### Training Pipeline
1. **Stage 1:** Train individual models (CNN, LSTM, Transformer)
2. **Stage 2:** Train ensemble/hybrid models
3. **Stage 3:** Fine-tune with physics-informed loss
4. **Stage 4:** Domain adaptation (Arctic vs Tropical)

### Hyperparameter Optimization
- Learning rate: [1e-5, 1e-3] (log scale)
- Batch size: [32, 64, 128]
- Sequence length: [12, 24, 48] hours
- Dropout: [0.1, 0.3, 0.5]
- Focal loss γ: [1.0, 2.0, 3.0]

---

## Evaluation Metrics

### Classification Metrics
```python
# Standard
- Accuracy (overall)
- Precision, Recall, F1 (per class)
- Confusion Matrix

# Class-Imbalance Aware
- Balanced Accuracy
- Macro F1 (average across classes)
- Matthews Correlation Coefficient

# Probabilistic
- ROC-AUC (one-vs-rest)
- Precision-Recall AUC (for minority classes)
- Calibration Curve (reliability)
```

### Domain-Specific Metrics
```python
# Maritime Safety
- Critical Outage Detection Rate (class 2)
- False Alarm Rate (predict outage when good)
- Early Warning Time (how far ahead we predict)

# Operational
- Link Availability Prediction Accuracy
- Average Prediction Error (SNR in dB)
- Geographic Performance (Arctic vs Equator)
```

### Ablation Studies
- Remove weather features → measure importance
- Remove temporal context → validate LSTM contribution
- Disable physics loss → show constraint benefit

---

## Deployment Architecture

```
┌──────────────────────────────────────────────┐
│            INFERENCE API (FastAPI)           │
│                                              │
│  POST /predict                               │
│  Input: {vessel_id, timestamp, position}     │
│  Output: {link_quality, confidence, snr_est} │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│         MODEL SERVING (TorchServe)           │
│                                              │
│  • Load trained ensemble                     │
│  • Batch inference                           │
│  • GPU acceleration                          │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│      REAL-TIME DATA INGESTION               │
│                                              │
│  • AIS stream (via AIS Hub API)             │
│  • Weather forecast (ERA5 or NOAA)          │
│  • Satellite TLE updates (CelesTrak)        │
└──────────────────────────────────────────────┘
```

---

## Connection to AI:GeoComm Research

### Stipend 1: Atmospheric AI Modeling
- ✅ Weather data fusion (ERA5 integration)
- ✅ Rain attenuation modeling (ITU-R + ML)
- ✅ Temporal forecasting (LSTM/Transformer)

### Stipend 2: Satellite Communications
- ✅ Link quality prediction (core objective)
- ✅ Multi-satellite systems (Iridium constellation)
- ✅ Adaptive protocols potential (based on predictions)

### Novel Research Directions
1. **Bidirectional Learning:** Use satellite signals to infer weather (inverse problem)
2. **Network Optimization:** Route traffic based on predicted link quality
3. **Federated Learning:** Train across multiple vessel fleets
4. **Digital Twin:** Real-time satellite network simulator

---

## Project Timeline

**Phase 1:** Data Acquisition (Day 1)
- Download AIS data
- Fetch ERA5 weather
- Process satellite TLEs

**Phase 2:** Preprocessing (Day 1-2)
- Multimodal alignment
- Feature engineering
- Train/val/test split

**Phase 3:** Model Development (Day 2-3)
- Implement all 7 architectures
- SLURM training scripts
- Hyperparameter tuning

**Phase 4:** Evaluation (Day 3)
- Comprehensive metrics
- Ablation studies
- Visualization

**Phase 5:** Documentation (Day 3-4)
- Architecture diagrams
- Results dashboard
- GitHub README

---

**Next Steps:** Proceed to implementation?
