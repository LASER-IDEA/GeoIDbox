# Weather-Inspired Neural Field Architecture

## Problem Diagnosis

The original model fails at LOSO because it tries to learn location-specific corrections without seeing that location during training. But weather is a **shared signal** - 99.9% correlated across sensors.

## Proposed Solution: Three-Component Decomposition

```
h_pred(loc, t) = ISA(pressure) + spatial_bias(loc) + weather_anomaly(t)
```

### Component 1: ISA Physics (Universal)
- Standard atmosphere model with universal constants
- Same for all locations and times

### Component 2: Spatial Bias (Location-Specific)
- Barometer calibration offset
- Local terrain effects
- Estimated via **spatial interpolation** from nearby sensors
- Methods: IDW, Kriging, or learned embedding

### Component 3: Weather Anomaly (Temporal, Shared)
- Synoptic pressure systems
- Temperature inversions
- Learned from **all sensors simultaneously** (weather stations)
- 99.9% correlation means one sensor can predict another's weather

## Implementation Approaches

### Approach A: Explicit Decomposition (Current Demo)
- Separate models for spatial and temporal components
- Spatial: IDW interpolation
- Temporal: Mean anomaly across sensors
- **Result**: 6.4m MAE (validated!)

### Approach B: Neural Weather Field
```python
class WeatherField(nn.Module):
    def __init__(self):
        # Spatial embedding (like terrain embedding)
        self.spatial_embed = SpatialEmbedding(n_sensors, dim=16)
        
        # Temporal weather encoder (shared across locations)
        self.weather_encoder = TemporalEncoder(
            input_dim=features,
            hidden_dim=64,
            uses_era5=True
        )
        
    def forward(self, location_id, time_features, sensor_features):
        # Spatial bias (location-specific)
        spatial_bias = self.spatial_embed(location_id)
        
        # Weather anomaly (shared temporal signal)
        weather_anomaly = self.weather_encoder(time_features)
        
        # Combine
        return spatial_bias + weather_anomaly
```

### Approach C: Graph Neural Network
- Treat sensors as nodes in a graph
- Edges based on spatial proximity
- Message passing shares weather information
- At inference: interpolate to new nodes

## Training Strategy

### Stage 1: Pre-train Weather Encoder
- Use all sensors
- Predict temporal anomaly from ERA5 + sensor features
- Loss: MSE against observed anomalies

### Stage 2: Fine-tune Spatial Embeddings
- LOSO validation
- Freeze weather encoder
- Learn spatial bias per sensor
- Spatial interpolation for test sensors

### Stage 3: End-to-End Fine-tuning
- Unfreeze all
- Small learning rate
- Full LOSO training

## Expected Performance

| Approach | Expected MAE | Requirements |
|----------|--------------|--------------|
| Baseline (ISA) | ~72m | None |
| Current Neural Field | ~53m | LOSO training |
| Explicit Decomposition | ~6m | Spatial interpolation |
| Neural Weather Field | ~3-5m | Pre-training + GNN |
| With Dense Network | <1m | 20+ sensors |

## Why This Works for Production

### Scenario 1: Interpolating Between Sensors
- 7 sensors at known locations
- Want prediction at location 8 (no training data)
- **Solution**: Interpolate spatial bias, use weather from any sensor

### Scenario 2: New Sensor Deployment
- Deploy sensor at new location
- No historical data
- **Solution**: 
  1. Place temporarily near existing sensor
  2. Learn spatial bias difference
  3. Move to target location
  4. Use weather from network

### Scenario 3: Sensor Failure
- One sensor goes offline
- **Solution**: Weather signal from other sensors, spatial bias from last known

## Implementation Priority

1. **Immediate** (done): Explicit decomposition with IDW
   - 6.4m MAE achieved
   - No neural network needed
   - Robust and interpretable

2. **Short-term**: Neural Weather Field
   - Replace IDW with learned spatial embeddings
   - Add uncertainty quantification
   - Target: 3-5m MAE

3. **Long-term**: Dense Network + ERA5 Integration
   - 20+ sensors for meter-level accuracy
   - Full weather model downscaling
   - Real-time correction service

## Reference: Weather Forecasting Techniques to Borrow

1. **Data Assimilation**: Combine sensor observations with ERA5 prior
2. **Ensemble Methods**: Multiple models for uncertainty
3. **Kriging**: Optimal spatial interpolation with uncertainty
4. **Spectral Methods**: Fourier features for periodic weather patterns
5. **Attention Mechanisms**: Learn which sensors to trust for weather
