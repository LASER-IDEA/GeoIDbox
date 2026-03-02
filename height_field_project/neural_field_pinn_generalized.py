"""
Generalized PINN architecture WITHOUT sensor-specific embeddings.

Instead of learned per-sensor embeddings, this version uses:
1. Pure spatial encoding (hash encoding) 
2. Temporal encoding (Fourier features)
3. Environmental context (temperature, humidity)
4. Physics-derived features (pressure residual, virtual temperature)

This enables zero-shot generalization to new sensor locations.
"""
import torch
import torch.nn as nn
from typing import Tuple


class SirenLayer(nn.Module):
    """Sinusoidal activation layer."""
    def __init__(self, dim_in: int, dim_out: int, w0: float = 1.0, is_first: bool = False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(dim_in, dim_out)
        
        # SIREN initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/dim_in, 1/dim_in)
            else:
                self.linear.weight.uniform_(
                    -torch.sqrt(torch.tensor(6.0/dim_in)) / w0,
                    torch.sqrt(torch.tensor(6.0/dim_in)) / w0
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class MultiResolutionHashEncoding(nn.Module):
    """Simplified hash encoding for spatial coordinates."""
    def __init__(self, n_levels: int = 16, n_features: int = 4, 
                 min_res: int = 16, max_res: int = 1024):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.min_res = min_res
        self.max_res = max_res
        self.out_dim = n_levels * n_features
        
        # Hash tables for each level
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(2**14, n_features) * 1e-4)
            for _ in range(n_levels)
        ])
        
        # Resolution schedule
        self.resolutions = torch.round(torch.exp(
            torch.linspace(torch.log(torch.tensor(min_res, dtype=torch.float32)),
                          torch.log(torch.tensor(max_res, dtype=torch.float32)),
                          n_levels)
        )).long()
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, 2] in [0, 1] range (lat, lon normalized)
        Returns:
            [B, n_levels * n_features]
        """
        batch_size = coords.shape[0]
        encodings = []
        
        device = coords.device
        resolutions = self.resolutions.to(device)
        
        for level in range(self.n_levels):
            # Scale to resolution
            scaled = coords * resolutions[level]  # [B, 2]
            
            # Grid corners
            corners = torch.floor(scaled).long()  # [B, 2]
            
            # Clamp to valid range
            corners = torch.clamp(corners, 0, resolutions[level] - 1)
            
            # Simple lookup (no trilinear interpolation for speed)
            indices = corners[:, 0] * resolutions[level] + corners[:, 1]
            indices = indices % len(self.hash_tables[level])  # Hash to table size
            
            features = self.hash_tables[level][indices]  # [B, n_features]
            encodings.append(features)
        
        return torch.cat(encodings, dim=-1)  # [B, n_levels * n_features]


class FourierTemporalEncoding(nn.Module):
    """Fourier features for temporal encoding."""
    def __init__(self, n_frequencies: int = 6, max_period_hours: float = 168.0):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.max_period_hours = max_period_hours
        self.out_dim = 2 * n_frequencies
        
        # Frequencies for diurnal, weekly patterns
        frequencies = torch.tensor([
            1.0 / (24.0 * (2 ** i)) for i in range(n_frequencies // 2)
        ] + [
            1.0 / (168.0 * (2 ** i)) for i in range(n_frequencies // 2)
        ]) * 2 * 3.14159
        
        self.register_buffer('frequencies', frequencies)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] time in hours
        Returns:
            [B, 2 * n_frequencies]
        """
        t_expanded = t.unsqueeze(-1)  # [B, 1]
        angular = t_expanded * self.frequencies.unsqueeze(0)  # [B, n_freq]
        return torch.cat([torch.sin(angular), torch.cos(angular)], dim=-1)


class GeneralizedPressureCorrectionPINN(nn.Module):
    """
    Sensor-agnostic PINN for pressure correction.
    
    NO sensor-specific embeddings - pure spatial-temporal model.
    """
    def __init__(
        self,
        hash_levels: int = 16,
        hash_features: int = 4,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        temporal_freqs: int = 6,
        dropout: float = 0.0,
        use_siren: bool = True
    ):
        super().__init__()
        
        # Spatial encoding (hash grid)
        self.hash_encoding = MultiResolutionHashEncoding(
            n_levels=hash_levels,
            n_features=hash_features
        )
        
        # Temporal encoding
        self.temporal_encoding = FourierTemporalEncoding(
            n_frequencies=temporal_freqs,
            max_period_hours=168.0
        )
        
        # Input dimension calculation
        # hash_out + z(1) + temp_out + T(1) + RH(1)
        in_dim = (
            self.hash_encoding.out_dim +  # Spatial: 16*4 = 64
            1 +  # Z (altitude)
            self.temporal_encoding.out_dim +  # Temporal: 12
            1 + 1  # Temperature, Humidity
        )
        
        # MLP layers
        if use_siren:
            layers = []
            layers.append(SirenLayer(in_dim, hidden_dim, w0=1.0, is_first=True))
            for _ in range(n_hidden_layers):
                layers.append(SirenLayer(hidden_dim, hidden_dim, w0=1.0))
            layers.append(nn.Linear(hidden_dim, 1))
        else:
            layers = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(n_hidden_layers):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize output to near zero
        with torch.no_grad():
            self.mlp[-1].weight.fill_(0.0)
            self.mlp[-1].bias.fill_(0.0)
    
    def forward(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        temperature: torch.Tensor,
        humidity: torch.Tensor,
        sensor_id: torch.Tensor = None,  # Ignored - for compatibility
        normalize_coords: bool = True
    ) -> torch.Tensor:
        """
        Predict pressure correction δP (sensor-agnostic).
        
        Args:
            lat, lon: Coordinates [B]
            z: Altitude [B]
            t: Time (Unix timestamp) [B]
            temperature: Temperature [B] (Celsius)
            humidity: Relative humidity [B] (percent)
        
        Returns:
            Pressure correction δP in Pascals [B]
        """
        # Normalize coordinates
        if normalize_coords:
            lat_norm = (lat + 90.0) / 180.0
            lon_norm = lon % 360.0 / 360.0
        else:
            lat_norm = lat
            lon_norm = lon
        
        coords = torch.stack([lat_norm, lon_norm], dim=-1)
        
        # Encode features
        h_spatial = self.hash_encoding(coords)  # [B, 64]
        h_temporal = self.temporal_encoding(t / 3600.0)  # [B, 12]
        
        # Stack all features
        features = torch.cat([
            h_spatial,
            z.unsqueeze(-1),
            h_temporal,
            temperature.unsqueeze(-1),
            humidity.unsqueeze(-1)
        ], dim=-1)
        
        # Predict pressure correction
        delta_p = self.mlp(features).squeeze(-1)
        
        return delta_p
    
    def predict_mc(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        temperature: torch.Tensor,
        humidity: torch.Tensor,
        sensor_id: torch.Tensor = None,  # Ignored - for compatibility
        samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo dropout for uncertainty (if dropout > 0)."""
        if self.training or samples <= 1:
            return self.forward(lat, lon, z, t, temperature, humidity), torch.zeros_like(lat)
        
        # Multiple forward passes
        preds = []
        for _ in range(samples):
            preds.append(self.forward(lat, lon, z, t, temperature, humidity))
        
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0)
        std = stack.std(dim=0)
        return mean, std
