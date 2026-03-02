"""
Physics-Informed Neural Field (PINN) for barometric pressure correction.

Architecture:
- Multi-resolution hash encoding for spatial coordinates
- SIREN activations for derivative-friendly continuous functions
- Sensor-specific embeddings for calibration biases
- Fourier temporal encoding for weather patterns
- Outputs pressure correction field δP(x, y, z, t)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


class HashEncoding(nn.Module):
    """
    Multi-resolution hash encoding (Instant-NGP style).
    
    Maps spatial coordinates to feature vectors at multiple resolutions
    for efficient high-frequency detail capture.
    """
    
    def __init__(
        self,
        n_levels: int = 16,
        n_features: int = 2,
        log2_hashmap_size: int = 19,  # ~524k entries
        base_resolution: int = 16,
        max_resolution: int = 512,
        device: str = "cuda"
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.device = device
        
        # Compute per-level resolutions (geometric progression)
        b = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1))
        self.resolutions = [int(base_resolution * (b ** i)) for i in range(n_levels)]
        
        # Create hash tables for each level
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, n_features) * 1e-4)
            for _ in range(n_levels)
        ])
        
        self.out_dim = n_levels * n_features
    
    def hash_function(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Spatial hash function for grid indexing.
        
        Args:
            coords: Integer grid coordinates [B, D]
        
        Returns:
            Hash indices [B]
        """
        # Large prime numbers for hashing
        primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.long, device=coords.device)
        
        coords = coords.long()
        hashed = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        
        for i in range(coords.shape[1]):
            hashed ^= coords[:, i] * primes[i]
        
        return hashed % self.hashmap_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hash encoding.
        
        Args:
            x: Spatial coordinates [B, 2] (lat, lon) normalized to [0, 1]
        
        Returns:
            Encoded features [B, n_levels * n_features]
        """
        batch_size = x.shape[0]
        encoded_levels = []
        
        for level in range(self.n_levels):
            resolution = self.resolutions[level]
            hash_table = self.hash_tables[level]
            
            # Scale coordinates to grid resolution
            scaled = x * resolution
            
            # Get grid cell corners (integer coordinates)
            grid_coords = torch.floor(scaled).long()
            
            # Get fractional parts for interpolation
            frac = scaled - grid_coords.float()
            
            # 2D grid interpolation
            # corners: (0,0), (1,0), (0,1), (1,1)
            corners = torch.tensor([
                [0, 0], [1, 0], [0, 1], [1, 1]
            ], device=x.device)
            
            corner_coords = grid_coords.unsqueeze(1) + corners.unsqueeze(0)  # [B, 4, 2]
            corner_coords = corner_coords.reshape(-1, 2)  # [B*4, 2]
            
            # Hash lookup
            corner_indices = self.hash_function(corner_coords)
            corner_features = hash_table[corner_indices]  # [B*4, n_features]
            corner_features = corner_features.reshape(batch_size, 4, self.n_features)
            
            # Bilinear interpolation
            w00 = (1 - frac[:, 0:1]) * (1 - frac[:, 1:2])  # [B, 1]
            w10 = frac[:, 0:1] * (1 - frac[:, 1:2])
            w01 = (1 - frac[:, 0:1]) * frac[:, 1:2]
            w11 = frac[:, 0:1] * frac[:, 1:2]
            
            weights = torch.cat([w00, w10, w01, w11], dim=1)  # [B, 4]
            
            # Weighted sum
            level_features = torch.sum(
                corner_features * weights.unsqueeze(-1),  # [B, 4, n_features]
                dim=1  # [B, n_features]
            )
            
            encoded_levels.append(level_features)
        
        # Concatenate all levels
        return torch.cat(encoded_levels, dim=-1)


class SineActivation(nn.Module):
    """
    SIREN sine activation with learnable frequency (w0).
    """
    
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """SIREN layer with proper weight initialization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float = 30.0,
        is_first: bool = False
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = SineActivation(w0)
        self.is_first = is_first
        self.w0 = w0
        
        # SIREN weight initialization
        with torch.no_grad():
            if is_first:
                # First layer: uniform in [-1/n, 1/n]
                bound = 1.0 / in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/n)/w0, sqrt(6/n)/w0]
                bound = np.sqrt(6.0 / in_features) / w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class FourierTemporalEncoding(nn.Module):
    """
    Fourier feature encoding for temporal information.
    
    Captures periodic patterns:
    - Diurnal cycles (24 hours)
    - Weekly weather patterns (168 hours)
    - Seasonal variations (8760 hours)
    """
    
    def __init__(self, n_frequencies: int = 4, max_period_hours: float = 168.0):
        super().__init__()
        # Log-spaced frequencies
        frequencies = torch.logspace(
            0, np.log10(max_period_hours), n_frequencies
        )
        self.register_buffer("frequencies", 2 * np.pi / frequencies)
        self.out_dim = 2 * n_frequencies
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time in hours (can be Unix timestamp / 3600)
        
        Returns:
            Encoded temporal features [B, 2*n_frequencies]
        """
        t_scaled = t.unsqueeze(-1) * self.frequencies  # [B, n_freq]
        return torch.cat([torch.sin(t_scaled), torch.cos(t_scaled)], dim=-1)


class PressureCorrectionPINN(nn.Module):
    """
    Physics-Informed Neural Network for barometric pressure correction.
    
    Learns a spatial-temporal pressure correction field:
        δP(x, y, z, t, sensor_id) = f_θ(x, y, z, t, T, RH, sensor_emb)
    
    The corrected pressure is then converted to HAE via the physics model:
        P_true = P_obs + δP
        H_MSL = f_hypsometric(P_true, T_v)
        H_HAE = H_MSL + N(φ, λ)
    """
    
    def __init__(
        self,
        n_sensors: int,
        embedding_dim: int = 8,
        hash_levels: int = 16,
        hash_features: int = 2,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        temporal_freqs: int = 4,
        dropout: float = 0.0,
        use_siren: bool = True
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.embedding_dim = embedding_dim
        self.use_siren = use_siren
        
        # Sensor-specific embedding (learns calibration bias per device)
        self.sensor_embedding = nn.Embedding(n_sensors, embedding_dim)
        
        # Spatial encoding
        self.hash_encoding = HashEncoding(
            n_levels=hash_levels,
            n_features=hash_features,
            log2_hashmap_size=19
        )
        
        # Temporal encoding
        self.temporal_encoding = FourierTemporalEncoding(
            n_frequencies=temporal_freqs,
            max_period_hours=168.0  # Weekly patterns
        )
        
        # Calculate input dimension
        # hash_out + z(1) + temp_out + T(1) + RH(1) + sensor_emb
        in_dim = (
            self.hash_encoding.out_dim +  # Spatial
            1 +  # Z (altitude)
            self.temporal_encoding.out_dim +  # Temporal
            1 + 1 +  # Temperature, Humidity
            embedding_dim  # Sensor ID
        )
        
        # MLP layers
        if use_siren:
            layers = []
            # First layer - reduced w0 for stability
            layers.append(SirenLayer(in_dim, hidden_dim, w0=1.0, is_first=True))
            # Hidden layers
            for _ in range(n_hidden_layers):
                layers.append(SirenLayer(hidden_dim, hidden_dim, w0=1.0))
            # Output layer (linear, no activation)
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
        
        # Initialize output layer to near zero (start with small corrections)
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
        sensor_id: torch.Tensor,
        normalize_coords: bool = True
    ) -> torch.Tensor:
        """
        Forward pass predicting pressure correction δP.
        
        Args:
            lat: Latitude [B]
            lon: Longitude [B]
            z: Altitude [B]
            t: Time (Unix timestamp) [B]
            temperature: Temperature [B] (Celsius)
            humidity: Relative humidity [B] (percent)
            sensor_id: Sensor index [B] (int)
            normalize_coords: Whether to normalize lat/lon to [0, 1]
        
        Returns:
            Pressure correction δP in Pascals [B]
        """
        batch_size = lat.shape[0]
        
        # Normalize spatial coordinates to [0, 1]
        if normalize_coords:
            # Lat: [-90, 90] -> [0, 1]
            # Lon: [-180, 180] or [0, 360] -> [0, 1]
            lat_norm = (lat + 90.0) / 180.0
            lon_norm = lon % 360.0 / 360.0
        else:
            lat_norm = lat
            lon_norm = lon
        
        coords = torch.stack([lat_norm, lon_norm], dim=-1)
        
        # Encode features
        h_spatial = self.hash_encoding(coords)
        h_temporal = self.temporal_encoding(t / 3600.0)  # Convert seconds to hours
        h_sensor = self.sensor_embedding(sensor_id)
        
        # Stack all features
        features = torch.cat([
            h_spatial,
            z.unsqueeze(-1),
            h_temporal,
            temperature.unsqueeze(-1),
            humidity.unsqueeze(-1),
            h_sensor
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
        sensor_id: torch.Tensor,
        samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo dropout inference for uncertainty quantification.
        
        Returns:
            (mean, std) of pressure correction
        """
        self.train()  # Enable dropout
        preds = []
        for _ in range(samples):
            preds.append(self.forward(lat, lon, z, t, temperature, humidity, sensor_id))
        
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0)
        std = stack.std(dim=0)
        return mean, std


class PhysicsInformedLoss(nn.Module):
    """
    Multi-component loss for PINN training.
    
    Components:
    1. Data fidelity: |h_pred - h_gnss|
    2. Hydrostatic constraint: |dP/dz + ρg|
    3. ERA5 consistency: |P_pred - P_era5| (optional)
    4. Smoothness: |∇²δP| (optional)
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_hydro: float = 0.1,
        lambda_era5: float = 0.0,
        lambda_smooth: float = 0.01
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_hydro = lambda_hydro
        self.lambda_era5 = lambda_era5
        self.lambda_smooth = lambda_smooth
    
    def compute_hydrostatic_residual(
        self,
        model: PressureCorrectionPINN,
        lat: torch.Tensor,
        lon: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        temperature: torch.Tensor,
        humidity: torch.Tensor,
        sensor_id: torch.Tensor,
        p_obs: torch.Tensor,
        p_ref: float,
        g: float = 9.80665,
        r_dry: float = 287.05
    ) -> torch.Tensor:
        """
        Compute hydrostatic equation residual.
        
        The hydrostatic equation states:
            dP/dz = -ρg = -Pg/(R*T_v)
        
        We compute the residual:
            residual = dP_true/dz + P_true*g/(R*T_v)
        
        Where P_true = P_obs + δP
        """
        # Enable gradients for z
        z_with_grad = z.clone().requires_grad_(True)
        
        # Forward pass
        delta_p = model(lat, lon, z_with_grad, t, temperature, humidity, sensor_id)
        p_true = p_obs + delta_p
        
        # Compute dP/dz using autograd
        dp_dz = torch.autograd.grad(
            outputs=p_true,
            inputs=z_with_grad,
            grad_outputs=torch.ones_like(p_true),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute virtual temperature
        t_celsius = temperature
        e_sat = 610.94 * torch.exp(17.625 * t_celsius / (t_celsius + 243.04))
        e = (humidity / 100.0) * e_sat
        r = 0.62198 * e / (p_true - e)
        t_v = (t_celsius + 273.15) * (1 + 0.608 * r)
        
        # Hydrostatic equation residual
        residual = dp_dz + (p_true * g) / (r_dry * t_v)
        
        return residual
    
    def forward(
        self,
        model: PressureCorrectionPINN,
        h_pred: torch.Tensor,
        h_gnss: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total PINN loss.
        
        Returns:
            (total_loss, loss_components_dict)
        """
        losses = {}
        
        # Data fidelity loss
        loss_data = torch.mean(torch.abs(h_pred - h_gnss))
        losses['data'] = loss_data
        
        total_loss = self.lambda_data * loss_data
        
        # Hydrostatic constraint
        if self.lambda_hydro > 0 and 'p_obs' in kwargs:
            hydro_residual = self.compute_hydrostatic_residual(
                model,
                kwargs['lat'],
                kwargs['lon'],
                kwargs['z'],
                kwargs['t'],
                kwargs['temperature'],
                kwargs['humidity'],
                kwargs['sensor_id'],
                kwargs['p_obs'],
                kwargs.get('p_ref', 101325.0)
            )
            loss_hydro = torch.mean(hydro_residual ** 2)
            losses['hydrostatic'] = loss_hydro
            total_loss += self.lambda_hydro * loss_hydro
        
        return total_loss, losses
