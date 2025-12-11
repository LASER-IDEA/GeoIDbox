import torch
import torch.nn as nn
from typing import Optional


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, L: int = 6, sigma: float = 10.0):
        super().__init__()
        self.B = sigma * torch.randn(in_dim, L)
        self.out_dim = in_dim + 2 * in_dim * L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        xb = 2.0 * torch.pi * x @ self.B  # [B, L]
        sin = torch.sin(xb)
        cos = torch.cos(xb)
        return torch.cat([x, sin, cos], dim=-1)


class ResidualNeuralField(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        depth: int = 5,
        fourier_L: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pe = FourierFeatures(in_dim, L=fourier_L)
        feat_dim = self.pe.out_dim

        layers = []
        for i in range(depth):
            inp = feat_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(inp, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        return self.net(x).squeeze(-1)

    def predict_mc(self, x: torch.Tensor, samples: int = 20) -> (torch.Tensor, torch.Tensor):
        """
        MC Dropout 推理，返回均值和标准差。
        """
        preds = []
        self.train()  # 启用 dropout
        for _ in range(samples):
            preds.append(self.forward(x))
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0)
        std = stack.std(dim=0)
        return mean, std
