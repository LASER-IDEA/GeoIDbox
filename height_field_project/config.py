import os
from dataclasses import dataclass, asdict
import json


@dataclass
class TrainingConfig:
    input_csv: str = "sensor_data_clean_stable.csv"
    artifacts_dir: str = os.path.join("height_field_project", "artifacts")
    epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    hidden_dim: int = 128
    depth: int = 5
    fourier_L: int = 6
    dropout: float = 0.1
    pseudo_ratio: float = 1.0  # 伪点数量 = ratio * 实点数量
    pseudo_weight: float = 0.5  # 伪点损失权重
    huber_delta: float = 1.0
    seed: int = 42
    lambda_phys: float = 0.1  # 物理残差损失权重


def save_config(cfg: TrainingConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def load_config(path: str) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainingConfig(**data)
