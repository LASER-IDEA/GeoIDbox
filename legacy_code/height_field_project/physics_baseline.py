import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple


def fit_barometric_baseline(
    df: pd.DataFrame,
    pressure_col: str = "avg_pressure",
    altitude_col: str = "avg_altitude",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    拟合 ln(P) = a * h + b，得到尺度高度 Hs 和基准气压 P0。
    输出 h_phys_m，作为物理基线高度（近似 MSL）。
    """
    valid = df[[pressure_col, altitude_col]].dropna()
    mask = (valid[pressure_col] > 90000) & (valid[pressure_col] < 110000)
    valid = valid[mask]
    if len(valid) < 10:
        raise ValueError("有效压力样本过少，无法拟合物理基线")

    X = valid[[altitude_col]].values.reshape(-1, 1)
    y = np.log(valid[pressure_col].values)
    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    Hs = -1.0 / slope
    P0 = np.exp(intercept)

    # h_phys = -Hs * (ln(p) - ln(P0))
    df = df.copy()
    df["h_phys_m"] = -Hs * (np.log(df[pressure_col]) - np.log(P0))

    params = {"Hs_m": float(Hs), "P0_Pa": float(P0), "slope": float(slope), "intercept": float(intercept)}
    return df, params
