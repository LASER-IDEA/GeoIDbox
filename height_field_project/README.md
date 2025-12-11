Height Conversion Field (Physics + Neural Residual Field)
==========================================================

目标
----
- 基于稀疏的“高度盒子”观测，构建可稠密查询的气压 ↔ MSL ↔ HAE 高度转换场。
- 物理模型作为强先验，神经场学习残差，未观测区自动回落到物理模型。
- 输出高度与不确定度。

文件结构
--------
- `config.py`：默认超参与路径。
- `physics_baseline.py`：气压-高度物理拟合与基准高度计算。
- `neural_field.py`：位置编码 + MLP 残差场，支持 MC Dropout。
- `train.py`：端到端训练（数据加载→物理基线→残差→神经场）。
- `infer.py`：加载模型推理，支持 MC Dropout 不确定度，网格或 CSV 点查询。
- `artifacts/`：模型与 scaler 持久化输出目录（训练时自动创建）。

快速开始（WSL，conda py310）
---------------------------
```bash
cd /mnt/d/workspace/workspace_wsl2/silas_data/height
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py310

# 1) 训练（使用 sensor_data_clean_stable.csv）
python -m height_field_project.train \
  --input_csv sensor_data_clean_stable.csv \
  --epochs 300 \
  --pseudo_ratio 1.0 \
  --pseudo_weight 0.5

# 2) 推理：对输入 CSV 生成校正高度与不确定度
python -m height_field_project.infer \
  --input_csv sensor_data_clean_stable.csv \
  --samples 20 \
  --out_csv artifacts/predictions.csv

# 3) 推理：生成经纬网格（固定高度切片）场
python -m height_field_project.infer \
  --grid_bbox 22.60 22.62 114.05 114.07 \
  --grid_res 80 \
  --grid_height 150 \
  --samples 30 \
  --out_csv artifacts/grid_slice.csv
```

建模要点
--------
- 物理基线：对 `ln(p)` 与观测高程做线性拟合，得到 Hs 与 P0；计算 `h_phys_m` 作为先验。
- 残差目标：`residual = h_obs - h_phys_m`。
- 特征：`lat, lon, h_phys_m, avg_temperature, avg_humidity, avg_pressure, week_seq`。
- 伪点：在观测外采样，标签 0，保证外推回落物理先验；权重由 `--pseudo_weight` 控制。
- 损失：Huber；随机点梯度正则可按需扩展。
- 不确定度：MC Dropout（多次前向计算均值/方差）。

数据假设
--------
- 输入 CSV 至少包含以下列：
  `avg_latitude, avg_longitude, avg_altitude, avg_pressure, avg_temperature, avg_humidity, week_seq`
- 若无 `week_seq`，脚本会自动填 0。

输出
-----
- 训练：`artifacts/model.pt`, `artifacts/scalers.pkl`, `artifacts/config.json`
- 推理：输出 CSV 包含
  - `h_phys_m`: 物理基线高度
  - `residual_mean`, `residual_std`
  - `h_pred_mean = h_phys_m + residual_mean`
  - `h_pred_std = residual_std`

后续可扩展
----------
- 接入 EGM geoid：在物理基线中加入 geoid undulation，输出 MSL/HAE。
- 接入 ERA5 垂直廓线：改进虚温与分层，替换简化气压拟合。
- 梯度正则 / 拉普拉斯正则：对 `∂Δ/∂z` 或 `∇²Δ` 做约束，进一步稳定外推。
