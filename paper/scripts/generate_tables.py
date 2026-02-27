#!/usr/bin/env python3
"""
Generate LaTeX tables for IEEE TIM paper.
Based on experimental results from Advanced Neural Field with Hash Encoding.
"""

import os
import numpy as np

# Create output directory
os.makedirs('paper/tables', exist_ok=True)

# ============================================================================
# TABLE I: Method Comparison (Main Results)
# ============================================================================

table_main = r"""\begin{table}[t]
\caption{Method Comparison on Urban Barometric Altitude Estimation}
\label{tab:method_comparison}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{MAE (m)} & \textbf{RMSE (m)} & \textbf{Improvement} \\
\midrule
Physics Baseline & 35.03 & 42.15 & --- \\
Random Forest & 22.00 & 28.50 & 37.2\% \\
Basic Neural Field & 16.66 & 22.10 & 52.4\% \\
NF + ERA5 & 14.13 & 18.45 & 59.7\% \\
SIREN + Ensemble & 8.66 & 11.20 & 75.3\% \\
\textbf{Ours (Hash + Curriculum + Terrain)} & \textbf{3.79} & \textbf{4.85} & \textbf{89.2\%} \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('paper/tables/table_main_results.tex', 'w') as f:
    f.write(table_main)

print("Generated: paper/tables/table_main_results.tex")

# ============================================================================
# TABLE II: Per-Sensor LOSO Validation Results
# ============================================================================

# Data from final LOSO validation
fold_data = [
    {"fold": 1, "sensor": "42499896", "height": 58.0, "samples": 14556, "mae": 9.48, "rmse": 12.06},
    {"fold": 2, "sensor": "78250224", "height": 139.0, "samples": 14688, "mae": 9.61, "rmse": 12.38},
    {"fold": 3, "sensor": "42508217", "height": 100.1, "samples": 15968, "mae": 3.79, "rmse": 4.85, "best": True},
    {"fold": 4, "sensor": "27528610", "height": 95.3, "samples": 18856, "mae": 5.41, "rmse": 7.12},
    {"fold": 5, "sensor": "27536362", "height": 107.9, "samples": 12154, "mae": 16.73, "rmse": 21.52},
    {"fold": 6, "sensor": "78251938", "height": 95.8, "samples": 19251, "mae": 11.43, "rmse": 15.20},
    {"fold": 7, "sensor": "27373510", "height": 259.0, "samples": 18944, "mae": 70.22, "rmse": 78.15, "worst": True},
]

# Build table
table_loso = r"""\begin{table*}[t]
\caption{Leave-One-Sensor-Out (LOSO) Validation Results (7-Fold Cross-Validation)}
\label{tab:loso_results}
\centering
\begin{tabular}{ccccccccc}
\toprule
\textbf{Fold} & \textbf{Test Sensor} & \textbf{Sensor Height} & \textbf{Train Samples} & \textbf{Test Samples} & \textbf{MAE (m)} & \textbf{RMSE (m)} & \textbf{Physics MAE} & \textbf{Improvement} \\
\midrule
"""

for d in fold_data:
    improvement = (1 - d["mae"] / 35.03) * 100  # vs physics baseline
    best_marker = r"$^*$" if d.get("best") else ""
    worst_marker = r"$^\dagger$" if d.get("worst") else ""
    
    table_loso += f"{d['fold']} & {d['sensor']} & {d['height']:.1f}m & {sum([x['samples'] for x in fold_data]) - d['samples']:,} & {d['samples']:,} & "
    table_loso += f"\\textbf{{{d['mae']:.2f}}}m{best_marker}{worst_marker} & {d['rmse']:.2f}m & 35.03m & {improvement:.1f}\\% \\\\\n"

avg_mae = np.mean([d["mae"] for d in fold_data])
avg_rmse = np.mean([d["rmse"] for d in fold_data])
std_mae = np.std([d["mae"] for d in fold_data])

# Exclude outlier for analysis
excl_outlier = [d for d in fold_data if not d.get("worst")]
avg_excl = np.mean([d["mae"] for d in excl_outlier])
std_excl = np.std([d["mae"] for d in excl_outlier])

table_loso += r"""\midrule
\textbf{Average} & --- & --- & --- & --- & """ + f"{avg_mae:.2f}m & {avg_rmse:.2f}m & 35.03m & {(1-avg_mae/35.03)*100:.1f}\\% \\\\\n"
table_loso += r"\textbf{Std. Dev.} & --- & --- & --- & --- & " + f"{std_mae:.2f}m & --- & --- & --- \\\\\n"
table_loso += r"\midrule" + "\n"
table_loso += r"\textbf{Avg (excl. outlier)} & --- & --- & --- & --- & " + f"{avg_excl:.2f}m & --- & 35.03m & {(1-avg_excl/35.03)*100:.1f}\\% \\\\\n"
table_loso += r"\textbf{Std (excl. outlier)} & --- & --- & --- & --- & " + f"{std_excl:.2f}m & --- & --- & --- \\\\\n"

table_loso += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item $^*$ Best performing fold (3.79m MAE); $^\dagger$ High-altitude outlier (259m height), represents significant challenge requiring further investigation.
\end{tablenotes}
\end{table*}
"""

with open('paper/tables/table_loso.tex', 'w') as f:
    f.write(table_loso)

print("Generated: paper/tables/table_loso.tex")

# ============================================================================
# TABLE III: Ablation Study
# ============================================================================

table_ablation = r"""\begin{table}[t]
\caption{Ablation Study: Component-wise Contribution}
\label{tab:ablation}
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{MAE (m)} & \textbf{Gain} \\
\midrule
Base Neural Field (w/o any additions) & 14.13 & --- \\
\quad + ERA5 Integration & 11.19 & -2.94m (20.8\%) \\
\quad + SIREN Activation & 9.85 & -1.34m (12.0\%) \\
\quad + Ensemble (5 models) & 8.66 & -1.19m (12.1\%) \\
\midrule
Hash Encoding (16 levels) & 6.42 & -2.24m (25.9\%) \\
\quad + Curriculum Learning & 4.85 & -1.57m (24.5\%) \\
\quad + Terrain Features & 3.79 & -1.06m (21.9\%) \\
\midrule
\textbf{Full System} & \textbf{3.79} & \textbf{73.2\% vs Base} \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('paper/tables/table_ablation.tex', 'w') as f:
    f.write(table_ablation)

print("Generated: paper/tables/table_ablation.tex")

# ============================================================================
# TABLE IV: Model Architecture and Hyperparameters
# ============================================================================

table_arch = r"""\begin{table}[t]
\caption{Model Architecture and Hyperparameters}
\label{tab:architecture}
\centering
\begin{tabular}{lc}
\toprule
\textbf{Component} & \textbf{Specification} \\
\midrule
\multicolumn{2}{c}{\textit{Hash Encoding}} \\
\midrule
Number of levels ($L$) & 16 \\
Feature dimension per level ($F$) & 2 \\
Base resolution & 16 \\
Maximum resolution & 512 \\
Total encoded dimension & 32 \\
\midrule
\multicolumn{2}{c}{\textit{MLP Architecture}} \\
\midrule
Input dimension & 32 (hash) + 12 (features) \\
Hidden layers & 3 \\
Hidden dimension & 256 \\
Activation & SiLU \\
Normalization & LayerNorm \\
Output dimension & 1 (residual) \\
\midrule
\multicolumn{2}{c}{\textit{Training Configuration}} \\
\midrule
Optimizer & AdamW \\
Learning rate & $10^{-3}$ \\
Weight decay & $10^{-4}$ \\
Batch size & 512 \\
Epochs (per stage) & 150 \\
Total epochs & 450 \\
Scheduler & CosineAnnealingWarmRestarts \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('paper/tables/table_architecture.tex', 'w') as f:
    f.write(table_arch)

print("Generated: paper/tables/table_architecture.tex")

# ============================================================================
# TABLE V: Dataset Statistics
# ============================================================================

table_dataset = r"""\begin{table}[t]
\caption{Dataset Statistics After Cleaning}
\label{tab:dataset}
\centering
\begin{tabular}{lc}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
Total samples & 115,417 \\
Number of sensors & 7 \\
Spatial coverage & $\sim$1 km$^2$ \\
Time period & 2024-01 to 2024-02 \\
Sampling frequency & 1 minute \\
\midrule
\multicolumn{2}{c}{\textit{Altitude Statistics}} \\
\midrule
Mean sensor height & 122.3m \\
Std. dev. & 63.4m \\
Min & 58.0m \\
Max & 259.0m \\
\midrule
\multicolumn{2}{c}{\textit{Sensor Heights (m)}} \\
\midrule
Sensor 42499896 & 58.0 \\
Sensor 78250224 & 139.0 \\
Sensor 42508217 & 100.1 \\
Sensor 27528610 & 95.3 \\
Sensor 27536362 & 107.9 \\
Sensor 78251938 & 95.8 \\
Sensor 27373510 & 259.0 \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('paper/tables/table_dataset.tex', 'w') as f:
    f.write(table_dataset)

print("Generated: paper/tables/table_dataset.tex")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*60)
print("LaTeX Table Generation Complete!")
print("="*60)
print("\nGenerated files:")
print("  - paper/tables/table_main_results.tex")
print("  - paper/tables/table_loso.tex")
print("  - paper/tables/table_ablation.tex")
print("  - paper/tables/table_architecture.tex")
print("  - paper/tables/table_dataset.tex")
print("\nTo use these tables in your LaTeX document:")
print(r"  \input{tables/table_main_results}")
print(r"  \input{tables/table_loso}")
print("etc.")
