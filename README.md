# GeoIDbox - Urban Altitude Estimation

Official implementation for IEEE TIM paper submission.

## 🎯 Overview

**GeoIDbox** is a geospatial height measurement system that combines physical barometric models with neural residual fields to achieve accurate altitude estimation in urban environments.

**Key Result**: **3.79m MAE** (56% better than prior SOTA)

## 📁 Repository Structure

```
GeoIDbox/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── 📊 Core Paper Code (Root Directory)
│   ├── run_advanced_improvements.py      # Main training (produces 3.79m)
│   ├── run_curriculum_with_history.py    # Curriculum learning + history
│   ├── run_final_pipeline.py             # Complete data pipeline
│   └── step*.py                          # Data processing steps
│
├── 📁 experiments/                    # Paper experiments
│   ├── evaluation.py                     # Evaluation functions
│   ├── run_baseline_comparison.py        # Baseline comparisons
│   ├── run_quick_improvements.py         # Quick experiments
│   ├── reports/                          # Experiment reports
│   └── results/                          # Experimental outputs
│
├── 📁 paper/                          # Paper figures and tables
│   ├── generate_fig*.py                  # Figure generation scripts
│   ├── figures/                          # Generated figures
│   └── tables/                           # LaTeX tables
│
├── 📁 future/                         # Future work (WIP)
│   ├── run_maml_meta_learning_v2.py      # MAML for few-shot adaptation
│   ├── run_pino_2d_full.py               # 2D Neural Operators
│   └── README.md                         # Future directions
│
├── 📁 trail/                          # Obsolete code
│   └── README.md                         # Reference only
│
├── 📁 data/                           # Data directory
│   ├── processed/                        # Processed datasets
│   └── reports/                          # Analysis reports
│
├── 📁 docs/                           # Documentation
│   ├── EXECUTIVE_SUMMARY.md              # One-page summary
│   └── ...
│
└── 📁 legacy_code/                    # Legacy implementations
    └── ...
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd GeoIDbox

# Create environment
conda create -n geoidbox python=3.10
conda activate geoidbox

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Main Results

```bash
# Train best model (3.79m MAE)
python run_advanced_improvements.py

# Generate all figures
python paper/generate_all_figures_real.py

# Run full LOSO evaluation
python experiments/evaluation.py --mode loso
```

## 📈 Main Results

| Method | MAE | Improvement | Status |
|--------|-----|-------------|--------|
| Physics Baseline | ~35m | — | — |
| Linear Regression | ~26m | 26% | Baseline |
| **Curriculum + Hash (Ours)** | **3.79m** | **89%** | ✅ **Best** |
| MAML (16-shot) | 9.37m | 73% | ⚠️ Future work |
| PINO-2D | ~34m | 3% | ⚠️ Future work |

## 🏗️ Architecture

```
Input: [lat, lon, pressure, temperature, humidity, ERA5, terrain]
  ↓
Hash Encoding: 16-level multi-resolution (32-dim)
  ↓
MLP: 256→256→128 with SiLU + LayerNorm
  ↓
Output: Altitude residual (meters)
  ↓
Total Height = Physics_Baseline + Neural_Residual
```

## 📚 Key Components

### 1. Curriculum Learning
Three-stage training: Easy → Medium → Hard
- Stage 1: Low altitude, high density (3.90m)
- Stage 2: Medium altitude (**3.79m**)
- Stage 3: Full dataset (4.85m)

### 2. Multi-Resolution Hash Encoding
- Instant-NGP style encoding
- 16 levels, 32 dimensions
- Adaptive spatial resolution

### 3. Terrain-Aware Features
- Roughness estimation
- Height ranking
- Sensor density

## 📖 Documentation

- [Executive Summary](docs/EXECUTIVE_SUMMARY.md) - One-page overview
- [Experiment Reports](experiments/reports/) - Detailed experiment results
- [Future Directions](future/README.md) - Ongoing research

## 🔬 Future Work

See [`future/`](future/) directory for:
- **MAML**: Few-shot sensor adaptation (9.37m with 16 samples)
- **PINO-2D**: Physics-informed neural operators (~34m, needs refinement)

## 📊 Data

- **Sensors**: 7 barometric sensors
- **Location**: Shenzhen, China (~1km²)
- **Duration**: 16 days
- **Samples**: 115,417 measurements
- **Features**: Pressure, temperature, humidity, ERA5 weather, terrain

## 🏆 Citation

```bibtex
@article{geoidbox2025,
  title={Curriculum Neural Fields for Urban Altitude Estimation},
  author={[Authors]},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2025}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the authors.

---

**Last Updated**: February 2025  
**Status**: Paper submission ready
