#!/bin/bash
# Master script to run all experiments

set -e

echo "================================================================================"
echo "COMPREHENSIVE EXPERIMENTAL SUITE FOR IEEE TIM"
echo "================================================================================"
echo ""
echo "This script runs all experiments:"
echo "  1. Baseline Comparisons (IDW, Kriging, RF, XGBoost)"
echo "  2. Ablation Studies (4 setups)"
echo "  3. Uncertainty Quantification (MC Dropout, Latency, Memory)"
echo "  4. Figure Generation (Publication-quality plots)"
echo ""
echo "Estimated total time: 2-4 hours"
echo "================================================================================"
echo ""

# Activate environment
echo "Activating conda environment..."
source /data/home/huxiao/miniconda3/bin/activate graphmamba

# Create output directories
mkdir -p experiments/figures
mkdir -p experiments/results

echo ""
echo "================================================================================"
echo "EXPERIMENT 1: Baseline Comparisons"
echo "================================================================================"
echo "Comparing PINN against:"
echo "  - IDW (Inverse Distance Weighting)"
echo "  - Ordinary Kriging"
echo "  - Random Forest"
echo "  - XGBoost"
echo ""

cd /data/home/huxiao/workspace/GeoIDbox
python experiments/01_baseline_comparisons/run_baseline_comparisons.py

echo ""
echo "================================================================================"
echo "EXPERIMENT 2: Ablation Studies"
echo "================================================================================"
echo "Testing 4 configurations:"
echo "  A: Base SIREN (No P_bias, No Curriculum)"
echo "  B: SIREN + P_bias"
echo "  D: Full (SIREN + P_bias + Curriculum)"
echo ""

python experiments/02_ablation_studies/run_ablation_studies.py

echo ""
echo "================================================================================"
echo "EXPERIMENT 3: Uncertainty Quantification"
echo "================================================================================"
echo "Measuring:"
echo "  - MC Dropout uncertainty vs actual error"
echo "  - Inference latency (ms/query)"
echo "  - Memory footprint (MB)"
echo ""

python experiments/03_uncertainty_quantification/run_uncertainty_quantification.py

echo ""
echo "================================================================================"
echo "GENERATING FIGURES"
echo "================================================================================"
echo "Creating publication-quality figures:"
echo "  - Figure 1: Baseline Comparison"
echo "  - Figure 2: Ablation Study"
echo "  - Figure 3: Uncertainty vs Error"
echo "  - Figure 4: Spatial Uncertainty Map"
echo "  - Figure 5: Per-Fold Results"
echo ""

python experiments/generate_figures.py

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "================================================================================"
echo ""
echo "Results location:"
echo "  - experiments/01_baseline_comparisons/results.csv"
echo "  - experiments/02_ablation_studies/results.csv"
echo "  - experiments/03_uncertainty_quantification/"
echo "    ├── uq_data.csv"
echo "    └── summary.csv"
echo ""
echo "Figures location:"
echo "  - experiments/figures/"
echo "    ├── 01_baseline_comparison.pdf/png"
echo "    ├── 02_ablation_study.pdf/png"
echo "    ├── 03_uncertainty_vs_error.pdf/png"
echo "    ├── 04_spatial_uncertainty.pdf/png"
echo "    └── 05_per_fold_results.pdf/png"
echo ""
echo "================================================================================"
