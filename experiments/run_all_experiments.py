"""
Run All Experiments for IEEE TIM Paper
======================================

Master script to run all experiments:
1. Baseline comparison
2. Deep Ensemble training
3. ST-GNN ablation study
4. Generate comparison plots and tables
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def run_command(cmd: list, description: str):
    """Run a command and print output."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"WARNING: Command failed with code {result.returncode}")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--data", type=str,
                       default="data/processed/sensor_data_clean_stable.csv")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/results")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_ensemble", action="store_true")
    parser.add_argument("--skip_stgnn", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("="*70)
    print("IEEE TIM Experiments - Full Pipeline")
    print("="*70)
    print(f"Start time: {start_time}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    success_count = 0
    total_count = 0
    
    # 1. Baseline Comparison
    if not args.skip_baselines:
        total_count += 1
        if run_command(
            [
                sys.executable, "-m", "experiments.run_baseline_comparison",
                "--data", args.data,
                "--output_dir", f"{args.output_dir}/baselines",
                "--methods", "all"
            ],
            "Baseline Comparison"
        ):
            success_count += 1
    
    # 2. Deep Ensemble
    if not args.skip_ensemble:
        total_count += 1
        if run_command(
            [
                sys.executable, "-m", "experiments.deep_ensemble.deep_ensemble_trainer",
                "--data", args.data,
                "--output_dir", f"{args.output_dir}/deep_ensemble",
                "--n_models", "5",
                "--epochs", str(args.epochs)
            ],
            "Deep Ensemble Training"
        ):
            success_count += 1
    
    # 3. ST-GNN Ablation
    if not args.skip_stgnn:
        total_count += 1
        if run_command(
            [
                sys.executable, "-m", "experiments.st_gnn.train_st_gnn",
                "--data", args.data,
                "--output_dir", f"{args.output_dir}/st_gnn",
                "--epochs", str(args.epochs)
            ],
            "ST-GNN Ablation Study"
        ):
            success_count += 1
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Completed: {success_count}/{total_count} experiments")
    print(f"Duration: {duration}")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)
    
    # List generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith(('.csv', '.png', '.json', '.pt')):
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                print(f"  {filepath} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
