"""
Run Baseline Comparison Experiments
====================================
Systematic comparison of all baseline methods.

This script:
1. Loads and prepares data
2. Trains all baseline methods
3. Evaluates on test set
4. Generates comparison plots and tables
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.evaluation import HeightEstimationEvaluator, print_metrics_table
from experiments.baselines.classical_methods import (
    StandardAtmosphereModel,
    BarometricLinearRegression,
    KrigingHeightModel,
    PolynomialRegression,
    RBFInterpolation,
    get_all_baselines
)
from experiments.baselines.ml_methods import (
    XGBoostHeightModel,
    RandomForestHeightModel,
    GaussianProcessHeightModel,
    StandardMLP,
    get_ml_baselines
)

# Try to import height_field_project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'height_field_project'))
try:
    from physics_baseline import fit_barometric_baseline
    from neural_field import ResidualNeuralField
    from train import prepare_features
    HAS_NEURAL_FIELD = True
except ImportError:
    HAS_NEURAL_FIELD = False
    print("Warning: height_field_project not fully available")


def load_and_prepare_data(data_path: str, era5_path: str = None):
    """
    Load and prepare data for experiments.
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Add temporal features if processed_time exists
    if 'processed_time' in df.columns:
        df['processed_time'] = pd.to_datetime(df['processed_time'])
        df['hour'] = df['processed_time'].dt.hour
        df['day_of_week'] = df['processed_time'].dt.dayofweek
    
    # Add week_seq if not present
    if 'week_seq' not in df.columns:
        df['week_seq'] = 0
    
    # Compute physics baseline if pressure available
    if 'avg_pressure' in df.columns and 'avg_altitude' in df.columns:
        try:
            df, baseline_params = fit_barometric_baseline(df)
            print(f"Physics baseline: Hs={baseline_params['Hs_m']:.1f}m, "
                  f"P0={baseline_params['P0_Pa']:.1f}Pa")
        except Exception as e:
            print(f"Could not compute physics baseline: {e}")
            df['h_phys_m'] = df['avg_altitude']  # Fallback
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # Stratified split by week to ensure temporal coverage
    if 'week_seq' in df.columns and df['week_seq'].nunique() > 1:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, 
            stratify=df['week_seq']
        )
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42
        )
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    return train_df, test_df


def run_classical_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                            evaluator: HeightEstimationEvaluator) -> Dict:
    """
    Run classical baseline methods.
    """
    print("\n" + "="*60)
    print("Running Classical Baselines")
    print("="*60)
    
    results = {}
    all_predictions = {}
    
    # 1. Standard Atmosphere Model
    print("\n[1/5] Standard Atmosphere Model...")
    try:
        isa = StandardAtmosphereModel()
        isa.fit(train_df)
        
        if 'avg_pressure' in test_df.columns:
            preds = isa.predict(test_df['avg_pressure'].values)
            metrics = evaluator.compute_basic_metrics(
                test_df['avg_altitude'].values,
                preds
            )
            results['ISA'] = metrics
            all_predictions['ISA'] = preds - test_df['avg_altitude'].values
            print_metrics_table(metrics, "Standard Atmosphere Model")
    except Exception as e:
        print(f"ISA failed: {e}")
    
    # 2. Barometric Linear Regression
    print("\n[2/5] Barometric Linear Regression...")
    try:
        baro = BarometricLinearRegression()
        baro.fit(train_df)
        
        if 'avg_pressure' in test_df.columns:
            preds = baro.predict(test_df['avg_pressure'].values)
            metrics = evaluator.compute_basic_metrics(
                test_df['avg_altitude'].values,
                preds
            )
            results['Baro_Linear'] = metrics
            all_predictions['Baro_Linear'] = preds - test_df['avg_altitude'].values
            print_metrics_table(metrics, "Barometric Linear Regression")
    except Exception as e:
        print(f"Barometric Linear failed: {e}")
    
    # 3. Polynomial Regression (with features)
    print("\n[3/5] Polynomial Regression...")
    try:
        feature_cols = ['avg_latitude', 'avg_longitude', 'avg_pressure', 
                       'avg_temperature', 'avg_humidity']
        available_cols = [c for c in feature_cols if c in train_df.columns]
        
        if len(available_cols) >= 3:
            poly = PolynomialRegression(degree=2)
            poly.fit(train_df, available_cols)
            preds = poly.predict(test_df)
            metrics = evaluator.compute_basic_metrics(
                test_df['avg_altitude'].values,
                preds
            )
            results['PolyReg_D2'] = metrics
            all_predictions['PolyReg_D2'] = preds - test_df['avg_altitude'].values
            print_metrics_table(metrics, "Polynomial Regression (Degree 2)")
    except Exception as e:
        print(f"Polynomial Regression failed: {e}")
    
    # 4. Kriging
    print("\n[4/5] Kriging...")
    try:
        krig = KrigingHeightModel()
        krig.fit(train_df)
        preds = krig.predict(
            test_df['avg_latitude'].values,
            test_df['avg_longitude'].values
        )
        metrics = evaluator.compute_basic_metrics(
            test_df['avg_altitude'].values,
            preds
        )
        results['Kriging'] = metrics
        all_predictions['Kriging'] = preds - test_df['avg_altitude'].values
        print_metrics_table(metrics, "Kriging")
    except Exception as e:
        print(f"Kriging failed: {e}")
    
    # 5. RBF Interpolation
    print("\n[5/5] RBF Interpolation...")
    try:
        rbf = RBFInterpolation()
        rbf.fit(train_df)
        preds = rbf.predict(
            test_df['avg_latitude'].values,
            test_df['avg_longitude'].values
        )
        metrics = evaluator.compute_basic_metrics(
            test_df['avg_altitude'].values,
            preds
        )
        results['RBF'] = metrics
        all_predictions['RBF'] = preds - test_df['avg_altitude'].values
        print_metrics_table(metrics, "RBF Interpolation")
    except Exception as e:
        print(f"RBF failed: {e}")
    
    return results, all_predictions


def run_ml_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     evaluator: HeightEstimationEvaluator) -> Dict:
    """
    Run ML baseline methods.
    """
    print("\n" + "="*60)
    print("Running ML Baselines")
    print("="*60)
    
    results = {}
    all_predictions = {}
    
    feature_cols = ['avg_latitude', 'avg_longitude', 'avg_pressure',
                   'avg_temperature', 'avg_humidity', 'week_seq']
    available_cols = [c for c in feature_cols if c in train_df.columns]
    
    if len(available_cols) < 3:
        print("Not enough features for ML methods")
        return results, all_predictions
    
    # 1. Random Forest
    print("\n[1/4] Random Forest...")
    try:
        rf = RandomForestHeightModel(n_estimators=100)
        rf.fit(train_df, available_cols)
        preds = rf.predict(test_df)
        metrics = evaluator.compute_basic_metrics(
            test_df['avg_altitude'].values,
            preds
        )
        results['RandomForest'] = metrics
        all_predictions['RandomForest'] = preds - test_df['avg_altitude'].values
        print_metrics_table(metrics, "Random Forest")
        
        # Feature importance
        importance = rf.feature_importance()
        print("Feature Importance:")
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.3f}")
    except Exception as e:
        print(f"Random Forest failed: {e}")
    
    # 2. XGBoost
    print("\n[2/4] XGBoost...")
    try:
        xgb = XGBoostHeightModel(n_estimators=100)
        xgb.fit(train_df, available_cols)
        preds = xgb.predict(test_df)
        metrics = evaluator.compute_basic_metrics(
            test_df['avg_altitude'].values,
            preds
        )
        results['XGBoost'] = metrics
        all_predictions['XGBoost'] = preds - test_df['avg_altitude'].values
        print_metrics_table(metrics, "XGBoost")
    except Exception as e:
        print(f"XGBoost failed: {e}")
    
    # 3. Standard MLP
    print("\n[3/4] Standard MLP...")
    try:
        mlp = StandardMLP(hidden_layers=(128, 128, 128))
        mlp.fit(train_df, available_cols)
        preds = mlp.predict(test_df)
        metrics = evaluator.compute_basic_metrics(
            test_df['avg_altitude'].values,
            preds
        )
        results['StandardMLP'] = metrics
        all_predictions['StandardMLP'] = preds - test_df['avg_altitude'].values
        print_metrics_table(metrics, "Standard MLP")
    except Exception as e:
        print(f"Standard MLP failed: {e}")
    
    # 4. Gaussian Process
    print("\n[4/4] Gaussian Process...")
    try:
        gp = GaussianProcessHeightModel()
        gp.fit(train_df, available_cols, max_samples=3000)
        preds = gp.predict(test_df)
        metrics = evaluator.compute_basic_metrics(
            test_df['avg_altitude'].values,
            preds
        )
        results['GaussianProcess'] = metrics
        all_predictions['GaussianProcess'] = preds - test_df['avg_altitude'].values
        print_metrics_table(metrics, "Gaussian Process")
    except Exception as e:
        print(f"Gaussian Process failed: {e}")
    
    return results, all_predictions


def run_neural_field_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                evaluator: HeightEstimationEvaluator) -> Dict:
    """
    Run neural field method if available.
    """
    print("\n" + "="*60)
    print("Running Neural Field (if available)")
    print("="*60)
    
    results = {}
    all_predictions = {}
    
    if not HAS_NEURAL_FIELD:
        print("Neural field module not available, skipping")
        return results, all_predictions
    
    try:
        import torch
        from height_field_project.train import prepare_features
        from height_field_project.neural_field import ResidualNeuralField
        
        print("\nTraining Neural Field...")
        # This would require integrating with the training pipeline
        # For now, skip detailed implementation
        print("Neural field training integrated in separate pipeline")
        
    except Exception as e:
        print(f"Neural field failed: {e}")
    
    return results, all_predictions


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison experiments")
    parser.add_argument("--data", type=str, 
                       default="data/processed/sensor_data_clean_stable.csv",
                       help="Path to input CSV")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/results",
                       help="Output directory for results")
    parser.add_argument("--methods", type=str,
                       default="all",
                       help="Methods to run: all, classical, ml")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HeightEstimationEvaluator(output_dir=args.output_dir)
    
    # Load data
    train_df, test_df = load_and_prepare_data(args.data)
    
    all_results = {}
    all_predictions = {}
    
    # Run experiments
    if args.methods in ["all", "classical"]:
        results, preds = run_classical_baselines(train_df, test_df, evaluator)
        all_results.update(results)
        all_predictions.update(preds)
    
    if args.methods in ["all", "ml"]:
        results, preds = run_ml_baselines(train_df, test_df, evaluator)
        all_results.update(results)
        all_predictions.update(preds)
    
    # Generate comparison plots
    print("\n" + "="*60)
    print("Generating Comparison Plots")
    print("="*60)
    
    evaluator.plot_error_distribution(
        all_predictions,
        save_path=os.path.join(args.output_dir, "figures", "baseline_comparison.png")
    )
    
    # Generate report table
    print("\nGenerating comparison table...")
    df_results = evaluator.generate_report(
        all_results,
        save_path=os.path.join(args.output_dir, "tables", "baseline_comparison.csv")
    )
    
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
