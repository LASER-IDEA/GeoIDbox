"""
Evaluation Utilities for IEEE TIM Experiments
==============================================
Comprehensive evaluation metrics for height estimation models.

Metrics:
- RMSE, MAE, MAPE (standard)
- Uncertainty calibration (ECE, reliability diagrams)
- Stratified analysis (by weather, height, time)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import os


class HeightEstimationEvaluator:
    """
    Comprehensive evaluator for barometric height estimation.
    """
    
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
        
    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute standard regression metrics.
        """
        # Filter NaN values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_t, y_p = y_true[mask], y_pred[mask]
        
        if len(y_t) == 0:
            return {"error": "No valid samples"}
        
        errors = y_p - y_t
        
        metrics = {
            "rmse": np.sqrt(np.mean(errors ** 2)),
            "mae": np.mean(np.abs(errors)),
            "mape": np.mean(np.abs(errors / (y_t + 1e-8))) * 100,
            "bias": np.mean(errors),
            "std": np.std(errors),
            "max_error": np.max(np.abs(errors)),
            "median_ae": np.median(np.abs(errors)),
            "r2": 1 - np.sum(errors ** 2) / np.sum((y_t - np.mean(y_t)) ** 2),
            "n_samples": len(y_t),
        }
        
        # Percentiles
        abs_errors = np.abs(errors)
        metrics["p50"] = np.percentile(abs_errors, 50)
        metrics["p75"] = np.percentile(abs_errors, 75)
        metrics["p90"] = np.percentile(abs_errors, 90)
        metrics["p95"] = np.percentile(abs_errors, 95)
        metrics["p99"] = np.percentile(abs_errors, 99)
        
        return metrics
    
    def compute_uncertainty_calibration(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_std: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Compute uncertainty calibration metrics.
        
        Expected Calibration Error (ECE): Measures how well predicted 
        uncertainties match actual errors.
        """
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_std)
        y_t, y_p, y_s = y_true[mask], y_pred[mask], y_std[mask]
        
        if len(y_t) == 0 or np.any(y_s <= 0):
            return {"error": "Invalid uncertainty values"}
        
        errors = np.abs(y_p - y_t)
        
        # Normalize errors by predicted std
        z_scores = errors / y_s
        
        # Bin by predicted uncertainty
        bin_edges = np.percentile(y_s, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-8  # Ensure last value is included
        
        ece = 0.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            mask_bin = (y_s >= bin_edges[i]) & (y_s < bin_edges[i + 1])
            if np.sum(mask_bin) == 0:
                continue
            
            bin_errors = errors[mask_bin]
            bin_std = y_s[mask_bin]
            
            # Expected: 68% of errors within 1 std, 95% within 2 std
            expected_within_1std = np.mean(bin_errors <= bin_std)
            expected_within_2std = np.mean(bin_errors <= 2 * bin_std)
            
            bin_accuracies.append(expected_within_1std)
            bin_confidences.append(0.6827)  # 68.27% for 1 std in Gaussian
            bin_counts.append(np.sum(mask_bin))
            
            ece += np.abs(expected_within_1std - 0.6827) * np.sum(mask_bin)
        
        ece /= len(y_t)
        
        # Negative Log Likelihood (lower is better)
        nll = np.mean(0.5 * np.log(2 * np.pi * y_s ** 2) + (errors ** 2) / (2 * y_s ** 2))
        
        # Continuous Ranked Probability Score (CRPS)
        crps = np.mean(errors * (2 * stats.norm.cdf(errors / y_s) - 1) + 
                       y_s * (2 * stats.norm.pdf(errors / y_s) - 1 / np.sqrt(np.pi)))
        
        return {
            "ece": ece,
            "nll": nll,
            "crps": crps,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
        }
    
    def stratified_analysis(
        self,
        df: pd.DataFrame,
        y_true_col: str = "avg_altitude",
        y_pred_col: str = "h_pred_mean",
        strata_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Perform stratified error analysis.
        
        Strata examples:
        - avg_pressure: weather conditions
        - hour: time of day
        - week_seq: seasonal variation
        """
        if strata_cols is None:
            strata_cols = ["avg_pressure", "avg_temperature", "hour", "week_seq"]
        
        results = []
        
        for col in strata_cols:
            if col not in df.columns:
                continue
            
            # Create bins
            if col == "hour":
                bins = [0, 6, 12, 18, 24]
                labels = ["Night", "Morning", "Afternoon", "Evening"]
            elif col == "avg_pressure":
                bins = 4  # Quartiles
                labels = None
            else:
                bins = 4
                labels = None
            
            try:
                df[f"{col}_bin"] = pd.cut(df[col], bins=bins, labels=labels)
            except:
                continue
            
            for group_name, group_df in df.groupby(f"{col}_bin"):
                if len(group_df) < 10:
                    continue
                
                metrics = self.compute_basic_metrics(
                    group_df[y_true_col].values,
                    group_df[y_pred_col].values
                )
                metrics["stratum"] = col
                metrics["group"] = str(group_name)
                metrics["n"] = len(group_df)
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot reliability diagram for uncertainty calibration.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        calib = self.compute_uncertainty_calibration(y_true, y_pred, y_std)
        
        if "error" not in calib:
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax1.scatter(calib["bin_confidences"], calib["bin_accuracies"], 
                       s=[c/10 for c in calib["bin_counts"]], alpha=0.6)
            ax1.set_xlabel('Expected Confidence')
            ax1.set_ylabel('Observed Accuracy')
            ax1.set_title(f'Reliability Diagram (ECE={calib["ece"]:.3f})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Error vs Uncertainty
        errors = np.abs(y_true - y_pred)
        ax2.scatter(y_std, errors, alpha=0.3, s=5)
        ax2.plot([0, np.max(y_std)], [0, np.max(y_std)], 'r--', label='1:1')
        ax2.plot([0, np.max(y_std)], [0, 2*np.max(y_std)], 'r:', label='2:1')
        ax2.set_xlabel('Predicted Uncertainty (std)')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Error vs Predicted Uncertainty')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "figures", "reliability_diagram.png"), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_distribution(
        self,
        errors_dict: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """
        Plot error distribution comparison across methods.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(errors_dict)))
        
        for idx, (name, errors) in enumerate(errors_dict.items()):
            errors = errors[np.isfinite(errors)]
            
            # Histogram
            axes[0].hist(errors, bins=50, alpha=0.5, label=name, color=colors[idx], density=True)
            
            # CDF
            sorted_errors = np.sort(np.abs(errors))
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            axes[1].plot(sorted_errors, cdf, label=name, color=colors[idx], linewidth=2)
            
            # Box plot data
            axes[2].boxplot(errors, positions=[idx], widths=0.6, 
                           patch_artist=True, boxprops=dict(facecolor=colors[idx], alpha=0.5))
        
        axes[0].set_xlabel('Error (m)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Absolute Error (m)')
        axes[1].set_ylabel('CDF')
        axes[1].set_title('Cumulative Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xticks(range(len(errors_dict)))
        axes[2].set_xticklabels(errors_dict.keys(), rotation=45, ha='right')
        axes[2].set_ylabel('Error (m)')
        axes[2].set_title('Error Box Plots')
        axes[2].grid(True, alpha=0.3)
        
        # Summary table
        axes[3].axis('off')
        table_data = []
        for name, errors in errors_dict.items():
            errors = errors[np.isfinite(errors)]
            table_data.append([
                name,
                f"{np.sqrt(np.mean(errors**2)):.3f}",
                f"{np.mean(np.abs(errors)):.3f}",
                f"{np.percentile(np.abs(errors), 95):.3f}"
            ])
        
        table = axes[3].table(
            cellText=table_data,
            colLabels=['Method', 'RMSE', 'MAE', 'P95'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[3].set_title('Performance Summary', pad=20)
        
        # Remove unused subplots
        for idx in [4, 5]:
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "figures", "error_comparison.png"), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(
        self,
        results_dict: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive comparison report.
        """
        rows = []
        for method_name, metrics in results_dict.items():
            row = {"Method": method_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns
        cols = ["Method", "rmse", "mae", "mape", "bias", "std", 
                "p50", "p75", "p90", "p95", "r2", "n_samples"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        # Save to CSV
        if save_path is None:
            save_path = os.path.join(self.output_dir, "tables", "comparison_results.csv")
        df.to_csv(save_path, index=False, float_format='%.4f')
        
        # Also save as LaTeX table for paper
        latex_path = save_path.replace('.csv', '.tex')
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, float_format='%.3f'))
        
        return df


def print_metrics_table(metrics: Dict[str, float], title: str = "Results"):
    """
    Pretty print metrics to console.
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    key_metrics = ["rmse", "mae", "mape", "bias", "std", "p95", "r2"]
    for key in key_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"{key.upper():>15}: {value:>10.4f}")
            else:
                print(f"{key.upper():>15}: {value}")
    
    print(f"{'='*60}\n")
