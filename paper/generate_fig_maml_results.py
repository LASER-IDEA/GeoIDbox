"""
Generate Figure: MAML Few-Shot Adaptation Results

This script creates visualizations for MAML meta-learning results:
- Few-shot adaptation curves
- Comparison with baselines
- Per-sensor analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150

def load_results(path='experiments/maml_v2/few_shot_results.json'):
    """Load few-shot results"""
    with open(path, 'r') as f:
        results = json.load(f)
    return results

def plot_few_shot_adaptation():
    """Generate few-shot adaptation curve"""
    
    results = load_results()
    
    # Extract data
    sensors = list(results.keys())
    k_shots = [4, 8, 16, 32, 64]
    
    # Calculate statistics
    mean_maes = []
    std_maes = []
    
    for k in k_shots:
        maes = [results[s][str(k)]['mean'] for s in sensors if str(k) in results[s]]
        mean_maes.append(np.mean(maes))
        std_maes.append(np.std(maes))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot adaptation curve
    ax.errorbar(k_shots, mean_maes, yerr=std_maes, 
                marker='o', markersize=10, linewidth=2.5,
                capsize=5, capthick=2, color='#2E86AB',
                label='MAML (Ours)')
    
    # Add individual sensor curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(sensors)))
    for i, sensor in enumerate(sensors):
        sensor_maes = [results[sensor][str(k)]['mean'] for k in k_shots if str(k) in results[sensor]]
        ax.plot(k_shots, sensor_maes, '--', alpha=0.4, color=colors[i], linewidth=1.5,
                label=f'Sensor {sensor}' if i < 3 else '')
    
    # Add horizontal lines for baselines
    ax.axhline(y=3.79, color='#E94F37', linestyle='--', linewidth=2, 
               label='Full Training (3.79m, 115k samples)')
    ax.axhline(y=30, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
               label='Physics Baseline (~30m)')
    
    # Styling
    ax.set_xlabel('Number of Samples (K-shot)', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (m)', fontweight='bold')
    ax.set_title('MAML: Few-Shot Adaptation to New Sensors\n(Real Data - 7 Sensors)', 
                 fontweight='bold', pad=15)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_shots)
    ax.set_xticklabels([str(k) for k in k_shots])
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    
    ax.set_ylim(0, 35)
    
    # Add annotations
    ax.annotate(f'16-shot: {mean_maes[2]:.2f}m', 
                xy=(16, mean_maes[2]), xytext=(25, mean_maes[2]+5),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5),
                fontsize=11, fontweight='bold', color='#2E86AB')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig('paper/figures/fig_maml_fewshot.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('paper/figures/fig_maml_fewshot.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: paper/figures/fig_maml_fewshot.png")
    
    plt.close()


def plot_per_sensor_comparison():
    """Compare few-shot performance across sensors"""
    
    results = load_results()
    sensors = list(results.keys())
    k_shots = [4, 8, 16, 32, 64]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Heatmap of MAE per sensor per k-shot
    mae_matrix = []
    for sensor in sensors:
        row = [results[sensor][str(k)]['mean'] for k in k_shots]
        mae_matrix.append(row)
    
    mae_matrix = np.array(mae_matrix)
    
    im = axes[0].imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=30)
    
    axes[0].set_xticks(range(len(k_shots)))
    axes[0].set_xticklabels([f'{k}-shot' for k in k_shots])
    axes[0].set_yticks(range(len(sensors)))
    axes[0].set_yticklabels([f'Sensor {s}' for s in sensors])
    
    # Add text annotations
    for i in range(len(sensors)):
        for j in range(len(k_shots)):
            text = axes[0].text(j, i, f'{mae_matrix[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    axes[0].set_title('MAE per Sensor (m)', fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='MAE (m)')
    
    # Right plot: Improvement from 4-shot to 64-shot
    improvement = mae_matrix[:, 0] - mae_matrix[:, -1]
    colors = ['green' if x > 0 else 'red' for x in improvement]
    
    bars = axes[1].barh(range(len(sensors)), improvement, color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(sensors)))
    axes[1].set_yticklabels([f'Sensor {s}' for s in sensors])
    axes[1].set_xlabel('MAE Improvement (m)', fontweight='bold')
    axes[1].set_title('Improvement: 4-shot → 64-shot', fontweight='bold')
    axes[1].axvline(x=0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvement)):
        axes[1].text(val + 0.5 if val > 0 else val - 0.5, i, f'{val:.1f}m',
                    va='center', ha='left' if val > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig_maml_per_sensor.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: paper/figures/fig_maml_per_sensor.png")
    
    plt.close()


def generate_summary_table():
    """Generate LaTeX table for paper"""
    
    results = load_results()
    sensors = list(results.keys())
    k_shots = [4, 8, 16, 32, 64]
    
    print("\n" + "="*80)
    print("MAML FEW-SHOT ADAPTATION RESULTS")
    print("="*80)
    
    # Table header
    print(f"\n{'Sensor':>12} | {'Altitude':>10} | " + " | ".join([f"{k:>8}-shot" for k in k_shots]))
    print("-" * 90)
    
    # Get altitude info (need to reload data)
    import pandas as pd
    df = pd.read_csv('data/processed/sensor_data_with_real_era5.csv')
    
    for sensor in sensors:
        # Find altitude for this sensor
        sensor_full = [s for s in df['uid'].unique() if s.endswith(sensor)][0]
        altitude = df[df['uid'] == sensor_full]['avg_altitude'].mean()
        
        maes = [results[sensor][str(k)]['mean'] for k in k_shots]
        print(f"{sensor:>12} | {altitude:>10.1f}m | " + " | ".join([f"{m:>8.2f}" for m in maes]))
    
    # Summary row
    print("-" * 90)
    mean_maes = []
    for k in k_shots:
        maes = [results[s][str(k)]['mean'] for s in sensors]
        mean_maes.append(np.mean(maes))
    
    print(f"{'Mean':>12} | {'':>10} | " + " | ".join([f"{m:>8.2f}" for m in mean_maes]))
    
    # LaTeX table
    print("\n" + "="*80)
    print("LaTeX TABLE")
    print("="*80)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{MAML Few-Shot Adaptation Results (MAE in meters)}")
    print("\\label{tab:maml}")
    print("\\begin{tabular}{l|c|" + "c"*len(k_shots) + "}")
    print("\\hline")
    print("Sensor & Altitude (m) & " + " & ".join([f"{k}-shot" for k in k_shots]) + " \\\\")
    print("\\hline")
    
    for sensor in sensors:
        sensor_full = [s for s in df['uid'].unique() if s.endswith(sensor)][0]
        altitude = df[df['uid'] == sensor_full]['avg_altitude'].mean()
        maes = [results[sensor][str(k)]['mean'] for k in k_shots]
        print(f"{sensor} & {altitude:.1f} & " + " & ".join([f"{m:.2f}" for m in maes]) + " \\\\")
    
    print("\\hline")
    print("\\textbf{Mean} & -- & " + " & ".join([f"\\textbf{{{m:.2f}}}" for m in mean_maes]) + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    """Generate all MAML figures"""
    print("="*60)
    print("GENERATING MAML VISUALIZATIONS")
    print("="*60)
    
    plot_few_shot_adaptation()
    plot_per_sensor_comparison()
    generate_summary_table()
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
