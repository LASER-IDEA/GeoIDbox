"""
Generate publication-quality figures for IEEE TIM paper.

Style reference: run_height_field_osm.py
- Background: #E8EFF6 (axes), white (figure)
- Land fill:  #F5F1EB
- Grid:       dotted, #B0BFCC
- Font:       Times New Roman / DejaVu Serif, serif
- Colors:     RdYlBu_r-inspired discrete palette
- 300 DPI, PDF + PNG
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import os

# ── Style (matches run_height_field_osm.py) ─────────────────────────────────
plt.rcParams.update({
    'font.family':  'serif',
    'font.serif':   ['Times New Roman', 'DejaVu Serif'],
    'font.size':    12,
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
})

# Discrete palette derived from RdYlBu_r + accent colours used in OSM map
PALETTE = {
    'deep_blue':   '#313695',   # RdYlBu dark blue  – best / lowest error
    'mid_blue':    '#4393C3',   # RdYlBu blue
    'light_blue':  '#74ADD1',   # RdYlBu light blue
    'yellow':      '#FEE090',   # RdYlBu yellow
    'orange':      '#F46D43',   # RdYlBu orange
    'red':         '#D62728',   # accent red (same as OSM sensor marker)
    'dark_red':    '#A50026',   # RdYlBu dark red   – worst / highest error
    'bg_axes':     '#E8EFF6',   # axes background (same as OSM figure)
    'bg_land':     '#F5F1EB',   # land fill
    'grid':        '#B0BFCC',   # grid lines
    'text':        '#1A1A2E',   # dark text
    'light_text':  '#6B7280',   # secondary text
}

OUT_DIR = 'experiments/figures'


def setup_figure(figsize=(10, 6), dpi=300):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor(PALETTE['bg_axes'])
    ax.grid(True, linestyle=':', linewidth=0.5, color=PALETTE['grid'],
            alpha=0.9, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(PALETTE['grid'])
        spine.set_linewidth(0.7)
    return fig, ax


def save_figure(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# Figure 1 – Baseline Comparison
# ════════════════════════════════════════════════════════════════════════
def plot_baseline_comparison(
        results_csv='experiments/01_baseline_comparisons/results.csv'):
    print("\nGenerating Figure 1: Baseline Comparison...")
    if not os.path.exists(results_csv):
        print(f"  Not found: {results_csv}"); return

    df = pd.read_csv(results_csv)
    methods_raw  = ['IDW',        'Kriging',     'RF',          'XGB'        ]
    labels       = ['IDW',        'Kriging',     'Random\nForest','XGBoost'  ]
    colors_bar   = [PALETTE['mid_blue'], PALETTE['light_blue'],
                    PALETTE['orange'],   PALETTE['dark_red']]

    means, stds = [], []
    for m in methods_raw:
        col = f'{m}_MAE'
        means.append(df[col].mean())
        stds.append(df[col].std())

    # Append Physics Baseline and PINF
    labels    += ['Physics\nBaseline', 'PINF\n(Ours)']
    means     += [36.96,  3.55]
    stds      += [4.04,   1.23]
    colors_bar += [PALETTE['yellow'], PALETTE['deep_blue']]

    fig, ax = setup_figure(figsize=(11, 6))
    x = np.arange(len(labels))

    bars = ax.bar(x, means, width=0.55, yerr=stds, capsize=6,
                  color=colors_bar, alpha=0.88,
                  edgecolor='white', linewidth=1.2,
                  error_kw=dict(ecolor=PALETTE['text'], lw=1.4, capthick=1.4),
                  zorder=3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                mean + std + 0.8,
                f'{mean:.2f}',
                ha='center', va='bottom', fontsize=11.5, fontweight='bold',
                color=PALETTE['text'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Highlight PINF bar with bracket
    pinf_x = len(labels) - 1
    # ax.annotate('', xy=(pinf_x, means[-1] + stds[-1] + 4),
    #             xytext=(0, means[0] + stds[0] + 4),
    #             arrowprops=dict(arrowstyle='<->', color=PALETTE['deep_blue'],
    #                             lw=1.8))
    # ax.text((pinf_x) / 2., max(means) * 0.82,
    #         f'−90.4 % vs Physics', ha='center', fontsize=11,
    #         color=PALETTE['deep_blue'], fontweight='bold',
    #         path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_ylabel('Mean Absolute Error (m)', fontsize=14, fontweight='bold',
                  labelpad=8)
    ax.set_xlabel('Method', fontsize=14, fontweight='bold', labelpad=8)
    ax.set_title(
        'Baseline Comparison: Urban Altitude Estimation\n'
        '(8-Fold Leave-One-Sensor-Out Cross-Validation)',
        fontsize=14, fontweight='bold', pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, max(means) * 1.35)

    from matplotlib.patches import Patch
    # ax.legend(handles=[Patch(facecolor=PALETTE['deep_blue'], label='Proposed PINF')],
    #           loc='upper right', fontsize=11, framealpha=0.92,
    #           edgecolor=PALETTE['grid'])

    save_figure(fig, '01_baseline_comparison')
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# Figure 2 – Ablation Study
# ════════════════════════════════════════════════════════════════════════
def plot_ablation_study(
        results_csv='experiments/02_ablation_studies/results.csv'):
    print("\nGenerating Figure 2: Ablation Study...")
    if not os.path.exists(results_csv):
        print(f"  Not found: {results_csv}"); return

    df = pd.read_csv(results_csv)
    label_map = {
        'Pure Physics Baseline':
            'Physics\nBaseline',
        'Setup A: Base NN (Direct Height)':
            'A: Direct Δh\n(No P_bias, No CL)',
        'Setup C: Direct Height + Curriculum (No P_bias)':
            'C: Direct Δh\n+Curriculum\n(No P_bias)',
        'Setup B: Bias-Aware Formulation (δP)':
            'B: P_bias-Aware\n(No CL)',
        'Setup D: Bias-Aware + Curriculum Learning':
            'D: Full Model\n(P_bias + CL)',
    }
    setups = [label_map.get(r, r) for r in df['setup']]
    values = df['mae'].tolist()

    # Waterfall palette: grey (baseline) → warm colours → deep blue (best)
    colors_bar = [
        PALETTE['light_text'],
        PALETTE['orange'],
        PALETTE['yellow'],
        PALETTE['mid_blue'],
        PALETTE['deep_blue'],
    ][:len(values)]

    fig, ax = setup_figure(figsize=(14, 7))
    x = np.arange(len(setups))

    bars = ax.bar(x, values, width=0.58, color=colors_bar,
                  alpha=0.88, edgecolor='white', linewidth=1.8, zorder=3)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2., val + 0.4,
                f'{val:.2f} m',
                ha='center', va='bottom', fontsize=12.5, fontweight='bold',
                color=PALETTE['text'],
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
        if i > 0:
            drop = values[i - 1] - values[i]
            if drop > 0:
                mid_y = values[i] + drop / 2
                ax.annotate('',
                            xy=(i - 0.28, values[i]),
                            xytext=(i - 0.28, values[i - 1]),
                            arrowprops=dict(arrowstyle='->', lw=1.8,
                                            color=PALETTE['deep_blue']))
                ax.text(i - 0.45, mid_y,
                        f'−{drop:.2f}',
                        fontsize=9.5, color=PALETTE['deep_blue'],
                        fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground='white')])

    ax.set_ylabel('Mean Absolute Error (m)', fontsize=14, fontweight='bold',
                  labelpad=8)
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold', labelpad=8)
    ax.set_title(
        'Ablation Study: Component Contribution Analysis\n'
        '(8-Fold LOSO Validation)',
        fontsize=14, fontweight='bold', pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(setups, fontsize=11)
    ax.set_ylim(0, max(values) * 1.18)

    save_figure(fig, '02_ablation_study')
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# Figure 3 – Uncertainty Quantification (binned bar)
# ════════════════════════════════════════════════════════════════════════
def plot_uncertainty_vs_error(
        uq_csv='experiments/03_uncertainty_quantification/uq_data.csv'):
    print("\nGenerating Figure 3: Uncertainty vs Error...")
    if not os.path.exists(uq_csv):
        print(f"  Not found: {uq_csv}"); return

    df = pd.read_csv(uq_csv)
    unc = df['uncertainty'].values
    err = df['abs_error'].values
    corr = np.corrcoef(unc, err)[0, 1]

    n_bins = 5
    edges = np.percentile(unc, np.linspace(0, 100, n_bins + 1))
    bin_labels, bin_errs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (unc >= edges[i]) & (unc <= edges[i + 1])
        if mask.sum() > 0:
            bin_labels.append(f'Q{i+1}')
            bin_errs.append(float(np.mean(err[mask])))
            bin_counts.append(int(mask.sum()))

    # Gradient: deep_blue (lowest) → dark_red (highest)
    grad = [PALETTE['deep_blue'], PALETTE['mid_blue'], PALETTE['yellow'],
            PALETTE['orange'], PALETTE['dark_red']]
    bar_colors = grad[:len(bin_labels)]

    fig, ax = setup_figure(figsize=(10, 6))
    bars = ax.bar(range(len(bin_labels)), bin_errs,
                  color=bar_colors, alpha=0.88,
                  edgecolor='white', linewidth=1.5, zorder=3)

    for bar, be, bc in zip(bars, bin_errs, bin_counts):
        ax.text(bar.get_x() + bar.get_width() / 2., be + 0.06,
                f'{be:.2f} m\n(n={bc:,})',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=PALETTE['text'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_xlabel('Uncertainty Quintile  (Q1 = lowest σ, Q5 = highest σ)',
                  fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel('Mean Absolute Error (m)', fontsize=13, fontweight='bold',
                  labelpad=8)
    ax.set_title(
        'Uncertainty Quantification: Prediction Error by Uncertainty Bin\n'
        '(MC Dropout, 30 Samples, Fold 0)',
        fontsize=14, fontweight='bold', pad=16)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=12)
    ax.set_ylim(0, max(bin_errs) * 1.35)

    info = (f'σ–|e| Correlation: {corr:.3f}\n'
            f'Mean |error|: {np.mean(err):.2f} m\n'
            f'Mean σ: {np.mean(unc)*1e6:.1f} µm')
    ax.text(0.97, 0.97, info, transform=ax.transAxes,
            fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec=PALETTE['grid'], alpha=0.92))

    save_figure(fig, '03_uncertainty_vs_error')
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# Figure 4 – Spatial Error Distribution
# ════════════════════════════════════════════════════════════════════════
def plot_spatial_uncertainty_map(
        uq_csv='experiments/03_uncertainty_quantification/uq_data.csv'):
    print("\nGenerating Figure 4: Spatial Error Distribution...")
    if not os.path.exists(uq_csv):
        print(f"  Not found: {uq_csv}"); return

    df = pd.read_csv(uq_csv)
    # Clip extreme outliers for colour scale
    vmax = np.percentile(df['abs_error'], 95)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor(PALETTE['bg_axes'])

    sc = ax.scatter(df['longitude'], df['latitude'],
                    c=df['abs_error'], cmap='RdYlBu_r',
                    vmin=0, vmax=vmax,
                    s=8, alpha=0.65, edgecolors='none', zorder=3)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01, shrink=0.82)
    cbar.set_label('Absolute Error (m)', fontsize=13, fontweight='bold',
                   labelpad=10)
    cbar.ax.tick_params(labelsize=11)

    ax.grid(True, linestyle=':', lw=0.5, color=PALETTE['grid'], alpha=0.9,
            zorder=1)
    for spine in ax.spines.values():
        spine.set_color(PALETTE['grid']); spine.set_linewidth(0.7)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude (°E)', fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel('Latitude (°N)',  fontsize=13, fontweight='bold', labelpad=8)

    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f°'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f°'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right',
             fontsize=10)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=10)

    ax.set_title(
        'Spatial Distribution of Prediction Error (Fold 0)\n'
        '(PINF · Held-out Sensor …A64197)',
        fontsize=14, fontweight='bold', pad=16)

    info = (f'Mean |error|: {df["abs_error"].mean():.2f} m\n'
            f'Median |error|: {df["abs_error"].median():.2f} m\n'
            f'95th pct: {vmax:.1f} m')
    ax.text(0.03, 0.97, info, transform=ax.transAxes,
            fontsize=11, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec=PALETTE['grid'], alpha=0.92))

    plt.tight_layout()
    save_figure(fig, '04_spatial_uncertainty')
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# Figure 5 – Per-Fold LOSO Results
# ════════════════════════════════════════════════════════════════════════
def plot_per_fold_results(
        csv_path='results/loso_summary.csv'):
    print("\nGenerating Figure 5: Per-Fold Results...")
    if not os.path.exists(csv_path):
        print(f"  Not found: {csv_path}"); return

    df = pd.read_csv(csv_path)
    maes  = df['mae'].tolist()
    rmses = df['rmse'].tolist()
    folds = np.arange(len(df))
    # Short sensor ID suffix for x-axis labels
    sensor_labels = [f'Fold {i}\n(S{i+1})' for i in range(len(df))]

    min_i = int(np.argmin(maes))
    max_i = int(np.argmax(maes))
    bar_colors = [PALETTE['mid_blue']] * len(maes)
    bar_colors[min_i] = PALETTE['deep_blue']   # best
    bar_colors[max_i] = PALETTE['dark_red']    # worst

    fig, ax = setup_figure(figsize=(12, 6))

    bars = ax.bar(folds - 0.18, maes, width=0.34,
                  color=bar_colors, alpha=0.88,
                  edgecolor='white', linewidth=1.4,
                  zorder=3, label='MAE (m)')
    ax.bar(folds + 0.18, rmses, width=0.34,
           color=[c + '88' for c in bar_colors],   # semi-transparent twin
           alpha=0.65, edgecolor='white', linewidth=1.2,
           zorder=3, label='RMSE (m)',
           # Use a uniform light colour for RMSE bars instead of hex tricks
           )
    # Redo RMSE bars with clean colour
    ax.containers[-1].remove()
    rmse_colors = [PALETTE['light_blue']] * len(rmses)
    ax.bar(folds + 0.18, rmses, width=0.34,
           color=rmse_colors, alpha=0.70,
           edgecolor='white', linewidth=1.2,
           zorder=3, label='RMSE (m)')

    mean_mae = float(np.mean(maes))
    ax.axhline(mean_mae, color=PALETTE['deep_blue'], linestyle='--',
               linewidth=1.8, zorder=4,
               label=f'Mean MAE: {mean_mae:.2f} m')

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2., mae + 0.1,
                f'{mae:.2f}',
                ha='center', va='bottom', fontsize=10.5, fontweight='bold',
                color=PALETTE['text'],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_xlabel('Fold (Held-out Sensor)', fontsize=13, fontweight='bold',
                  labelpad=8)
    ax.set_ylabel('Error (m)', fontsize=13, fontweight='bold', labelpad=8)
    ax.set_title(
        'Per-Fold LOSO Cross-Validation Results\n'
        '(PINF · Setup D: P_bias + Curriculum)',
        fontsize=14, fontweight='bold', pad=16)
    ax.set_xticks(folds)
    ax.set_xticklabels(sensor_labels, fontsize=10.5)
    ax.set_ylim(0, max(rmses) * 1.28)
    ax.legend(fontsize=11, framealpha=0.92, edgecolor=PALETTE['grid'])

    save_figure(fig, '05_per_fold_results')
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
def generate_all_figures():
    print("=" * 68)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("Style: Times New Roman · RdYlBu palette · 300 DPI · PDF + PNG")
    print("=" * 68)

    plot_baseline_comparison()
    plot_ablation_study()
    plot_uncertainty_vs_error()
    plot_spatial_uncertainty_map()
    plot_per_fold_results()

    print("\n" + "=" * 68)
    print("ALL FIGURES GENERATED  →  experiments/figures/")
    print("=" * 68)
    print("\nNote: Figure 06 (height field on map) is generated separately:")
    print("  python experiments/04_uncertainty_map/run_height_field_osm.py")


if __name__ == "__main__":
    generate_all_figures()
