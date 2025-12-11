"""
Deep Sensor Analysis Script - Three Core Dimensions
=====================================================
1. Vertical Precision Tiering
2. Environmental Coupling Analysis
3. Diurnal Cycle Analysis (Multipath Detection)

Output: Charts + Analysis Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
import os
from datetime import datetime

# ========== Style Configuration ==========
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.unicode_minus': False,
    'figure.figsize': (14, 8),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# Create output directory
OUTPUT_DIR = 'data/reports/deep_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme - Tech aesthetic
COLORS = {
    'tier1': '#00D4AA',      # Teal - High precision
    'tier2': '#FFB800',      # Amber - Standard
    'tier3': '#FF4757',      # Coral - Warning
    'primary': '#4A90D9',    # Primary blue
    'secondary': '#7B68EE',  # Purple
    'accent': '#FF6B6B',     # Accent
    'bg_dark': '#1A1A2E',    # Dark background
    'grid': '#2D2D44'        # Grid color
}


def load_data(file_path):
    """Load and preprocess data"""
    print(f"Loading data: {file_path}")
    df = pd.read_csv(file_path)
    df['processed_time'] = pd.to_datetime(df['processed_time'])
    df['hour'] = df['processed_time'].dt.hour
    df['date'] = df['processed_time'].dt.date
    df['day_of_week'] = df['processed_time'].dt.dayofweek

    print(f"Loaded: {df['uid'].nunique()} devices, {len(df):,} records")
    print(f"   Time range: {df['processed_time'].min()} ~ {df['processed_time'].max()}")
    return df


# =============================================================================
# Dimension 1: Vertical Precision Tiering
# =============================================================================
def analyze_vertical_precision(df):
    """
    Vertical Precision Tiering Analysis
    - Calculate MSL altitude standard deviation per device
    - Tiers: Tier 1 (<1.5m), Tier 2 (<5m), Tier 3 (>5m)
    """
    print("\n" + "="*70)
    print("Dimension 1: Vertical Precision Tiering")
    print("="*70)

    # Calculate statistics per device
    device_stats = df.groupby('uid').agg({
        'avg_altitude': ['std', 'mean', 'min', 'max', 'count'],
        'avg_satellites': ['mean', 'std'],
        'avg_hdop': ['mean', 'std'],
        'avg_pressure': ['std'],
        'avg_temperature': ['std']
    })
    device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns.values]
    device_stats = device_stats.rename(columns={
        'avg_altitude_std': 'alt_std',
        'avg_altitude_mean': 'alt_mean',
        'avg_altitude_min': 'alt_min',
        'avg_altitude_max': 'alt_max',
        'avg_altitude_count': 'record_count',
        'avg_satellites_mean': 'sat_mean',
        'avg_satellites_std': 'sat_std',
        'avg_hdop_mean': 'hdop_mean',
        'avg_hdop_std': 'hdop_std',
        'avg_pressure_std': 'pressure_std',
        'avg_temperature_std': 'temp_std'
    })

    # Calculate altitude range
    device_stats['alt_range'] = device_stats['alt_max'] - device_stats['alt_min']

    # Tier definitions
    def assign_tier(std):
        if std <= 1.5:
            return 'Tier 1 (High)'
        elif std <= 5.0:
            return 'Tier 2 (Standard)'
        else:
            return 'Tier 3 (Warning)'

    device_stats['tier'] = device_stats['alt_std'].apply(assign_tier)
    device_stats = device_stats.sort_values('alt_std')

    # Count devices per tier
    tier_counts = device_stats['tier'].value_counts()

    print("\nPrecision Tier Statistics:")
    print("-" * 50)
    for tier in ['Tier 1 (High)', 'Tier 2 (Standard)', 'Tier 3 (Warning)']:
        count = tier_counts.get(tier, 0)
        pct = count / len(device_stats) * 100
        print(f"   {tier}: {count} devices ({pct:.1f}%)")

    # Detailed statistics
    print("\nDetailed Statistics by Tier:")
    print("-" * 50)
    tier_summary = device_stats.groupby('tier').agg({
        'alt_std': ['mean', 'min', 'max'],
        'sat_mean': 'mean',
        'hdop_mean': 'mean',
        'record_count': 'sum'
    }).round(3)
    print(tier_summary)

    # Identify best and worst devices
    print("\nGold Standard Devices (Top 5 - Most Stable):")
    print(device_stats.head()[['alt_std', 'alt_mean', 'sat_mean', 'hdop_mean', 'record_count', 'tier']])

    print("\nWarning Devices (Least Stable):")
    tier3_devices = device_stats[device_stats['tier'] == 'Tier 3 (Warning)']
    if len(tier3_devices) > 0:
        print(tier3_devices[['alt_std', 'alt_range', 'sat_mean', 'hdop_mean', 'tier']])
    else:
        print("   No warning-level devices")

    # ===== Visualization =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Tier Distribution (Bar Chart)
    ax1 = axes[0, 0]
    tier_order = ['Tier 1 (High)', 'Tier 2 (Standard)', 'Tier 3 (Warning)']
    colors_tier = [COLORS['tier1'], COLORS['tier2'], COLORS['tier3']]
    counts = [tier_counts.get(t, 0) for t in tier_order]
    bars = ax1.bar(tier_order, counts, color=colors_tier, edgecolor='white', linewidth=2)
    ax1.set_title('Device Precision Tier Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Devices')
    for bar, count in zip(bars, counts):
        ax1.annotate(f'{count}\n({count/len(device_stats)*100:.1f}%)',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. STD Distribution Histogram
    ax2 = axes[0, 1]
    ax2.hist(device_stats['alt_std'], bins=30, color=COLORS['primary'],
             edgecolor='white', alpha=0.8)
    ax2.axvline(1.5, color=COLORS['tier1'], linestyle='--', linewidth=2, label='Tier 1 Threshold (1.5m)')
    ax2.axvline(5.0, color=COLORS['tier3'], linestyle='--', linewidth=2, label='Tier 2 Threshold (5.0m)')
    ax2.set_title('Altitude STD Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Altitude STD (m)')
    ax2.set_ylabel('Number of Devices')
    ax2.legend()

    # 3. STD vs Satellite Count Scatter
    ax3 = axes[1, 0]
    tier_colors = device_stats['tier'].map({
        'Tier 1 (High)': COLORS['tier1'],
        'Tier 2 (Standard)': COLORS['tier2'],
        'Tier 3 (Warning)': COLORS['tier3']
    })
    scatter = ax3.scatter(device_stats['sat_mean'], device_stats['alt_std'],
                          c=tier_colors, s=device_stats['record_count']/50 + 30,
                          alpha=0.7, edgecolors='white', linewidth=0.5)
    ax3.axhline(1.5, color=COLORS['tier1'], linestyle='--', alpha=0.7)
    ax3.axhline(5.0, color=COLORS['tier3'], linestyle='--', alpha=0.7)
    ax3.set_title('Vertical Precision vs Avg Satellites', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Average Satellite Count')
    ax3.set_ylabel('Altitude STD (m) - Lower is Better')
    ax3.set_ylim(0, device_stats['alt_std'].max() * 1.1)

    # 4. STD vs HDOP Scatter
    ax4 = axes[1, 1]
    scatter2 = ax4.scatter(device_stats['hdop_mean'], device_stats['alt_std'],
                           c=tier_colors, s=80, alpha=0.7, edgecolors='white')
    ax4.axhline(1.5, color=COLORS['tier1'], linestyle='--', alpha=0.7, label='Tier 1 Threshold')
    ax4.axhline(5.0, color=COLORS['tier3'], linestyle='--', alpha=0.7, label='Tier 2 Threshold')
    ax4.set_title('Vertical Precision vs Avg HDOP', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Average HDOP (Dilution of Precision)')
    ax4.set_ylabel('Altitude STD (m)')
    ax4.legend()

    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['tier1'], label='Tier 1 (High <=1.5m)'),
        Patch(facecolor=COLORS['tier2'], label='Tier 2 (Standard <=5.0m)'),
        Patch(facecolor=COLORS['tier3'], label='Tier 3 (Warning >5.0m)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.02), fontsize=11)

    plt.suptitle('Vertical Precision Tiering Analysis',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_vertical_precision_tiering.png')
    plt.show()

    # Save device statistics
    device_stats.to_csv(f'{OUTPUT_DIR}/device_tier_stats.csv')
    print(f"\nDevice stats saved to: {OUTPUT_DIR}/device_tier_stats.csv")

    return device_stats


# =============================================================================
# Dimension 2: Environmental Coupling Analysis
# =============================================================================
def analyze_environmental_coupling(df, device_stats):
    """
    Environmental Coupling Analysis
    - Pressure vs Altitude: Physical atmosphere consistency check
    - Temperature/Humidity vs Altitude: Thermal drift/wet delay detection
    """
    print("\n" + "="*70)
    print("Dimension 2: Environmental Coupling Analysis")
    print("="*70)

    # Select device with most data from Tier 1 as golden reference
    tier1_devices = device_stats[device_stats['tier'] == 'Tier 1 (High)'].index
    if len(tier1_devices) > 0:
        record_counts = df[df['uid'].isin(tier1_devices)].groupby('uid').size()
        golden_uid = record_counts.idxmax()
    else:
        # No Tier 1, use device with most data
        golden_uid = df['uid'].value_counts().idxmax()

    golden_data = df[df['uid'] == golden_uid].copy()
    print(f"\nSelected reference device for deep analysis: {golden_uid[-8:]}")
    print(f"   Records: {len(golden_data):,}, STD: {device_stats.loc[golden_uid, 'alt_std']:.3f}m")

    # ===== Correlation Analysis =====
    env_cols = ['avg_altitude', 'avg_pressure', 'avg_temperature', 'avg_humidity']

    # Global correlation
    print("\nGlobal Environmental Factor Correlation (all devices):")
    global_corr = df[env_cols].corr()
    print(global_corr.round(4))

    # Single device correlation (cleaner, no location differences)
    print(f"\nReference Device Environmental Factor Correlation:")
    device_corr = golden_data[env_cols].corr()
    print(device_corr.round(4))

    # ===== Key Metrics =====
    alt_pressure_corr = device_corr.loc['avg_altitude', 'avg_pressure']
    alt_temp_corr = device_corr.loc['avg_altitude', 'avg_temperature']
    alt_humid_corr = device_corr.loc['avg_altitude', 'avg_humidity']

    print("\nKey Coupling Coefficients:")
    print("-" * 50)
    print(f"   Altitude vs Pressure: {alt_pressure_corr:.4f}")
    print(f"   Altitude vs Temperature: {alt_temp_corr:.4f}")
    print(f"   Altitude vs Humidity: {alt_humid_corr:.4f}")

    # Physical consistency interpretation
    print("\nPhysical Consistency Interpretation:")
    if alt_pressure_corr < -0.3:
        print("   [OK] Pressure-altitude shows significant negative correlation (follows barometric formula)")
    elif alt_pressure_corr < 0:
        print("   [WARN] Pressure-altitude shows weak negative correlation")
    else:
        print("   [ERROR] Pressure-altitude shows positive or no correlation - ANOMALY")

    # ===== Visualization =====
    fig = plt.figure(figsize=(18, 14))

    # 1. Correlation Heatmap
    ax1 = fig.add_subplot(2, 3, 1)
    sns.heatmap(device_corr, annot=True, cmap='RdBu_r', vmin=-1, vmax=1,
                fmt='.3f', square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax1.set_title(f'Reference Device Correlation Matrix\n(UID: ...{golden_uid[-6:]})', fontsize=12, fontweight='bold')

    # 2. Pressure vs Altitude Scatter (KEY!)
    ax2 = fig.add_subplot(2, 3, 2)
    scatter = ax2.scatter(golden_data['avg_pressure'], golden_data['avg_altitude'],
                          c=golden_data['hour'], cmap='twilight',
                          s=15, alpha=0.6)
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        golden_data['avg_pressure'], golden_data['avg_altitude']
    )
    x_line = np.linspace(golden_data['avg_pressure'].min(), golden_data['avg_pressure'].max(), 100)
    ax2.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
             label=f'Linear Fit: R2={r_value**2:.4f}')
    ax2.set_xlabel('Pressure (Pa)')
    ax2.set_ylabel('GNSS Altitude (m)')
    ax2.set_title('Pressure vs GNSS Altitude\n(Physical Consistency Check)', fontsize=12, fontweight='bold')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Hour', shrink=0.8)

    # 3. Temperature vs Altitude Scatter (Thermal Drift)
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(golden_data['avg_temperature'], golden_data['avg_altitude'],
                           c=golden_data['hour'], cmap='plasma', s=15, alpha=0.6)
    ax3.set_xlabel('Temperature (C)')
    ax3.set_ylabel('GNSS Altitude (m)')
    ax3.set_title(f'Temperature vs Altitude (Thermal Drift)\nCorr={alt_temp_corr:.4f}',
                  fontsize=12, fontweight='bold')
    plt.colorbar(scatter3, ax=ax3, label='Hour', shrink=0.8)

    # Add trend line
    slope_t, intercept_t, r_t, _, _ = stats.linregress(
        golden_data['avg_temperature'], golden_data['avg_altitude']
    )
    x_t = np.linspace(golden_data['avg_temperature'].min(), golden_data['avg_temperature'].max(), 50)
    ax3.plot(x_t, slope_t * x_t + intercept_t, 'r--', linewidth=2, alpha=0.7)

    # 4. Humidity vs Altitude Scatter (Wet Delay)
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(golden_data['avg_humidity'], golden_data['avg_altitude'],
                           c=golden_data['avg_temperature'], cmap='coolwarm',
                           s=15, alpha=0.6)
    ax4.set_xlabel('Humidity (%)')
    ax4.set_ylabel('GNSS Altitude (m)')
    ax4.set_title(f'Humidity vs Altitude (Wet Delay)\nCorr={alt_humid_corr:.4f}',
                  fontsize=12, fontweight='bold')
    plt.colorbar(scatter4, ax=ax4, label='Temperature (C)', shrink=0.8)

    # 5. Time Series: Altitude vs Pressure (Dual Y-axis)
    ax5 = fig.add_subplot(2, 3, 5)
    # Take last 3 days
    recent_dates = sorted(golden_data['date'].unique())[-3:]
    recent_data = golden_data[golden_data['date'].isin(recent_dates)]

    color1 = COLORS['primary']
    ax5.plot(recent_data['processed_time'], recent_data['avg_altitude'],
             color=color1, alpha=0.8, linewidth=1, label='GNSS Altitude')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('GNSS Altitude (m)', color=color1)
    ax5.tick_params(axis='y', labelcolor=color1)

    ax5_twin = ax5.twinx()
    color2 = COLORS['accent']
    ax5_twin.plot(recent_data['processed_time'], recent_data['avg_pressure'],
                  color=color2, alpha=0.8, linewidth=1, linestyle='--', label='Pressure')
    ax5_twin.set_ylabel('Pressure (Pa)', color=color2)
    ax5_twin.tick_params(axis='y', labelcolor=color2)
    ax5_twin.invert_yaxis()  # Invert for visual sync

    ax5.set_title('Time Series: Altitude vs Pressure (Last 3 Days)', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=30)

    # 6. Multi-device Coupling Strength Comparison
    ax6 = fig.add_subplot(2, 3, 6)

    # Calculate altitude-pressure correlation per device
    device_coupling = []
    for uid in df['uid'].unique():
        uid_data = df[df['uid'] == uid]
        if len(uid_data) > 100:
            corr_val = uid_data['avg_altitude'].corr(uid_data['avg_pressure'])
            tier = device_stats.loc[uid, 'tier'] if uid in device_stats.index else 'Unknown'
            device_coupling.append({
                'uid': uid[-6:],
                'corr': corr_val,
                'tier': tier,
                'count': len(uid_data)
            })

    coupling_df = pd.DataFrame(device_coupling)

    # Sort by correlation and plot
    coupling_df = coupling_df.sort_values('corr')
    tier_colors_map = {
        'Tier 1 (High)': COLORS['tier1'],
        'Tier 2 (Standard)': COLORS['tier2'],
        'Tier 3 (Warning)': COLORS['tier3']
    }
    bar_colors = coupling_df['tier'].map(tier_colors_map).fillna('gray')

    bars = ax6.barh(range(len(coupling_df)), coupling_df['corr'],
                    color=bar_colors, edgecolor='white', height=0.7)
    ax6.set_yticks(range(len(coupling_df)))
    ax6.set_yticklabels(coupling_df['uid'], fontsize=8)
    ax6.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax6.axvline(-0.5, color='green', linestyle='--', alpha=0.5, label='Strong Coupling Threshold')
    ax6.set_xlabel('Altitude-Pressure Correlation')
    ax6.set_title('Device Pressure Coupling Strength\n(Negative = Follows Physics)', fontsize=12, fontweight='bold')
    ax6.legend(loc='lower right')

    plt.suptitle('Environmental Coupling Analysis',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_environmental_coupling.png')
    plt.show()

    # Print coupling analysis conclusion
    print("\nEnvironmental Coupling Analysis Conclusion:")
    print("-" * 50)
    strong_coupling = coupling_df[coupling_df['corr'] < -0.5]
    weak_coupling = coupling_df[coupling_df['corr'] >= -0.3]
    print(f"   Strong pressure coupling devices (r < -0.5): {len(strong_coupling)}")
    print(f"   Weak pressure coupling devices (r >= -0.3): {len(weak_coupling)}")

    # Save coupling analysis data
    coupling_df.to_csv(f'{OUTPUT_DIR}/device_coupling_analysis.csv', index=False)

    return coupling_df, golden_uid


# =============================================================================
# Dimension 3: Diurnal Cycle Analysis (Multipath Detection)
# =============================================================================
def analyze_diurnal_cycle(df, device_stats):
    """
    Diurnal Cycle Analysis - Multipath Effect Detection
    - Aggregate altitude deviation by hour
    - FFT spectrum analysis to detect 24-hour periodicity
    """
    print("\n" + "="*70)
    print("Dimension 3: Diurnal Cycle Analysis (Multipath Detection)")
    print("="*70)

    # Select representative devices: most stable vs least stable
    stable_uid = device_stats.index[0]  # Lowest STD
    unstable_uid = device_stats.index[-1]  # Highest STD

    print(f"\nComparison Analysis Devices:")
    print(f"   Most Stable: ...{stable_uid[-6:]} (STD={device_stats.loc[stable_uid, 'alt_std']:.3f}m)")
    print(f"   Least Stable: ...{unstable_uid[-6:]} (STD={device_stats.loc[unstable_uid, 'alt_std']:.3f}m)")

    # ===== Diurnal Analysis Function =====
    def compute_diurnal_pattern(uid_data):
        """Compute device's diurnal pattern"""
        mean_alt = uid_data['avg_altitude'].mean()
        hourly_stats = uid_data.groupby('hour').agg({
            'avg_altitude': ['mean', 'std', 'count']
        })
        hourly_stats.columns = ['mean_alt', 'std_alt', 'count']
        hourly_stats['deviation'] = hourly_stats['mean_alt'] - mean_alt
        return hourly_stats

    # Compute diurnal patterns
    stable_pattern = compute_diurnal_pattern(df[df['uid'] == stable_uid])
    unstable_pattern = compute_diurnal_pattern(df[df['uid'] == unstable_uid])

    # ===== FFT Spectrum Analysis =====
    def analyze_periodicity(uid_data, device_name):
        """FFT analysis to detect periodicity"""
        # Sort by time and resample to uniform series
        uid_data = uid_data.sort_values('processed_time').set_index('processed_time')
        # Hourly average altitude
        hourly = uid_data['avg_altitude'].resample('1h').mean().dropna()

        if len(hourly) < 48:  # Need at least 2 days
            return None, None

        # Detrend
        detrended = hourly - hourly.rolling(24, center=True).mean()
        detrended = detrended.dropna()

        if len(detrended) < 24:
            return None, None

        # FFT
        signal = detrended.values
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, d=1)  # Hourly sampling, frequency unit is 1/hour

        # Take positive frequencies only
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = np.abs(yf[pos_mask])

        # Convert to period (hours)
        periods = 1 / xf_pos

        return periods, yf_pos

    # Execute FFT analysis
    periods_stable, amplitude_stable = analyze_periodicity(
        df[df['uid'] == stable_uid].copy(), 'stable'
    )
    periods_unstable, amplitude_unstable = analyze_periodicity(
        df[df['uid'] == unstable_uid].copy(), 'unstable'
    )

    # ===== Visualization =====
    fig = plt.figure(figsize=(18, 14))

    # 1. Diurnal Pattern Comparison
    ax1 = fig.add_subplot(2, 3, 1)
    hours = range(24)
    ax1.plot(stable_pattern.index, stable_pattern['deviation'],
             'o-', color=COLORS['tier1'], linewidth=2, markersize=8,
             label=f'Stable Device ...{stable_uid[-4:]}')
    ax1.fill_between(stable_pattern.index,
                     stable_pattern['deviation'] - stable_pattern['std_alt'],
                     stable_pattern['deviation'] + stable_pattern['std_alt'],
                     alpha=0.2, color=COLORS['tier1'])

    ax1.plot(unstable_pattern.index, unstable_pattern['deviation'],
             's--', color=COLORS['tier3'], linewidth=2, markersize=8,
             label=f'Unstable Device ...{unstable_uid[-4:]}')
    ax1.fill_between(unstable_pattern.index,
                     unstable_pattern['deviation'] - unstable_pattern['std_alt'],
                     unstable_pattern['deviation'] + unstable_pattern['std_alt'],
                     alpha=0.2, color=COLORS['tier3'])

    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Altitude Deviation (m)')
    ax1.set_title('24-Hour Altitude Deviation Cycle (Diurnal Pattern)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Stable Device - Heatmap (Hour vs Date)
    ax2 = fig.add_subplot(2, 3, 2)
    stable_data = df[df['uid'] == stable_uid].copy()
    stable_data['date_str'] = stable_data['date'].astype(str)
    pivot_stable = stable_data.pivot_table(
        values='avg_altitude', index='hour', columns='date_str', aggfunc='mean'
    )
    # Subtract mean
    pivot_stable = pivot_stable - pivot_stable.mean()

    sns.heatmap(pivot_stable, cmap='RdBu_r', center=0, ax=ax2,
                cbar_kws={'label': 'Altitude Deviation (m)'})
    ax2.set_title(f'Stable Device Diurnal Heatmap\n(...{stable_uid[-6:]})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Hour')

    # 3. Unstable Device - Heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    unstable_data = df[df['uid'] == unstable_uid].copy()
    unstable_data['date_str'] = unstable_data['date'].astype(str)
    pivot_unstable = unstable_data.pivot_table(
        values='avg_altitude', index='hour', columns='date_str', aggfunc='mean'
    )
    pivot_unstable = pivot_unstable - pivot_unstable.mean()

    sns.heatmap(pivot_unstable, cmap='RdBu_r', center=0, ax=ax3,
                cbar_kws={'label': 'Altitude Deviation (m)'})
    ax3.set_title(f'Unstable Device Diurnal Heatmap\n(...{unstable_uid[-6:]})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Hour')

    # 4. FFT Spectrum - Stable Device
    ax4 = fig.add_subplot(2, 3, 4)
    if periods_stable is not None:
        # Show only periods 4-48 hours
        mask = (periods_stable >= 4) & (periods_stable <= 48)
        ax4.plot(periods_stable[mask], amplitude_stable[mask],
                 color=COLORS['tier1'], linewidth=1.5)
        ax4.axvline(24, color='red', linestyle='--', alpha=0.7, label='24-hour Period')
        ax4.axvline(12, color='orange', linestyle='--', alpha=0.5, label='12-hour Period')
        ax4.set_xlabel('Period (hours)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title(f'Stable Device FFT Spectrum\n(...{stable_uid[-6:]})', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.set_xlim(4, 48)
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax4.transAxes)

    # 5. FFT Spectrum - Unstable Device
    ax5 = fig.add_subplot(2, 3, 5)
    if periods_unstable is not None:
        mask = (periods_unstable >= 4) & (periods_unstable <= 48)
        ax5.plot(periods_unstable[mask], amplitude_unstable[mask],
                 color=COLORS['tier3'], linewidth=1.5)
        ax5.axvline(24, color='red', linestyle='--', alpha=0.7, label='24-hour Period')
        ax5.axvline(12, color='orange', linestyle='--', alpha=0.5, label='12-hour Period')
        ax5.set_xlabel('Period (hours)')
        ax5.set_ylabel('Amplitude')
        ax5.set_title(f'Unstable Device FFT Spectrum\n(...{unstable_uid[-6:]})', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.set_xlim(4, 48)
    else:
        ax5.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax5.transAxes)

    # 6. Diurnal Amplitude Distribution by Tier
    ax6 = fig.add_subplot(2, 3, 6)

    # Calculate diurnal amplitude (max - min hourly deviation) per device
    diurnal_amplitudes = []
    for uid in df['uid'].unique():
        uid_data = df[df['uid'] == uid]
        if len(uid_data) > 24 * 3:  # At least 3 days
            pattern = compute_diurnal_pattern(uid_data)
            amplitude = pattern['deviation'].max() - pattern['deviation'].min()
            tier = device_stats.loc[uid, 'tier'] if uid in device_stats.index else 'Unknown'
            diurnal_amplitudes.append({
                'uid': uid[-6:],
                'amplitude': amplitude,
                'tier': tier
            })

    amp_df = pd.DataFrame(diurnal_amplitudes)

    # Boxplot by tier
    tier_order = ['Tier 1 (High)', 'Tier 2 (Standard)', 'Tier 3 (Warning)']
    tier_colors_list = [COLORS['tier1'], COLORS['tier2'], COLORS['tier3']]

    bp = ax6.boxplot([amp_df[amp_df['tier'] == t]['amplitude'].values for t in tier_order if t in amp_df['tier'].values],
                     labels=[t.split()[0] + ' ' + t.split()[1] for t in tier_order if t in amp_df['tier'].values],
                     patch_artist=True)

    for patch, color in zip(bp['boxes'], tier_colors_list[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax6.set_ylabel('Diurnal Amplitude (m)')
    ax6.set_title('Diurnal Amplitude by Precision Tier', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Diurnal Cycle Analysis (Multipath Detection)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_diurnal_cycle_multipath.png')
    plt.show()

    # ===== Multipath Effect Assessment =====
    print("\nMultipath Effect Analysis Conclusion:")
    print("-" * 50)

    # Calculate 24-hour cycle significance
    stable_amp = stable_pattern['deviation'].max() - stable_pattern['deviation'].min()
    unstable_amp = unstable_pattern['deviation'].max() - unstable_pattern['deviation'].min()

    print(f"   Stable device diurnal amplitude: {stable_amp:.2f}m")
    print(f"   Unstable device diurnal amplitude: {unstable_amp:.2f}m")

    if unstable_amp > 5:
        print("\n   [WARNING] Unstable device shows significant diurnal cycle (amplitude>5m)")
        print("   Possible cause: Multipath effect - signal reflection")
        print("   Suggestion: Check for reflective surfaces near antenna (metal, glass, water)")
    elif unstable_amp > 2:
        print("\n   [CAUTION] Moderate diurnal variation detected")
        print("   Possible cause: Thermal drift or minor multipath effect")
    else:
        print("\n   [OK] Diurnal amplitude within normal range")

    # Save diurnal analysis data
    if len(amp_df) > 0:
        amp_df.to_csv(f'{OUTPUT_DIR}/diurnal_amplitude_analysis.csv', index=False)

    return amp_df


# =============================================================================
# Summary Report Generation
# =============================================================================
def generate_summary_report(device_stats, coupling_df, diurnal_df):
    """Generate comprehensive analysis report"""

    report = f"""
# Sensor Deep Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Vertical Precision Tiering

### Device Tier Statistics
| Tier | Devices | Percentage | Threshold |
|------|---------|------------|-----------|
| Tier 1 (High) | {len(device_stats[device_stats['tier'] == 'Tier 1 (High)'])} | {len(device_stats[device_stats['tier'] == 'Tier 1 (High)'])/len(device_stats)*100:.1f}% | STD <= 1.5m |
| Tier 2 (Standard) | {len(device_stats[device_stats['tier'] == 'Tier 2 (Standard)'])} | {len(device_stats[device_stats['tier'] == 'Tier 2 (Standard)'])/len(device_stats)*100:.1f}% | STD <= 5.0m |
| Tier 3 (Warning) | {len(device_stats[device_stats['tier'] == 'Tier 3 (Warning)'])} | {len(device_stats[device_stats['tier'] == 'Tier 3 (Warning)'])/len(device_stats)*100:.1f}% | STD > 5.0m |

### Gold Standard Devices (Recommended for High-Precision Applications)
"""

    tier1_devices = device_stats[device_stats['tier'] == 'Tier 1 (High)'].head()
    for uid, row in tier1_devices.iterrows():
        report += f"- `{uid[-12:]}`: STD={row['alt_std']:.3f}m, Satellites={row['sat_mean']:.1f}\n"

    report += f"""

## 2. Environmental Coupling Analysis

### Pressure Coupling Strength Distribution
- Strong coupling devices (r < -0.5): {len(coupling_df[coupling_df['corr'] < -0.5])}
- Medium coupling (-0.5 <= r < -0.3): {len(coupling_df[(coupling_df['corr'] >= -0.5) & (coupling_df['corr'] < -0.3)])}
- Weak coupling (r >= -0.3): {len(coupling_df[coupling_df['corr'] >= -0.3])}

**Conclusion**: Negative pressure-altitude correlation indicates GNSS measurements follow atmospheric physics.

## 3. Diurnal Cycle Analysis (Multipath Detection)

### Multipath Risk Assessment
"""

    if diurnal_df is not None and len(diurnal_df) > 0:
        high_risk = diurnal_df[diurnal_df['amplitude'] > 5]
        medium_risk = diurnal_df[(diurnal_df['amplitude'] > 2) & (diurnal_df['amplitude'] <= 5)]
        report += f"- High risk (amplitude>5m): {len(high_risk)} devices\n"
        report += f"- Medium risk (2-5m): {len(medium_risk)} devices\n"
        report += f"- Low risk (<2m): {len(diurnal_df) - len(high_risk) - len(medium_risk)} devices\n"

    report += f"""

## Output Files
- `device_tier_stats.csv`: Device precision tier details
- `device_coupling_analysis.csv`: Environmental coupling analysis
- `diurnal_amplitude_analysis.csv`: Diurnal cycle analysis
- `01_vertical_precision_tiering.png`: Precision tiering visualization
- `02_environmental_coupling.png`: Environmental coupling visualization
- `03_diurnal_cycle_multipath.png`: Diurnal cycle analysis visualization

---
*This report was auto-generated by the deep sensor analysis script*
"""

    # Save report
    report_path = f'{OUTPUT_DIR}/deep_analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nAnalysis report saved to: {report_path}")
    return report


# =============================================================================
# Main Program
# =============================================================================
def main():
    print("=" * 70)
    print("Deep Sensor Analysis - Three Core Dimensions")
    print("=" * 70)

    # Load data
    df = load_data('sensor_data_clean_stable.csv')

    if df is None or df.empty:
        print("Data loading failed")
        return

    # 1. Vertical Precision Tiering
    device_stats = analyze_vertical_precision(df)

    # 2. Environmental Coupling Analysis
    coupling_df, golden_uid = analyze_environmental_coupling(df, device_stats)

    # 3. Diurnal Cycle Analysis
    diurnal_df = analyze_diurnal_cycle(df, device_stats)

    # Generate summary report
    report = generate_summary_report(device_stats, coupling_df, diurnal_df)

    print("\n" + "=" * 70)
    print("Deep Analysis Complete!")
    print("=" * 70)
    print(f"All output files saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
