#!/usr/bin/env python3
"""
Generate Figure 5: Curriculum Learning Curves from REAL training history.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set output directory
os.makedirs('paper/figures', exist_ok=True)

plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.dpi'] = 300

print("=" * 70)
print("GENERATING FIGURE 5: Curriculum Learning Curves")
print("=" * 70)

# Load training history
history_file = '../experiments/results/curriculum_history/training_history.json'
if not os.path.exists(history_file):
    history_file = 'experiments/results/curriculum_history/training_history.json'

if not os.path.exists(history_file):
    print(f"\n❌ History file not found: {history_file}")
    print("Please run training first:")
    print("  python run_curriculum_with_history.py")
    exit(1)

print(f"\n[1] Loading training history...")
with open(history_file) as f:
    history = json.load(f)

print(f"  ✓ Loaded {len(history['epochs'])} checkpoints")
print(f"  ✓ Stages: {set(history['stages'])}")

# Extract data
epochs = np.array(history['epochs'])
losses = np.array(history['losses'])
maes = np.array(history['maes'])
stages = np.array(history['stages'])

# Separate by stage
stage1_mask = stages == 1
stage2_mask = stages == 2
stage3_mask = stages == 3

# Create figure
print("\n[2] Generating figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Training Loss
ax1.plot(epochs[stage1_mask], losses[stage1_mask], 
         label='Stage 1: Easy', color='#E67E22', linewidth=2.5, marker='o', markersize=4)
ax1.plot(epochs[stage2_mask], losses[stage2_mask], 
         label='Stage 2: Medium', color='#D35400', linewidth=2.5, marker='s', markersize=4)
ax1.plot(epochs[stage3_mask], losses[stage3_mask], 
         label='Stage 3: Hard', color='#C0392B', linewidth=2.5, marker='^', markersize=4)

ax1.set_xlabel('Global Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Training Loss (MSE)', fontsize=13, fontweight='bold')
ax1.set_title('Curriculum Learning: Training Loss\n(Real Training History)', 
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', framealpha=0.95)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#FEF9E7')

# Add stage boundaries
stage1_end = epochs[stage1_mask].max() if stage1_mask.any() else 0
stage2_end = epochs[stage2_mask].max() if stage2_mask.any() else 0
ax1.axvline(x=stage1_end, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax1.axvline(x=stage2_end, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

# Add stage labels
ax1.text(stage1_end/2, ax1.get_ylim()[1]*0.9, 'Stage 1\n(Easy)', 
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='#E67E22', alpha=0.3))
ax1.text((stage1_end+stage2_end)/2, ax1.get_ylim()[1]*0.9, 'Stage 2\n(Medium)', 
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='#D35400', alpha=0.3))
ax1.text((stage2_end+epochs.max())/2, ax1.get_ylim()[1]*0.9, 'Stage 3\n(Hard)', 
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='#C0392B', alpha=0.3))

# Plot 2: Validation MAE
ax2.plot(epochs[stage1_mask], maes[stage1_mask], 
         label='Stage 1: Easy', color='#27AE60', linewidth=2.5, marker='o', markersize=4)
ax2.plot(epochs[stage2_mask], maes[stage2_mask], 
         label='Stage 2: Medium', color='#F39C12', linewidth=2.5, marker='s', markersize=4)
ax2.plot(epochs[stage3_mask], maes[stage3_mask], 
         label='Stage 3: Hard', color='#C0392B', linewidth=2.5, marker='^', markersize=4)

ax2.set_xlabel('Global Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('Validation MAE (m)', fontsize=13, fontweight='bold')
ax2.set_title('Curriculum Learning: Validation MAE\n(Real Training History)', 
              fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#FEF9E7')

# Add target line
ax2.axhline(y=10, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Target: 10m')

# Add stage boundaries
ax2.axvline(x=stage1_end, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax2.axvline(x=stage2_end, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

# Add final result annotation
final_mae = maes[-1]
ax2.annotate(f'Final MAE:\n{final_mae:.2f}m', 
             xy=(epochs[-1], final_mae), 
             xytext=(epochs[-1]-100, final_mae+5),
             fontsize=12, fontweight='bold', color='#C0392B',
             arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5),
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='#C0392B', linewidth=2))

plt.tight_layout()
plt.savefig('paper/figures/fig5_curriculum.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("  ✓ Saved: paper/figures/fig5_curriculum.png")

# Print summary
print("\n" + "=" * 70)
print("FIGURE 5 GENERATION COMPLETE")
print("=" * 70)
print(f"\nTraining Summary:")
print(f"  Total epochs: {epochs.max()}")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Initial MAE: {maes[0]:.2f}m")
print(f"  Final MAE: {maes[-1]:.2f}m")
print(f"  Best MAE: {maes.min():.2f}m")
print("\n" + "=" * 70)
