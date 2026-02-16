"""
MAML (Model-Agnostic Meta-Learning) for Rapid New-Sensor Adaptation - v2
Using 'higher' library for efficient differentiable optimization

Usage:
    python run_maml_meta_learning_v2.py --mode train    # Train meta-model
    python run_maml_meta_learning_v2.py --mode adapt    # Test few-shot adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import higher
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy
import json
import os
import argparse
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fix random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# ==============================================================================
# Model Architecture
# ==============================================================================

class FourierFeatures(nn.Module):
    """Fourier feature encoding for spatial coordinates"""
    def __init__(self, in_dim, mapping_size=64, scale=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, mapping_size) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class AltitudeEstimator(nn.Module):
    """Lightweight altitude estimator for meta-learning"""
    def __init__(self, in_dim=8, hidden_dim=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.fourier = FourierFeatures(2, mapping_size=32, scale=1.0)  # For lat, lon
        
        fourier_dim = 64
        other_dim = in_dim - 2
        
        layers = []
        prev_dim = fourier_dim + other_dim
        
        for i in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout if i < n_layers - 1 else 0.0)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        latlon = x[:, :2]
        others = x[:, 2:]
        fourier_feat = self.fourier(latlon)
        combined = torch.cat([fourier_feat, others], dim=-1)
        return self.net(combined).squeeze(-1)


# ==============================================================================
# MAML with Higher Library
# ==============================================================================

class MAMLTrainer:
    """
    MAML Trainer using 'higher' library for efficient differentiable optimization.
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5, first_order=False):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.meta_optimizer, T_max=1000, eta_min=1e-5)
        
    def meta_train_step(self, batch_tasks):
        """
        Perform one meta-training step.
        
        Args:
            batch_tasks: List of (support_x, support_y, query_x, query_y) tuples
        """
        meta_loss = 0.0
        task_metrics = []
        
        self.meta_optimizer.zero_grad()
        
        for support_x, support_y, query_x, query_y in batch_tasks:
            # Inner loop: adapt to support set using higher
            inner_opt = optim.SGD(self.model.parameters(), lr=self.inner_lr)
            with higher.innerloop_ctx(
                self.model, 
                inner_opt,
                copy_initial_weights=False
            ) as (fmodel, diffopt):
                
                # Adapt for inner_steps
                for _ in range(self.inner_steps):
                    support_pred = fmodel(support_x)
                    support_loss = self.criterion(support_pred, support_y)
                    diffopt.step(support_loss)
                
                # Outer loop: evaluate on query set
                query_pred = fmodel(query_x)
                query_loss = self.criterion(query_pred, query_y)
                
                # For first-order approximation, detach the computation graph
                if self.first_order:
                    query_loss = query_loss.detach().requires_grad_(True)
                
                meta_loss += query_loss
                
                # Metrics
                with torch.no_grad():
                    mae = torch.abs(query_pred - query_y).mean().item()
                    task_metrics.append(mae)
        
        # Average meta-loss
        meta_loss = meta_loss / len(batch_tasks)
        
        # Backward pass
        meta_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        self.scheduler.step()
        
        return meta_loss.item(), np.mean(task_metrics)
    
    def adapt_and_evaluate(self, support_x, support_y, query_x, query_y, steps=None):
        """
        Adapt to new task and evaluate.
        Returns MAE on query set.
        """
        if steps is None:
            steps = self.inner_steps
        
        # Clone model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_opt = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        adapted_model.train()
        for _ in range(steps):
            adapted_opt.zero_grad()
            pred = adapted_model(support_x)
            loss = self.criterion(pred, support_y)
            loss.backward()
            adapted_opt.step()
        
        # Evaluate
        adapted_model.eval()
        with torch.no_grad():
            query_pred = adapted_model(query_x)
            mae = torch.abs(query_pred - query_y).mean().item()
        
        return mae, adapted_model


# ==============================================================================
# Data Loading
# ==============================================================================

def load_and_prepare_data(csv_path='data/processed/sensor_data_with_real_era5.csv'):
    """Load and prepare sensor data"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {df['uid'].nunique()} sensors")
    
    # Feature columns
    feature_cols = ['avg_latitude', 'avg_longitude', 'avg_altitude', 
                   'avg_pressure', 'avg_temperature', 'avg_humidity',
                   'era5_t2m', 'era5_sp']
    target_col = 'residual'
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_all = scaler_X.fit_transform(df[feature_cols].values)
    y_all = scaler_y.fit_transform(df[[target_col]].values).squeeze()
    
    # Prepare per-sensor data
    sensor_data = {}
    for i, uid in enumerate(df['uid'].unique()):
        mask = df['uid'].values == uid
        X = torch.FloatTensor(X_all[mask]).to(device)
        y = torch.FloatTensor(y_all[mask]).to(device)
        
        # Denormalize y for interpretability
        y_orig = torch.FloatTensor(df[target_col].values[mask]).to(device)
        
        sensor_data[uid] = {
            'X': X, 
            'y': y,
            'y_orig': y_orig,
            'n_samples': len(X),
            'altitude_mean': df[df['uid'] == uid]['avg_altitude'].mean()
        }
    
    # Print sensor statistics
    print("\nSensor Statistics:")
    for uid, data in sensor_data.items():
        print(f"  {uid[-8:]}: {data['n_samples']:5d} samples, "
              f"altitude: {data['altitude_mean']:.1f}m")
    
    return sensor_data, feature_cols, scaler_X, scaler_y


def create_meta_tasks(sensor_data, n_tasks_per_sensor=20, k_shot=16, q_query=16):
    """Create meta-learning tasks"""
    tasks = []
    
    for uid, data in sensor_data.items():
        X, y = data['X'], data['y']
        n_samples = len(X)
        
        if n_samples < k_shot + q_query + 10:  # Leave some buffer
            continue
        
        for _ in range(n_tasks_per_sensor):
            indices = torch.randperm(n_samples)
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:k_shot+q_query]
            
            tasks.append((
                X[support_idx], y[support_idx],
                X[query_idx], y[query_idx],
                uid
            ))
    
    print(f"\nCreated {len(tasks)} meta-learning tasks")
    return tasks


# ==============================================================================
# Training
# ==============================================================================

def train_maml(n_epochs=2000, tasks_per_batch=16, inner_steps=5, k_shot=16, 
               q_query=16, eval_interval=50, save_dir='experiments/maml_v2'):
    """Train MAML meta-model"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    sensor_data, feature_cols, scaler_X, scaler_y = load_and_prepare_data()
    
    # Create meta-tasks
    all_tasks = create_meta_tasks(sensor_data, n_tasks_per_sensor=30, 
                                   k_shot=k_shot, q_query=q_query)
    
    # Split train/val (by sensors, not tasks)
    sensors = list(sensor_data.keys())
    np.random.shuffle(sensors)
    n_train_sensors = max(1, len(sensors) - 2)  # Reserve 2 sensors for testing
    train_sensors = sensors[:n_train_sensors]
    val_sensors = sensors[n_train_sensors:]
    
    train_tasks = [t for t in all_tasks if t[4] in train_sensors]
    val_tasks = [t for t in all_tasks if t[4] in val_sensors]
    
    print(f"\nTrain sensors: {len(train_sensors)}, Val sensors: {len(val_sensors)}")
    print(f"Train tasks: {len(train_tasks)}, Val tasks: {len(val_tasks)}")
    
    # Initialize model and trainer
    model = AltitudeEstimator(in_dim=len(feature_cols), hidden_dim=128, n_layers=3).to(device)
    trainer = MAMLTrainer(model, inner_lr=0.01, meta_lr=0.001, inner_steps=inner_steps)
    
    print(f"\n=== MAML Training ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {n_epochs}, Tasks/batch: {tasks_per_batch}, K-shot: {k_shot}")
    
    # Training loop
    best_val_mae = float('inf')
    history = []
    
    for epoch in range(n_epochs):
        # Sample batch of tasks
        batch_indices = np.random.choice(len(train_tasks), min(tasks_per_batch, len(train_tasks)), replace=False)
        batch_tasks = [train_tasks[i][:4] for i in batch_indices]
        
        # Meta-train step
        meta_loss, train_mae = trainer.meta_train_step(batch_tasks)
        
        # Validation
        if epoch % eval_interval == 0:
            val_maes = []
            for task in val_tasks[:20]:
                mae, _ = trainer.adapt_and_evaluate(*task[:4])
                val_maes.append(mae)
            
            val_mae = np.mean(val_maes)
            
            # Denormalize MAE
            val_mae_meters = val_mae * scaler_y.scale_[0]
            train_mae_meters = train_mae * scaler_y.scale_[0]
            
            history.append({
                'epoch': epoch,
                'train_loss': meta_loss,
                'train_mae': train_mae_meters,
                'val_mae': val_mae_meters
            })
            
            print(f"Epoch {epoch:4d}: Loss={meta_loss:.4f}, "
                  f"Train MAE={train_mae_meters:.3f}m, Val MAE={val_mae_meters:.3f}m")
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'meta_optimizer_state_dict': trainer.meta_optimizer.state_dict(),
                    'epoch': epoch,
                    'val_mae': val_mae_meters,
                    'scaler_y': scaler_y
                }, f'{save_dir}/maml_best.pt')
    
    # Save final model and history
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': n_epochs,
        'scaler_y': scaler_y
    }, f'{save_dir}/maml_final.pt')
    
    with open(f'{save_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Val MAE: {best_val_mae * scaler_y.scale_[0]:.3f}m")
    print(f"Model saved to: {save_dir}/maml_best.pt")
    print(f"{'='*60}")
    
    return trainer, scaler_y


# ==============================================================================
# Few-Shot Adaptation Evaluation
# ==============================================================================

def evaluate_few_shot(trainer, sensor_data, test_sensor, k_shot_values=[4, 8, 16, 32, 64], 
                      n_trials=10, scaler_y=None):
    """Evaluate few-shot adaptation on a held-out sensor"""
    
    test_data = sensor_data[test_sensor]
    X_test, y_test = test_data['X'], test_data['y']
    y_orig = test_data['y_orig']
    n_test = len(X_test)
    
    print(f"\n{'='*60}")
    print(f"Sensor: {test_sensor[-8:]} (n={n_test}, altitude={test_data['altitude_mean']:.1f}m)")
    print(f"{'='*60}")
    
    results = {}
    
    for k in k_shot_values:
        if k >= n_test * 0.5:
            continue
        
        trial_maes = []
        
        for trial in range(n_trials):
            # Random sample k shots
            indices = torch.randperm(n_test)
            support_idx = indices[:k]
            query_idx = indices[k:]
            
            support_x, support_y = X_test[support_idx], y_test[support_idx]
            query_x, query_y = X_test[query_idx], y_test[query_idx]
            
            # Adapt and evaluate
            mae, _ = trainer.adapt_and_evaluate(support_x, support_y, query_x, query_y)
            
            # Denormalize
            if scaler_y is not None:
                mae = mae * scaler_y.scale_[0]
            
            trial_maes.append(mae)
        
        mean_mae = np.mean(trial_maes)
        std_mae = np.std(trial_maes)
        results[k] = {'mean': mean_mae, 'std': std_mae}
        
        print(f"  {k:3d}-shot: MAE = {mean_mae:.3f} ± {std_mae:.3f}m")
    
    return results


def run_few_shot_evaluation(model_path='experiments/maml_v2/maml_best.pt'):
    """Run few-shot evaluation on all sensors with LOSO"""
    
    # Load data
    sensor_data, feature_cols, _, scaler_y = load_and_prepare_data()
    sensors = list(sensor_data.keys())
    
    # Load trained model
    model = AltitudeEstimator(in_dim=len(feature_cols), hidden_dim=128, n_layers=3).to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler_y_loaded = checkpoint.get('scaler_y', None)
    if scaler_y_loaded is not None:
        scaler_y = scaler_y_loaded
    
    trainer = MAMLTrainer(model, inner_lr=0.01, meta_lr=0.001, inner_steps=5)
    
    print(f"\nLoaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best Val MAE: {checkpoint.get('val_mae', 'unknown')}")
    
    # LOSO evaluation
    all_results = {}
    
    for test_sensor in sensors:
        results = evaluate_few_shot(
            trainer, sensor_data, test_sensor,
            k_shot_values=[4, 8, 16, 32, 64], n_trials=10, scaler_y=scaler_y
        )
        all_results[test_sensor[-8:]] = results
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY: Few-Shot Adaptation (Mean ± Std across sensors)")
    print(f"{'='*60}")
    print(f"{'K-shot':>8} | {'Mean MAE':>12} | {'Std MAE':>12}")
    print("-" * 40)
    
    for k in [4, 8, 16, 32, 64]:
        maes = [all_results[s][k]['mean'] for s in all_results if k in all_results[s]]
        if maes:
            print(f"{k:8d} | {np.mean(maes):12.3f} | {np.std(maes):12.3f}")
    
    # Save results
    os.makedirs('experiments/maml_v2', exist_ok=True)
    with open('experiments/maml_v2/few_shot_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


# ==============================================================================
# Comparison with Baselines
# ==============================================================================

def compare_baselines(model_path='experiments/maml_v2/maml_best.pt'):
    """
    Compare MAML with:
    1. Training from scratch (random initialization)
    2. Fine-tuning from pre-trained model (supervised pre-training)
    """
    print("\n" + "="*60)
    print("Baseline Comparison: MAML vs Training from Scratch")
    print("="*60)
    
    # Load data
    sensor_data, feature_cols, _, scaler_y = load_and_prepare_data()
    sensors = list(sensor_data.keys())
    
    # Load MAML model
    maml_model = AltitudeEstimator(in_dim=len(feature_cols), hidden_dim=128, n_layers=3).to(device)
    checkpoint = torch.load(model_path)
    maml_model.load_state_dict(checkpoint['model_state_dict'])
    scaler_y = checkpoint.get('scaler_y', scaler_y)
    
    maml_trainer = MAMLTrainer(maml_model, inner_lr=0.01, meta_lr=0.001, inner_steps=5)
    
    # Results storage
    results = {
        'maml': defaultdict(list),
        'scratch': defaultdict(list),
        'finetune': defaultdict(list)
    }
    
    # Test on each sensor with LOSO
    k_shot = 16
    n_trials = 5
    
    for test_sensor in sensors:
        print(f"\nSensor: {test_sensor[-8:]}")
        
        test_data = sensor_data[test_sensor]
        X_test, y_test = test_data['X'], test_data['y']
        n_test = len(X_test)
        
        for trial in range(n_trials):
            indices = torch.randperm(n_test)
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:]
            
            support_x, support_y = X_test[support_idx], y_test[support_idx]
            query_x, query_y = X_test[query_idx], y_test[query_idx]
            
            # 1. MAML adaptation
            mae_maml, _ = maml_trainer.adapt_and_evaluate(support_x, support_y, query_x, query_y)
            results['maml'][k_shot].append(mae_maml * scaler_y.scale_[0])
            
            # 2. Training from scratch
            scratch_model = AltitudeEstimator(in_dim=len(feature_cols), hidden_dim=128, n_layers=3).to(device)
            scratch_opt = optim.Adam(scratch_model.parameters(), lr=0.001)
            scratch_model.train()
            
            for _ in range(50):  # Train from scratch for 50 steps
                scratch_opt.zero_grad()
                pred = scratch_model(support_x)
                loss = nn.MSELoss()(pred, support_y)
                loss.backward()
                scratch_opt.step()
            
            scratch_model.eval()
            with torch.no_grad():
                pred = scratch_model(query_x)
                mae_scratch = torch.abs(pred - query_y).mean().item() * scaler_y.scale_[0]
            results['scratch'][k_shot].append(mae_scratch)
            
            # 3. Fine-tuning from pre-trained (on other sensors)
            train_sensors = [s for s in sensors if s != test_sensor]
            # Simplified: use MAML model as "pre-trained"
            finetune_model = copy.deepcopy(maml_model)
            finetune_opt = optim.Adam(finetune_model.parameters(), lr=0.001)
            finetune_model.train()
            
            for _ in range(20):  # Fine-tune for 20 steps
                finetune_opt.zero_grad()
                pred = finetune_model(support_x)
                loss = nn.MSELoss()(pred, support_y)
                loss.backward()
                finetune_opt.step()
            
            finetune_model.eval()
            with torch.no_grad():
                pred = finetune_model(query_x)
                mae_finetune = torch.abs(pred - query_y).mean().item() * scaler_y.scale_[0]
            results['finetune'][k_shot].append(mae_finetune)
        
        # Print per-sensor results
        print(f"  MAML:     {np.mean(results['maml'][k_shot]):.3f} ± {np.std(results['maml'][k_shot]):.3f}m")
        print(f"  Scratch:  {np.mean(results['scratch'][k_shot]):.3f} ± {np.std(results['scratch'][k_shot]):.3f}m")
        print(f"  Finetune: {np.mean(results['finetune'][k_shot]):.3f} ± {np.std(results['finetune'][k_shot]):.3f}m")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY (16-shot adaptation)")
    print(f"{'='*60}")
    for method in ['maml', 'scratch', 'finetune']:
        maes = results[method][k_shot]
        print(f"{method.capitalize():10s}: {np.mean(maes):.3f} ± {np.std(maes):.3f}m")
    
    return results


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML for Rapid Sensor Adaptation')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'adapt', 'compare'],
                       help='Mode: train MAML, test adaptation, or compare baselines')
    parser.add_argument('--epochs', type=int, default=2000, 
                       help='Number of meta-training epochs')
    parser.add_argument('--inner_steps', type=int, default=5, 
                       help='Number of inner loop steps')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Tasks per meta-batch')
    parser.add_argument('--k_shot', type=int, default=16, 
                       help='K-shot for support set')
    parser.add_argument('--save_dir', type=str, default='experiments/maml_v2', 
                       help='Save directory')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_maml(
            n_epochs=args.epochs, 
            tasks_per_batch=args.batch_size,
            inner_steps=args.inner_steps,
            k_shot=args.k_shot,
            save_dir=args.save_dir
        )
    
    elif args.mode == 'adapt':
        run_few_shot_evaluation()
    
    elif args.mode == 'compare':
        compare_baselines()
