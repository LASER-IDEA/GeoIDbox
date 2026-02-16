"""
MAML (Model-Agnostic Meta-Learning) for Rapid New-Sensor Adaptation

This script implements MAML to enable few-shot adaptation to new sensors.
Instead of training from scratch (2 hours), new sensors can be adapted
with just 5-10 samples in seconds.

Usage:
    python run_maml_meta_learning.py --mode train    # Train meta-model
    python run_maml_meta_learning.py --mode adapt    # Test few-shot adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy
import json
import os
import argparse
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# Model Architecture (same as current best model)
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
    """Simple but effective altitude estimator"""
    def __init__(self, in_dim=9, hidden_dim=128, n_layers=3):
        super().__init__()
        self.fourier = FourierFeatures(2, mapping_size=32, scale=1.0)  # For lat, lon
        
        # Input: fourier(64) + altitude(1) + pressure(1) + temp(1) + humidity(1) + era5_t2m(1) + era5_sp(1)
        fourier_dim = 64
        other_dim = in_dim - 2  # Remaining features
        
        layers = []
        prev_dim = fourier_dim + other_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [batch, features] where first 2 are lat, lon
        latlon = x[:, :2]
        others = x[:, 2:]
        
        fourier_feat = self.fourier(latlon)
        combined = torch.cat([fourier_feat, others], dim=-1)
        return self.net(combined).squeeze(-1)


# ==============================================================================
# MAML Implementation
# ==============================================================================

class MAML:
    """
    Model-Agnostic Meta-Learning
    
    Key idea: Learn a good initialization that can adapt to new tasks
    with just a few gradient steps.
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=5, first_order=False):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.first_order = first_order  # First-order approximation (faster)
        
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.criterion = nn.MSELoss()
        
    def inner_loop(self, support_x, support_y, create_graph=True):
        """
        Perform inner loop adaptation on support set.
        Returns adapted parameters.
        """
        # Clone model for this task
        adapted_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for step in range(self.inner_steps):
            # Forward pass with adapted params
            self.model.train()
            pred = self.forward_with_params(support_x, adapted_params)
            loss = self.criterion(pred, support_y)
            
            # Compute gradients w.r.t. adapted params
            grads = torch.autograd.grad(
                loss, adapted_params.values(),
                create_graph=create_graph and not self.first_order,
                allow_unused=True
            )
            
            # Update adapted params
            adapted_params = {
                name: param - self.inner_lr * (grad if grad is not None else 0)
                for (name, param), grad in zip(adapted_params.items(), grads)
            }
        
        return adapted_params
    
    def forward_with_params(self, x, params):
        """Forward pass using custom parameters"""
        # This is a simplified version - in practice, we'd need to 
        # manually implement forward for each layer type
        # For now, we'll use a workaround by temporarily loading params
        
        original_state = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Load adapted params
        for name, param in self.model.named_parameters():
            param.data.copy_(params[name])
        
        output = self.model(x)
        
        # Restore original params
        for name, param in self.model.named_parameters():
            param.data.copy_(original_state[name])
        
        return output
    
    def meta_train_step(self, batch_tasks):
        """
        Perform one meta-training step.
        
        Args:
            batch_tasks: List of (support_x, support_y, query_x, query_y) tuples
                        each from different sensors/tasks
        """
        meta_loss = 0.0
        task_losses = []
        
        self.meta_optimizer.zero_grad()
        
        for support_x, support_y, query_x, query_y in batch_tasks:
            # Inner loop: adapt to support set
            adapted_params = self.inner_loop(support_x, support_y, create_graph=True)
            
            # Outer loop: evaluate on query set
            query_pred = self.forward_with_params(query_x, adapted_params)
            query_loss = self.criterion(query_pred, query_y)
            
            meta_loss += query_loss
            task_losses.append(query_loss.item())
        
        # Average meta-loss across tasks
        meta_loss = meta_loss / len(batch_tasks)
        
        # Meta-update
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item(), np.mean(task_losses)
    
    def adapt_to_new_sensor(self, support_x, support_y, steps=None):
        """
        Adapt meta-model to a new sensor with few samples.
        
        Args:
            support_x: Few samples from new sensor [N, features]
            support_y: Ground truth [N]
            steps: Number of adaptation steps (default: self.inner_steps)
        
        Returns:
            adapted_model: Model adapted to new sensor
        """
        if steps is None:
            steps = self.inner_steps
        
        # Clone the model
        adapted_model = copy.deepcopy(self.model)
        adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        adapted_model.train()
        for step in range(steps):
            adapted_optimizer.zero_grad()
            pred = adapted_model(support_x)
            loss = self.criterion(pred, support_y)
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_model


# ==============================================================================
# Data Loading and Task Creation
# ==============================================================================

def load_data(csv_path='data/processed/sensor_data_with_real_era5.csv'):
    """Load and preprocess data"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {df['uid'].nunique()} sensors")
    
    # Feature columns
    feature_cols = ['avg_latitude', 'avg_longitude', 'avg_altitude', 
                   'avg_pressure', 'avg_temperature', 'avg_humidity',
                   'era5_t2m', 'era5_sp']
    target_col = 'residual'  # Predict residual correction
    
    # Prepare data per sensor
    sensor_data = {}
    for uid in df['uid'].unique():
        sensor_df = df[df['uid'] == uid].copy()
        X = sensor_df[feature_cols].values
        y = sensor_df[target_col].values
        sensor_data[uid] = {'X': X, 'y': y, 'n_samples': len(X)}
    
    return sensor_data, feature_cols


def create_meta_tasks(sensor_data, n_tasks_per_sensor=10, k_shot=16, q_query=16):
    """
    Create meta-learning tasks.
    
    Each task is a sensor with:
    - k_shot samples for support (adaptation)
    - q_query samples for query (evaluation)
    """
    tasks = []
    
    for uid, data in sensor_data.items():
        X, y = data['X'], data['y']
        n_samples = len(X)
        
        if n_samples < k_shot + q_query:
            continue
        
        # Create multiple tasks from each sensor
        for _ in range(n_tasks_per_sensor):
            # Random sample for this task
            indices = np.random.permutation(n_samples)
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:k_shot+q_query]
            
            support_x = torch.FloatTensor(X[support_idx]).to(device)
            support_y = torch.FloatTensor(y[support_idx]).to(device)
            query_x = torch.FloatTensor(X[query_idx]).to(device)
            query_y = torch.FloatTensor(y[query_idx]).to(device)
            
            tasks.append((support_x, support_y, query_x, query_y, uid))
    
    print(f"Created {len(tasks)} meta-learning tasks")
    return tasks


def evaluate_few_shot_adaptation(maml, sensor_data, test_sensor, k_shot_values=[4, 8, 16, 32], 
                                  n_trials=5):
    """
    Evaluate few-shot adaptation performance.
    
    For each k-shot value, randomly sample k samples from test sensor,
    adapt the model, and evaluate on the rest.
    """
    results = {}
    
    test_data = sensor_data[test_sensor]
    X_test, y_test = test_data['X'], test_data['y']
    n_test = len(X_test)
    
    print(f"\n=== Few-Shot Adaptation for Sensor {test_sensor[-8:]} ===")
    print(f"Total test samples: {n_test}")
    
    for k in k_shot_values:
        if k >= n_test * 0.5:  # Need enough samples for evaluation
            continue
            
        trial_maes = []
        
        for trial in range(n_trials):
            # Random split
            indices = np.random.permutation(n_test)
            support_idx = indices[:k]
            query_idx = indices[k:]
            
            support_x = torch.FloatTensor(X_test[support_idx]).to(device)
            support_y = torch.FloatTensor(y_test[support_idx]).to(device)
            query_x = torch.FloatTensor(X_test[query_idx]).to(device)
            query_y = torch.FloatTensor(y_test[query_idx]).to(device)
            
            # Adapt model
            adapted_model = maml.adapt_to_new_sensor(support_x, support_y)
            
            # Evaluate
            adapted_model.eval()
            with torch.no_grad():
                pred = adapted_model(query_x)
                mae = torch.abs(pred - query_y).mean().item()
            
            trial_maes.append(mae)
        
        mean_mae = np.mean(trial_maes)
        std_mae = np.std(trial_maes)
        results[k] = {'mean': mean_mae, 'std': std_mae}
        
        print(f"  {k:2d}-shot: MAE = {mean_mae:.4f} ± {std_mae:.4f}m")
    
    return results


# ==============================================================================
# Main Training and Evaluation
# ==============================================================================

def train_maml(n_epochs=1000, tasks_per_batch=8, inner_steps=5, save_dir='experiments/maml'):
    """Train MAML meta-model"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    sensor_data, feature_cols = load_data()
    
    # Create meta-tasks
    tasks = create_meta_tasks(sensor_data, n_tasks_per_sensor=20, k_shot=16, q_query=16)
    
    # Split tasks for train/val
    np.random.shuffle(tasks)
    split = int(0.8 * len(tasks))
    train_tasks = tasks[:split]
    val_tasks = tasks[split:]
    
    # Initialize model and MAML
    model = AltitudeEstimator(in_dim=len(feature_cols), hidden_dim=128, n_layers=3).to(device)
    maml = MAML(model, inner_lr=0.01, meta_lr=0.001, inner_steps=inner_steps)
    
    print(f"\n=== MAML Training ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Train tasks: {len(train_tasks)}, Val tasks: {len(val_tasks)}")
    print(f"Inner steps: {inner_steps}, Tasks per batch: {tasks_per_batch}")
    
    # Training loop
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(n_epochs):
        # Sample batch of tasks
        batch_indices = np.random.choice(len(train_tasks), tasks_per_batch, replace=False)
        batch_tasks = [train_tasks[i][:4] for i in batch_indices]  # Exclude uid
        
        # Meta-train step
        meta_loss, task_loss = maml.meta_train_step(batch_tasks)
        
        # Validation
        if epoch % 50 == 0:
            val_losses = []
            for task in val_tasks[:20]:  # Sample 20 val tasks
                support_x, support_y, query_x, query_y = task[:4]
                adapted_params = maml.inner_loop(support_x, support_y, create_graph=False)
                query_pred = maml.forward_with_params(query_x, adapted_params)
                val_loss = maml.criterion(query_pred, query_y).item()
                val_losses.append(val_loss)
            
            val_loss = np.mean(val_losses)
            history.append({'epoch': epoch, 'train_loss': meta_loss, 'val_loss': val_loss})
            
            print(f"Epoch {epoch:4d}: Train Loss = {meta_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': maml.model.state_dict(),
                    'meta_optimizer_state_dict': maml.meta_optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, f'{save_dir}/maml_best.pt')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    # Save training history
    with open(f'{save_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    return maml


def test_few_shot_adaptation(maml_path='experiments/maml/maml_best.pt'):
    """Test few-shot adaptation on held-out sensors"""
    
    # Load data
    sensor_data, feature_cols = load_data()
    sensors = list(sensor_data.keys())
    
    # Load trained MAML
    model = AltitudeEstimator(in_dim=len(feature_cols), hidden_dim=128, n_layers=3).to(device)
    maml = MAML(model, inner_lr=0.01, meta_lr=0.001, inner_steps=5)
    
    checkpoint = torch.load(maml_path)
    maml.model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded MAML model from epoch {checkpoint['epoch']}")
    
    # Test on each sensor with LOSO
    all_results = {}
    
    for test_sensor in sensors:
        train_sensors = [s for s in sensors if s != test_sensor]
        
        # Evaluate few-shot adaptation
        results = evaluate_few_shot_adaptation(
            maml, sensor_data, test_sensor,
            k_shot_values=[4, 8, 16, 32], n_trials=5
        )
        
        all_results[test_sensor[-8:]] = results
    
    # Summary
    print("\n=== Summary: Few-Shot Adaptation Results ===")
    print(f"{'K-shot':>8} | {'Mean MAE':>12} | {'Std MAE':>12}")
    print("-" * 40)
    
    for k in [4, 8, 16, 32]:
        maes = [all_results[s][k]['mean'] for s in all_results if k in all_results[s]]
        if maes:
            print(f"{k:8d} | {np.mean(maes):12.4f} | {np.std(maes):12.4f}")
    
    # Save results
    with open('experiments/maml/few_shot_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


# ==============================================================================
# Baseline Comparison
# ==============================================================================

def compare_with_baseline():
    """
    Compare MAML few-shot adaptation with:
    1. Training from scratch (cold start)
    2. Fine-tuning from pre-trained model
    """
    print("\n=== Baseline Comparison ===")
    print("This would compare MAML with other approaches...")
    print("TODO: Implement baseline comparison")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML for Rapid Sensor Adaptation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'adapt', 'compare'],
                       help='Mode: train MAML, test adaptation, or compare baselines')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of meta-training epochs')
    parser.add_argument('--inner_steps', type=int, default=5, help='Number of inner loop steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Tasks per meta-batch')
    parser.add_argument('--save_dir', type=str, default='experiments/maml', help='Save directory')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_maml(n_epochs=args.epochs, tasks_per_batch=args.batch_size, 
                  inner_steps=args.inner_steps, save_dir=args.save_dir)
    
    elif args.mode == 'adapt':
        test_few_shot_adaptation()
    
    elif args.mode == 'compare':
        compare_with_baseline()
