"""
Real-time training monitor for fullerene diffusion model.

Tracks loss trends, component analysis, and generates progress reports.

Usage:
    python monitor_training.py training_v4_high_lambda_*.log
"""

import sys
import re
import numpy as np
from pathlib import Path

def analyze_training_log(log_file):
    """Analyze training log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    print("="*90)
    print(f" TRAINING MONITOR: {Path(log_file).name}".center(90))
    print("="*90)
    print()
    
    # Extract epoch summaries
    epochs = re.findall(r'Epoch (\d+)/\d+\s+Train loss: ([\d\.]+)\s+Val loss: ([\d\.]+)', content)
    
    if epochs:
        print(f"📊 Epoch Progress ({len(epochs)} epochs completed):\n")
        print("Epoch | Train Loss | Val Loss  | Δ Train  | Status")
        print("------|------------|-----------|----------|--------")
        
        for i, (ep, train, val) in enumerate(epochs[-10:]):  # Last 10 epochs
            train_f = float(train)
            val_f = float(val)
            
            if i > 0:
                prev_train = float(epochs[max(0, len(epochs)-10+i-1)][1])
                delta = train_f - prev_train
                trend = "↓" if delta < -0.01 else "↑" if delta > 0.01 else "→"
                change = f"{delta:+.3f}"
            else:
                trend, change = "-", "-"
            
            status = "✅" if val_f < 1.0 else "→"
            print(f"{ep:>5} | {train:>10s} | {val:>9s} | {change:>8s} {trend} | {status}")
    
    # Component analysis
    components = re.findall(r'noise=([\d\.]+), bond=([\d\.]+), angle=([\d\.]+), sphere=([\d\.]+)', content)
    
    if len(components) > 70:
        batches_per_epoch = 71
        num_epochs = len(components) // batches_per_epoch
        
        print(f"\n🔬 Component Trends (per epoch average):\n")
        print("Epoch | Noise  | Bond   | Angle  | Sphere | Bond×λ  | Physics Total")
        print("------|--------|--------|--------|--------|---------|---------------")
        
        lambda_bond = 100.0  # Current config
        lambda_angle = 5.0
        lambda_sphere = 2.0
        
        for ep_idx in range(min(num_epochs, 10)):
            start = ep_idx * batches_per_epoch
            end = start + batches_per_epoch
            epoch_comps = components[start:end]
            
            if epoch_comps:
                avg_noise = np.mean([float(c[0]) for c in epoch_comps])
                avg_bond = np.mean([float(c[1]) for c in epoch_comps])
                avg_angle = np.mean([float(c[2]) for c in epoch_comps])
                avg_sphere = np.mean([float(c[3]) for c in epoch_comps])
                
                bond_contrib = avg_bond * lambda_bond
                physics_total = bond_contrib + avg_angle * lambda_angle + avg_sphere * lambda_sphere
                
                print(f"{ep_idx+1:>5} | {avg_noise:.4f} | {avg_bond:.4f} | {avg_angle:.4f} | {avg_sphere:.4f} | {bond_contrib:>7.2f} | {physics_total:>13.2f}")
        
        # Bond loss trend analysis
        print(f"\n📉 CRITICAL: Bond Loss Trend Analysis:\n")
        
        first_epoch_bond = np.mean([float(c[1]) for c in components[:batches_per_epoch]])
        
        if num_epochs >= 5:
            mid_epoch_bond = np.mean([float(c[1]) for c in components[4*batches_per_epoch:5*batches_per_epoch]])
            reduction_mid = ((first_epoch_bond - mid_epoch_bond) / first_epoch_bond) * 100
            print(f"  Epoch 1:  {first_epoch_bond:.4f}")
            print(f"  Epoch 5:  {mid_epoch_bond:.4f}")
            print(f"  Change:   {reduction_mid:+.1f}%")
            
            if reduction_mid > 10:
                print(f"  ✅ GOOD: Bond loss decreasing significantly!")
            elif reduction_mid > 3:
                print(f"  ⚠️  MODERATE: Slow decrease")
            elif reduction_mid > -3:
                print(f"  ❌ STAGNANT: No meaningful change")
            else:
                print(f"  ❌ INCREASING: Model not learning!")
        
        if num_epochs >= 10:
            last_epoch_bond = np.mean([float(c[1]) for c in components[9*batches_per_epoch:10*batches_per_epoch]])
            reduction_total = ((first_epoch_bond - last_epoch_bond) / first_epoch_bond) * 100
            
            print(f"\n  Epoch 10: {last_epoch_bond:.4f}")
            print(f"  Total change: {reduction_total:+.1f}%")
            
            if abs(reduction_total) < 5:
                print(f"\n  🚨 WARNING: Minimal progress after 10 epochs!")
                print(f"     → Lambda may still be too weak")
                print(f"     → Consider lambda_bond: 200-500")
    
    else:
        print("  (Insufficient data for component analysis)")
    
    print(f"\n{'='*90}\n")
    
    # Latest status
    if epochs:
        latest_ep = epochs[-1][0]
        print(f"Latest: Epoch {latest_ep}/150 running...")
        print(f"Check progress: tail -f {log_file}")
    
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Find latest log
        logs = sorted(Path('.').glob('training_v4_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            analyze_training_log(logs[0])
        else:
            print("No training logs found. Usage: python monitor_training.py <log_file>")
    else:
        analyze_training_log(sys.argv[1])
