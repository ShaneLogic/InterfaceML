"""
Real-time training monitor for fullerene diffusion model.

Tracks loss trends, component analysis, and generates progress reports.

Usage:
    python monitor_training.py training_v4_high_lambda_*.log
"""

import logging
import sys
import re
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def analyze_training_log(log_file):
    """Analyze training log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    logger.info("=" * 90)
    logger.info(" TRAINING MONITOR: %s", Path(log_file).name)
    logger.info("=" * 90)
    logger.info("")
    
    # Extract epoch summaries
    epochs = re.findall(r'Epoch (\d+)/\d+\s+Train loss: ([\d\.]+)\s+Val loss: ([\d\.]+)', content)
    
    if epochs:
        logger.info("📊 Epoch Progress (%d epochs completed):", len(epochs))
        logger.info("Epoch | Train Loss | Val Loss  | Δ Train  | Status")
        logger.info("------|------------|-----------|----------|--------")
        
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
            logger.info("%5s | %10s | %9s | %8s %s | %s", ep, train, val, change, trend, status)
    
    # Component analysis
    components = re.findall(r'noise=([\d\.]+), bond=([\d\.]+), angle=([\d\.]+), sphere=([\d\.]+)', content)
    
    if len(components) > 70:
        batches_per_epoch = 71
        num_epochs = len(components) // batches_per_epoch
        
        logger.info("🔬 Component Trends (per epoch average):")
        logger.info("Epoch | Noise  | Bond   | Angle  | Sphere | Bond×λ  | Physics Total")
        logger.info("------|--------|--------|--------|--------|---------|---------------")
        
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
                
                logger.info("%5d | %.4f | %.4f | %.4f | %.4f | %7.2f | %13.2f", ep_idx+1, avg_noise, avg_bond, avg_angle, avg_sphere, bond_contrib, physics_total)
        
        # Bond loss trend analysis
        logger.info("📉 CRITICAL: Bond Loss Trend Analysis:")
        
        first_epoch_bond = np.mean([float(c[1]) for c in components[:batches_per_epoch]])
        
        if num_epochs >= 5:
            mid_epoch_bond = np.mean([float(c[1]) for c in components[4*batches_per_epoch:5*batches_per_epoch]])
            reduction_mid = ((first_epoch_bond - mid_epoch_bond) / first_epoch_bond) * 100
            logger.info("  Epoch 1:  %.4f", first_epoch_bond)
            logger.info("  Epoch 5:  %.4f", mid_epoch_bond)
            logger.info("  Change:   %+.1f%%", reduction_mid)
            
            if reduction_mid > 10:
                logger.info("  ✅ GOOD: Bond loss decreasing significantly!")
            elif reduction_mid > 3:
                logger.info("  ⚠️  MODERATE: Slow decrease")
            elif reduction_mid > -3:
                logger.info("  ❌ STAGNANT: No meaningful change")
            else:
                logger.info("  ❌ INCREASING: Model not learning!")
        
        if num_epochs >= 10:
            last_epoch_bond = np.mean([float(c[1]) for c in components[9*batches_per_epoch:10*batches_per_epoch]])
            reduction_total = ((first_epoch_bond - last_epoch_bond) / first_epoch_bond) * 100
            
            logger.info("  Epoch 10: %.4f", last_epoch_bond)
            logger.info("  Total change: %+.1f%%", reduction_total)
            
            if abs(reduction_total) < 5:
                logger.warning("  🚨 WARNING: Minimal progress after 10 epochs!")
                logger.warning("     → Lambda may still be too weak")
                logger.warning("     → Consider lambda_bond: 200-500")
    
    else:
        logger.info("  (Insufficient data for component analysis)")
    
    logger.info("=" * 90)
    
    # Latest status
    if epochs:
        latest_ep = epochs[-1][0]
        logger.info("Latest: Epoch %s/150 running...", latest_ep)
        logger.info("Check progress: tail -f %s", log_file)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if len(sys.argv) < 2:
        # Find latest log
        logs = sorted(Path('.').glob('training_v4_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            analyze_training_log(logs[0])
        else:
            logger.info("No training logs found. Usage: python monitor_training.py <log_file>")
    else:
        analyze_training_log(sys.argv[1])
