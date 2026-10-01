#!/usr/bin/env python3
"""Quick script to check PaiNN+FM training progress."""
import re, os, sys

LOG = os.path.join(os.path.dirname(__file__), 'training_painn_fm.log')
if not os.path.exists(LOG):
    print("No log file found at", LOG)
    sys.exit(1)

data = open(LOG, 'rb').read()
lines = re.split(rb'[\r\n]+', data)

val_lines, best_lines, status_lines = [], [], []
last_epoch = ''
epoch_losses = []

for line in lines:
    t = line.decode('utf-8', errors='replace').strip()
    if 'INFO' in t and 'val=' in t:
        val_lines.append(t)
    if 'New best' in t:
        best_lines.append(t)
    if '100%' in t and 'Epoch' in t and 'noise=' in t:
        last_epoch = t
        # Extract epoch number and loss
        m = re.search(r'Epoch (\d+).*loss=([0-9.]+).*noise=([0-9.]+)', t)
        if m:
            epoch_losses.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if any(k in t for k in ['Training complete', 'Traceback', 'Total NaN', 'Best val_loss']):
        status_lines.append(t)

print("=" * 60)
print("PaiNN + Flow Matching Training Status")
print("=" * 60)

print("\n--- Epoch Loss Trend (last batch per epoch) ---")
# Deduplicate - keep last entry per epoch
seen = {}
for ep, loss, noise in epoch_losses:
    seen[ep] = (loss, noise)
for ep in sorted(seen.keys()):
    loss, noise = seen[ep]
    print(f"  Epoch {ep:3d}: loss={loss:.4f}  noise={noise:.4f}")

print(f"\n--- Validation History ({len(val_lines)} entries) ---")
for v in val_lines:
    print(f"  {v}")

print(f"\n--- Best Model Saves ({len(best_lines)} entries) ---")
for b in best_lines:
    print(f"  {b}")

print("\n--- Completion Status ---")
for s in status_lines:
    print(f"  {s[:250]}")
if not status_lines:
    print("  Training not yet complete (no completion/error message found)")

# Check checkpoint files
ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', 'painn_fm')
if os.path.isdir(ckpt_dir):
    files = sorted(os.listdir(ckpt_dir))
    print(f"\n--- Checkpoint Files ({len(files)}) ---")
    for f in files:
        sz = os.path.getsize(os.path.join(ckpt_dir, f))
        print(f"  {f} ({sz/1024/1024:.1f} MB)")

print("=" * 60)
