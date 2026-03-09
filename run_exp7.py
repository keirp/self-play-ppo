"""Experiment 7: Batch size with gpi2048 (5k iters each)."""
import subprocess
import sys
import time
import json
import os
import numpy as np

configs = [
    ("exp7_bs256", {"games_per_iter": 2048, "batch_size": 256}),
    ("exp7_bs1024", {"games_per_iter": 2048, "batch_size": 1024}),
    ("exp7_bs2048", {"games_per_iter": 2048, "batch_size": 2048}),
]

NUM_ITERS = 5000

procs = []
log_files = []
t0 = time.time()

for name, overrides in configs:
    cmd = [
        sys.executable, "experiment_runner.py",
        name, str(NUM_ITERS), json.dumps(overrides),
        "--eval-interval", "250",
    ]
    os.makedirs(f"experiments/{name}", exist_ok=True)
    log_path = f"experiments/{name}/train.log"
    lf = open(log_path, "w")
    print(f"Starting {name}... (log: {log_path})")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    procs.append((name, p))
    log_files.append(lf)

# Wait for all to finish
for (name, p), lf in zip(procs, log_files):
    p.wait()
    lf.close()
    print(f"{name} finished (exit code {p.returncode})")

elapsed = time.time() - t0
print(f"All done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

print("\n" + "="*80)
print(f"{'Config':<20} {'Final Elo':>10} {'Avg5':>8} {'Peak':>8} {'HeurW':>8} {'Time':>8}")
print("="*80)
for name, _ in configs:
    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        hw = data["metrics"]["vs_heuristic_win_rate"]
        wt = data["wall_time"]
        print(f"{name:<20} {elos[-1]:>10.0f} {np.mean(elos[-5:]):>8.0f} {max(elos):>8.0f} {np.mean(hw[-5:]):>8.1%} {wt/60:>7.0f}m")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")
