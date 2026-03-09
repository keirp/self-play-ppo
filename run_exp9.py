"""Experiment 9: Breaking the 1830 plateau.

Key learning: gpi=4096 with bs=256 is too slow (2.3s/iter from 800 grad steps).
Use bs proportional to data for consistent iter time.

Three approaches, run sequentially:
1. gpi=4096 bs=1024 — double data, proportional batch size
2. gpi=2048 cosine LR — anneal LR to stabilize late training
3. gpi=2048 lr=1e-3 — higher LR for faster adaptation

Each for 5k iters with eval every 250.
"""
import subprocess
import sys
import time
import json
import os
import numpy as np

configs = [
    ("exp9_gpi4096", {"games_per_iter": 4096, "batch_size": 1024, "elo_games_per_opp": 40}),
    ("exp9_cosine_lr", {"games_per_iter": 2048, "batch_size": 256, "lr": 3e-4, "lr_schedule": "cosine", "lr_min_frac": 0.1, "elo_games_per_opp": 40}),
    ("exp9_lr1e3", {"games_per_iter": 2048, "batch_size": 256, "lr": 1e-3, "elo_games_per_opp": 40}),
]

NUM_ITERS = 5000

for name, overrides in configs:
    os.makedirs(f"experiments/{name}", exist_ok=True)
    log_path = f"experiments/{name}/train.log"
    cmd = [
        sys.executable, "experiment_runner.py",
        name, str(NUM_ITERS), json.dumps(overrides),
        "--eval-interval", "250",
    ]
    print(f"\nStarting {name} ({NUM_ITERS} iters)...", flush=True)
    print(f"Log: {log_path}", flush=True)

    t0 = time.time()
    with open(log_path, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        p.wait()
    elapsed = time.time() - t0

    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        hw = data["metrics"]["vs_heuristic_win_rate"]
        wt = data["wall_time"]
        print(f"  Done in {elapsed/60:.1f}m | Final: {elos[-1]:.0f} | Avg5: {np.mean(elos[-5:]):.0f} | Peak: {max(elos):.0f} | HeurW: {np.mean(hw[-5:]):.1%}", flush=True)
        print(f"  Elo trajectory: {[f'{e:.0f}' for e in elos]}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

print("\n" + "="*80)
print(f"{'Config':<20} {'Final Elo':>10} {'Avg5':>8} {'Peak':>8} {'HeurW':>8}")
print("="*80)
for name, _ in configs:
    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        hw = data["metrics"]["vs_heuristic_win_rate"]
        print(f"{name:<20} {elos[-1]:>10.0f} {np.mean(elos[-5:]):>8.0f} {max(elos):>8.0f} {np.mean(hw[-5:]):>8.1%}")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")
