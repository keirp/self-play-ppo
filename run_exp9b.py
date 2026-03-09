"""Experiment 9b: Breaking the 1830 plateau — focused experiments.

Key insight: gradient updates per data point > raw data volume.
gpi=2048 bs=256 gets ~234 updates/iter, which is the proven sweet spot.

Now testing changes to optimization dynamics:
1. Cosine LR decay — reduce oscillation in late training
2. More PPO epochs (8) — extract more from each batch
3. Higher LR (1e-3) — faster adaptation

Each for 5k iters (eval every 250) to compare against baseline avg ~1830.
"""
import subprocess
import sys
import time
import json
import os
import numpy as np

configs = [
    ("exp9b_cosine_lr", {
        "games_per_iter": 2048, "batch_size": 256,
        "lr": 3e-4, "lr_schedule": "cosine", "lr_min_frac": 0.1,
        "elo_games_per_opp": 40,
    }),
    ("exp9b_ppo8", {
        "games_per_iter": 2048, "batch_size": 256,
        "ppo_epochs": 8,
        "elo_games_per_opp": 40,
    }),
    ("exp9b_lr1e3", {
        "games_per_iter": 2048, "batch_size": 256,
        "lr": 1e-3,
        "elo_games_per_opp": 40,
    }),
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
        print(f"  Done in {elapsed/60:.1f}m | Final: {elos[-1]:.0f} | Avg5: {np.mean(elos[-5:]):.0f} | Peak: {max(elos):.0f} | HeurW: {np.mean(hw[-5:]):.1%}", flush=True)
        print(f"  Elo: {[f'{e:.0f}' for e in elos]}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

print("\n" + "="*80)
print(f"{'Config':<20} {'Final':>8} {'Avg5':>8} {'Peak':>8} {'HeurW':>8}")
print("="*80)
for name, _ in configs:
    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        hw = data["metrics"]["vs_heuristic_win_rate"]
        print(f"{name:<20} {elos[-1]:>8.0f} {np.mean(elos[-5:]):>8.0f} {max(elos):>8.0f} {np.mean(hw[-5:]):>8.1%}")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")
