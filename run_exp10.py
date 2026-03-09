"""Experiment 10: Stability + diversity for pushing past 1900.

Key findings so far:
- Best configs reach ~1890 stochastic Elo (100 games/opp)
- Agent has blind spots vs specific reference pool members (78% vs iter_1400!)
- This is self-play overfitting: agent overfits to its own play style

Strategy:
- clip_eps=0.1 for more conservative policy updates (less oscillation)
- pool_max=200 + snapshot_interval=5 for much more diverse opponents
- Train for 10k iters

Also test: even larger pool (500) and smaller clip (0.05).
"""
import subprocess
import sys
import time
import json
import os
import numpy as np

configs = [
    ("exp10_stable", {
        "games_per_iter": 2048,
        "batch_size": 256,
        "clip_eps": 0.1,
        "opponent_pool_max": 200,
        "snapshot_interval": 5,
        "elo_games_per_opp": 40,
    }),
    ("exp10_very_stable", {
        "games_per_iter": 2048,
        "batch_size": 256,
        "clip_eps": 0.05,
        "opponent_pool_max": 500,
        "snapshot_interval": 3,
        "elo_games_per_opp": 40,
    }),
]

NUM_ITERS = 10000

for name, overrides in configs:
    os.makedirs(f"experiments/{name}", exist_ok=True)
    log_path = f"experiments/{name}/train.log"
    cmd = [
        sys.executable, "experiment_runner.py",
        name, str(NUM_ITERS), json.dumps(overrides),
        "--eval-interval", "500",
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
print(f"{'Config':<25} {'Final':>8} {'Avg5':>8} {'Peak':>8} {'HeurW':>8}")
print("="*80)
for name, _ in configs:
    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        hw = data["metrics"]["vs_heuristic_win_rate"]
        print(f"{name:<25} {elos[-1]:>8.0f} {np.mean(elos[-5:]):>8.0f} {max(elos):>8.0f} {np.mean(hw[-5:]):>8.1%}")
    except Exception as e:
        print(f"{name:<25} ERROR: {e}")
