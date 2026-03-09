"""Experiment 8: Long run gpi2048 (15k iters).

Goal: See how high Elo goes with more training.
If it plateaus, we know we need something else.
"""
import subprocess
import sys
import time
import json
import os
import numpy as np

name = "exp8_gpi2048_15k"
overrides = {
    "games_per_iter": 2048,
    "batch_size": 512,
    "lr": 3e-4,
    "elo_games_per_opp": 40,  # more games for stable Elo
}
NUM_ITERS = 15000

os.makedirs(f"experiments/{name}", exist_ok=True)
log_path = f"experiments/{name}/train.log"

cmd = [
    sys.executable, "experiment_runner.py",
    name, str(NUM_ITERS), json.dumps(overrides),
    "--eval-interval", "500",
]

print(f"Starting {name} ({NUM_ITERS} iters)...")
print(f"Log: {log_path}")

t0 = time.time()
with open(log_path, "w") as lf:
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    p.wait()

elapsed = time.time() - t0
print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

try:
    with open(f"experiments/{name}/results.json") as f:
        data = json.load(f)
    elos = data["metrics"]["elo"]
    hw = data["metrics"]["vs_heuristic_win_rate"]
    iters = data["metrics"]["eval_iteration"]
    print(f"\nFinal Elo: {elos[-1]:.0f}")
    print(f"Avg(last5): {np.mean(elos[-5:]):.0f}")
    print(f"Peak: {max(elos):.0f}")
    print(f"HeurW(last5): {np.mean(hw[-5:]):.1%}")
    print(f"\nElo trajectory:")
    for it, elo in zip(iters, elos):
        print(f"  iter {it:5d}: {elo:.0f}")
except Exception as e:
    print(f"ERROR: {e}")
