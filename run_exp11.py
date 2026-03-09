"""Experiment 11: High-precision long run.

Key insight: the ~1830 Elo with 40 games/opp becomes ~1890 with 100 games/opp.
The "oscillation" is partly eval noise.

Run baseline config for 10k iters with 100 games/opp evals to get TRUE trajectory.
Also save checkpoints for later evaluation.
"""
import subprocess
import sys
import time
import json
import os
import numpy as np

name = "exp11_precision"
overrides = {
    "games_per_iter": 2048,
    "batch_size": 256,
    "elo_games_per_opp": 100,  # high precision
}
NUM_ITERS = 10000

os.makedirs(f"experiments/{name}", exist_ok=True)
log_path = f"experiments/{name}/train.log"

cmd = [
    sys.executable, "experiment_runner.py",
    name, str(NUM_ITERS), json.dumps(overrides),
    "--eval-interval", "500",
]

print(f"Starting {name} ({NUM_ITERS} iters)...", flush=True)
print(f"Log: {log_path}", flush=True)
print(f"Config: gpi=2048, bs=256, clip=0.2, pool=50, 100 games/opp evals", flush=True)

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
    for it, elo, h in zip(iters, elos, hw):
        print(f"  iter {it:5d}: Elo {elo:.0f} | HeurW {h:.1%}")
except Exception as e:
    print(f"ERROR: {e}")
