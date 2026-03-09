"""Experiment 13: gpi=4096 with bs=256 for maximum gradient steps per iter.

Previous gpi=4096 attempt (exp9) used bs=1024 (proportional scaling) which
halved gradient steps and performed worse. Here we keep bs=256 to get ~4x
more gradient steps than baseline gpi=2048. Each iteration is slower but
each update uses much more diverse data.

Run for 10k iters with 100 games/opp eval every 500 iters.
"""
import subprocess
import sys
import time
import json
import os
import numpy as np

name = "exp13_gpi4096_bs256"
overrides = {
    "games_per_iter": 4096,
    "batch_size": 256,
    "elo_games_per_opp": 100,
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
    iters = data["metrics"]["eval_iteration"]
    print(f"\nDone in {elapsed/60:.1f}m | Final: {elos[-1]:.0f} | Avg5: {np.mean(elos[-5:]):.0f} | Peak: {max(elos):.0f}")
    for it, elo, h in zip(iters, elos, hw):
        print(f"  iter {it:5d}: Elo {elo:.0f} | HeurW {h:.1%}")
except Exception as e:
    print(f"ERROR: {e}")
