"""Experiment 12c: Mirror augmentation with conservative clipping.

Mirror + clip_eps=0.1 to prevent KL explosion from asymmetric transitions.
"""
import subprocess
import sys
import time
import json
import os
import numpy as np

name = "exp12c_mirror_clip01"
overrides = {
    "games_per_iter": 2048,
    "batch_size": 256,
    "mirror_augment": True,
    "clip_eps": 0.1,
    "elo_games_per_opp": 100,
}
NUM_ITERS = 5000

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
