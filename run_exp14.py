"""Experiment 14: Long baseline run + ppo_epochs=2 variant.

Two experiments run sequentially:
1. exp14_long20k: baseline gpi=2048 for 20k iters (our best config, just longer)
2. exp14b_ppo2: gpi=2048 with ppo_epochs=2 for 10k iters (fewer gradient steps per iter)

Both use 100 games/opp for accurate Elo measurement.
"""
import subprocess
import sys
import time
import json
import os
import numpy as np


def run_one(name, overrides, num_iters):
    os.makedirs(f"experiments/{name}", exist_ok=True)
    log_path = f"experiments/{name}/train.log"
    cmd = [
        sys.executable, "experiment_runner.py",
        name, str(num_iters), json.dumps(overrides),
        "--eval-interval", "500",
    ]
    print(f"\nStarting {name} ({num_iters} iters)...", flush=True)
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
        iters_list = data["metrics"]["eval_iteration"]
        print(f"Done in {elapsed/60:.1f}m | Final: {elos[-1]:.0f} | Avg5: {np.mean(elos[-5:]):.0f} | Peak: {max(elos):.0f}")
        for it, elo, h in zip(iters_list, elos, hw):
            print(f"  iter {it:5d}: Elo {elo:.0f} | HeurW {h:.1%}")
    except Exception as e:
        print(f"ERROR: {e}")


# 1. Long baseline
run_one("exp14_long20k", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "elo_games_per_opp": 100,
}, 20000)

# 2. PPO epochs = 2
run_one("exp14b_ppo2", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "ppo_epochs": 2,
    "elo_games_per_opp": 100,
}, 10000)
