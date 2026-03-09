"""Experiment 15: Entropy schedule + multi-opponent approaches.

Three approaches to break the 1890 Elo plateau:
a) Entropy schedule: ent_coef=0.01 cosine decay to 0.001 (more exploration early)
b) Multi-opponent: 4 opponents per iteration (more diverse training data)
c) Combined: both entropy schedule + multi-opponent

All use gpi=2048, bs=256, 100 games/opp eval, 10k iters.
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


# a) Entropy schedule: high early, low late
run_one("exp15a_ent_schedule", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "ent_coef": 0.01,       # 10x baseline
    "ent_schedule": "cosine",
    "ent_min_frac": 0.1,     # decays to 0.001
    "elo_games_per_opp": 100,
}, 10000)

# b) Multi-opponent per iteration (4 opponents, 512 games each)
run_one("exp15b_multi_opp", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "num_opps_per_iter": 4,
    "elo_games_per_opp": 100,
}, 10000)

# c) Combined
run_one("exp15c_combined", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "ent_coef": 0.01,
    "ent_schedule": "cosine",
    "ent_min_frac": 0.1,
    "num_opps_per_iter": 4,
    "elo_games_per_opp": 100,
}, 10000)
