"""Experiment 17: Best combined approaches.

a) opp_temp annealing 1.5→1.0: curriculum learning (diverse→precise opponents)
b) opp_temp=1.5 constant + multi_opp=2: diversity + stronger per-opp signal
c) opp_temp=1.5 + cosine LR: combine two small improvements

All use gpi=2048, bs=256, 100 games/opp eval, 10k iters.
Best checkpoint saving is now active.
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
        best_elo = max(elos)
        best_idx = elos.index(best_elo)
        print(f"Done in {elapsed/60:.1f}m | Final: {elos[-1]:.0f} | Avg5: {np.mean(elos[-5:]):.0f} | Peak: {best_elo:.0f} @ iter {iters_list[best_idx]}")
        for it, elo, h in zip(iters_list, elos, hw):
            print(f"  iter {it:5d}: Elo {elo:.0f} | HeurW {h:.1%}")
    except Exception as e:
        print(f"ERROR: {e}")


# a) Opponent temperature annealing: 1.5 → 1.0 (curriculum)
run_one("exp17a_temp_anneal", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "opp_temperature": 1.5,
    "opp_temp_schedule": "cosine",
    "elo_games_per_opp": 100,
}, 10000)

# b) opp_temp=1.5 + 2 opponents per iter
run_one("exp17b_temp15_2opp", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "opp_temperature": 1.5,
    "num_opps_per_iter": 2,
    "elo_games_per_opp": 100,
}, 10000)

# c) opp_temp=1.5 + cosine LR
run_one("exp17c_temp15_cosine", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "opp_temperature": 1.5,
    "lr_schedule": "cosine",
    "lr_min_frac": 0.1,
    "elo_games_per_opp": 100,
}, 10000)
