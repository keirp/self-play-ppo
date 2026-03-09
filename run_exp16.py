"""Experiment 16: Opponent temperature for training diversity.

Higher opponent temperature makes the training opponent play more randomly,
creating diverse board states the agent must handle. This directly addresses
the self-play narrowness problem.

a) opp_temperature=1.5 (moderate diversity)
b) opp_temperature=2.0 (high diversity)
c) Larger model h=384 with gpi=2048

All use 100 games/opp eval, 10k iters.
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


# a) Opponent temperature 1.5
run_one("exp16a_opp_temp15", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "opp_temperature": 1.5,
    "elo_games_per_opp": 100,
}, 10000)

# b) Opponent temperature 2.0
run_one("exp16b_opp_temp20", {
    "games_per_iter": 2048,
    "batch_size": 256,
    "opp_temperature": 2.0,
    "elo_games_per_opp": 100,
}, 10000)
