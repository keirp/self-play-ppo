"""Experiment 19: Multi-seed runs to find the highest peak.

Run the best config (opp_temp=1.5) 3 times with different seeds.
Take the best peak checkpoint and do deterministic eval.

Also try: gpi=4096 + ppo_epochs=2 + opp_temp=1.5
(same gradient steps as baseline but 2x data diversity per iter)
"""
import subprocess
import sys
import time
import json
import os
import numpy as np


def run_one(name, overrides, num_iters, eval_interval=250):
    os.makedirs(f"experiments/{name}", exist_ok=True)
    log_path = f"experiments/{name}/train.log"
    cmd = [
        sys.executable, "experiment_runner.py",
        name, str(num_iters), json.dumps(overrides),
        "--eval-interval", str(eval_interval),
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
        iters_list = data["metrics"]["eval_iteration"]
        best_elo = max(elos)
        best_idx = elos.index(best_elo)
        print(f"Done in {elapsed/60:.1f}m | Peak: {best_elo:.0f} @ iter {iters_list[best_idx]} | Avg5: {np.mean(elos[-5:]):.0f}")
        return best_elo
    except Exception as e:
        print(f"ERROR: {e}")
        return 0


best_overall = 0
best_name = ""

# Run 3 seeds with baseline best config
for seed_idx in range(3):
    name = f"exp19_seed{seed_idx}"
    elo = run_one(name, {
        "games_per_iter": 2048,
        "batch_size": 256,
        "opp_temperature": 1.5,
        "elo_games_per_opp": 100,
    }, 7000, eval_interval=250)
    if elo > best_overall:
        best_overall = elo
        best_name = name

# Try gpi=4096 + ppo_epochs=2 + opp_temp=1.5
name = "exp19_gpi4096_ppo2_temp15"
elo = run_one(name, {
    "games_per_iter": 4096,
    "batch_size": 256,
    "ppo_epochs": 2,
    "opp_temperature": 1.5,
    "elo_games_per_opp": 100,
}, 7000, eval_interval=250)
if elo > best_overall:
    best_overall = elo
    best_name = name

print(f"\n{'='*60}")
print(f"BEST: {best_name} with peak Elo {best_overall:.0f}")

# Deterministic eval of the best checkpoint
if best_overall > 1900:
    print("\n--- High-precision deterministic eval ---")
    from src.elo import compute_elo
    best_params = np.load(f"experiments/{best_name}/best_params.npy")
    result = compute_elo(best_params, games_per_opponent=200, deterministic=True)
    print(f"Deterministic (200g/opp): {result['elo']:.0f}")
    for opp in result["per_opponent"]:
        total = opp["wins"] + opp["draws"] + opp["losses"]
        wr = (opp["wins"] + 0.5 * opp["draws"]) / total
        print(f"  {opp['name']:>12s} (Elo {opp['ref_elo']:4.0f}): {wr:.1%}")
