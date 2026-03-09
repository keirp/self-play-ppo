"""Experiment 18: Best config (temp=1.5) with checkpoint saving + deterministic eval.

Run the best-performing config for 10k iters, save best checkpoint,
then do high-precision deterministic evaluation (200 games/opp).

Deterministic eval was ~20-50 Elo higher than stochastic in previous tests.
If peak stochastic is 1970, deterministic might be 2020+.
"""
import subprocess
import sys
import time
import json
import os
import numpy as np

name = "exp18_temp15_best"
overrides = {
    "games_per_iter": 2048,
    "batch_size": 256,
    "opp_temperature": 1.5,
    "elo_games_per_opp": 100,
}
NUM_ITERS = 10000

os.makedirs(f"experiments/{name}", exist_ok=True)
log_path = f"experiments/{name}/train.log"
cmd = [
    sys.executable, "experiment_runner.py",
    name, str(NUM_ITERS), json.dumps(overrides),
    "--eval-interval", "250",  # More frequent evals to catch the peak
]

print(f"Starting {name} ({NUM_ITERS} iters, eval every 250)...", flush=True)

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
    print(f"\nDone in {elapsed/60:.1f}m | Final: {elos[-1]:.0f} | Avg5: {np.mean(elos[-5:]):.0f} | Peak: {best_elo:.0f} @ iter {iters_list[best_idx]}")
    for it, elo, h in zip(iters_list, elos, hw):
        print(f"  iter {it:5d}: Elo {elo:.0f} | HeurW {h:.1%}")
except Exception as e:
    print(f"ERROR: {e}")

# High-precision deterministic eval of best checkpoint
print("\n--- High-precision eval of best checkpoint ---", flush=True)
try:
    from src.elo import compute_elo
    best_params = np.load(f"experiments/{name}/best_params.npy")

    # Stochastic eval (200 games/opp)
    result_stoch = compute_elo(best_params, games_per_opponent=200, deterministic=False)
    print(f"Stochastic (200 g/opp): Elo {result_stoch['elo']:.0f}")

    # Deterministic eval (200 games/opp)
    result_det = compute_elo(best_params, games_per_opponent=200, deterministic=True)
    print(f"Deterministic (200 g/opp): Elo {result_det['elo']:.0f}")

    # Print per-opponent results
    print("\nPer-opponent (deterministic):")
    for opp in result_det["per_opponent"]:
        total = opp["wins"] + opp["draws"] + opp["losses"]
        wr = (opp["wins"] + 0.5 * opp["draws"]) / total
        print(f"  {opp['name']:>12s} (Elo {opp['ref_elo']:4.0f}): {wr:.1%} ({opp['wins']}W/{opp['draws']}D/{opp['losses']}L)")
except Exception as e:
    print(f"Eval ERROR: {e}")
