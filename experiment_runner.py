"""Run Connect 4 experiments with different configs.

Usage:
    python experiment_runner.py <experiment_name> <num_iters> <config_json> [--eval-interval N] [--eval-games N]

Example:
    python experiment_runner.py ent_001 2000 '{"ent_coef": 0.01}'
"""

import json
import sys
import time
import os
import numpy as np
from src.connect4_c import Connect4TrainerC

# Default config
BASE_CONFIG = {
    "hidden_size": 256,
    "num_layers": 6,
    "games_per_iter": 512,
    "lr": 3e-4,
    "ent_coef": 0.001,
    "batch_size": 256,
    "ppo_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "draw_reward": 0.0,
    "opponent_pool_max": 50,
    "snapshot_interval": 25,
    "opponent_sampling": "uniform",
}


def run_experiment(name, num_iters, config_overrides, eval_interval=100, elo_games=20):
    config = {**BASE_CONFIG, **config_overrides}
    out_dir = f"experiments/{name}"
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    trainer = Connect4TrainerC(config)

    t0 = time.time()
    metrics = trainer.train(num_iters, eval_interval=eval_interval, verbose=True)
    wall_time = time.time() - t0

    # Save results
    results = {
        "metrics": metrics,
        "wall_time": wall_time,
        "total_params": trainer.total_params,
        "config": config,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f)

    # Save final params
    np.save(os.path.join(out_dir, "final_params.npy"), trainer.params)

    # Save best checkpoint if available
    if trainer.best_params is not None:
        np.save(os.path.join(out_dir, "best_params.npy"), trainer.best_params)
        print(f"  Best Elo: {trainer.best_elo:.0f} (saved best_params.npy)")

    # Print summary
    elos = metrics.get("elo", [])
    if elos:
        last_n = min(10, len(elos))
        avg_elo = np.mean(elos[-last_n:])
        peak_elo = max(elos)
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"  Final Elo: {elos[-1]:.0f}")
        print(f"  Avg Elo (last {last_n} evals): {avg_elo:.0f}")
        print(f"  Peak Elo: {peak_elo:.0f}")
        print(f"  Wall time: {wall_time:.0f}s ({wall_time/60:.1f} min)")
        print(f"{'='*60}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    name = sys.argv[1]
    num_iters = int(sys.argv[2])
    config_overrides = json.loads(sys.argv[3])

    eval_interval = 100
    for i, arg in enumerate(sys.argv):
        if arg == "--eval-interval" and i + 1 < len(sys.argv):
            eval_interval = int(sys.argv[i + 1])

    run_experiment(name, num_iters, config_overrides, eval_interval=eval_interval)
