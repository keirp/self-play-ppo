"""Run AlphaStar-style league training for Connect 4.

Compares:
1. League training (Main Agent + Main Exploiter + League Exploiter)
2. Baseline self-play (same compute budget per main agent iteration)
"""
import sys
import os
import time
import json
import numpy as np

from src.league import LeagueTrainer
from src.connect4_c import Connect4TrainerC


def run_league(num_iters, out_dir):
    """Run league training."""
    config = {
        "hidden_size": 256,
        "num_layers": 6,
        "lr": 3e-4,
        "ent_coef": 0.001,
        "batch_size": 256,
        "clip_eps": 0.2,
        "games_per_iter": 2048,
        "opp_temperature": 1.5,
        "elo_games_per_opp": 100,
        # League-specific
        "snapshot_interval": 50,
        "pfsp_p": 2.0,
        "self_play_frac": 0.35,
        "exploiter_reset_timeout": 500,
        "exploiter_reset_threshold": 0.7,
    }

    trainer = LeagueTrainer(config)
    result = trainer.train(
        num_iterations=num_iters,
        eval_interval=250,
        out_dir=out_dir,
    )
    return result


def run_baseline(num_iters, out_dir):
    """Run baseline self-play with the same config."""
    config = {
        "hidden_size": 256,
        "num_layers": 6,
        "lr": 3e-4,
        "ent_coef": 0.001,
        "batch_size": 256,
        "clip_eps": 0.2,
        "games_per_iter": 2048,
        "opp_temperature": 1.5,
        "elo_games_per_opp": 100,
        "opponent_pool_max": 50,
        "snapshot_interval": 25,
    }

    os.makedirs(out_dir, exist_ok=True)

    trainer = Connect4TrainerC(config)
    metrics = trainer.train(
        num_iterations=num_iters,
        eval_interval=250,
    )

    # Save results
    np.save(os.path.join(out_dir, "final_params.npy"), trainer.params)
    if trainer.best_params is not None:
        np.save(os.path.join(out_dir, "best_params.npy"), trainer.best_params)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"metrics": metrics}, f)

    return {
        "best_elo": trainer.best_elo,
        "metrics": metrics,
    }


if __name__ == "__main__":
    num_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    print("=" * 60)
    print(f"LEAGUE TRAINING ({num_iters} iters)")
    print("=" * 60)
    t0 = time.time()
    league_result = run_league(num_iters, "experiments/league_v1")
    league_time = time.time() - t0
    print(f"\nLeague training done in {league_time/60:.1f}m")
    print(f"League best Elo: {league_result['best_elo']:.0f}")

    # Note: baseline uses 3x less total compute since league trains 3 agents
    # For fair comparison, we run baseline with same number of main agent iterations
    print("\n" + "=" * 60)
    print(f"BASELINE SELF-PLAY ({num_iters} iters, same compute per agent)")
    print("=" * 60)
    t0 = time.time()
    baseline_result = run_baseline(num_iters, "experiments/league_baseline")
    baseline_time = time.time() - t0
    print(f"\nBaseline done in {baseline_time/60:.1f}m")
    print(f"Baseline best Elo: {baseline_result['best_elo']:.0f}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"League best Elo:   {league_result['best_elo']:.0f}")
    print(f"Baseline best Elo: {baseline_result['best_elo']:.0f}")
    print(f"Improvement:       {league_result['best_elo'] - baseline_result['best_elo']:.0f}")
