"""Run self-play PPO experiments with hyperparameter tuning for tic-tac-toe."""

import os
import sys
import json
import time
import copy
import numpy as np
import torch

from src.ppo import SelfPlayTrainer, log
from src.report import plot_training_curves, plot_comparison, generate_pdf_report


RESULTS_DIR = "experiments"
REPORTS_DIR = "reports"


def run_single_experiment(config, name, verbose=True):
    """Run a single training experiment and return metrics."""
    log(f"\n{'='*60}")
    log(f"Experiment: {name}")
    log(f"Config: {json.dumps({k: v for k, v in config.items() if k != 'device'}, indent=2)}")
    log(f"{'='*60}")

    start_time = time.time()

    trainer = SelfPlayTrainer(config)
    num_iterations = config.get("num_iterations", 300)
    eval_interval = config.get("eval_interval", 10)

    metrics = trainer.train(num_iterations, eval_interval=eval_interval, verbose=verbose)

    elapsed = time.time() - start_time
    metrics["training_time"] = elapsed
    log(f"Training completed in {elapsed:.1f}s")

    # Final evaluation
    final_eval = trainer.evaluate()
    metrics["final_eval"] = final_eval
    log(f"Final: vs_random_win={final_eval['vs_random_win_rate']:.3f}, "
        f"vs_optimal_draw={final_eval['vs_optimal_draw_rate']:.3f}, "
        f"vs_optimal_loss={final_eval['vs_optimal_loss_rate']:.3f}")

    return metrics, trainer


def save_experiment(metrics, config, name, exp_dir):
    """Save experiment results and generate plots."""
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    save_config = {k: v for k, v in config.items() if k != "device"}
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(save_config, f, indent=2)

    # Save metrics (convert numpy types)
    save_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            save_metrics[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        elif isinstance(v, dict):
            save_metrics[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()}
        else:
            save_metrics[k] = v
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(save_metrics, f, indent=2)

    # Generate plots
    plots = plot_training_curves(metrics, exp_dir, title_prefix=f"{name}: ")
    return plots


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

BASE_CONFIG = {
    "device": "cpu",
    "hidden_size": 128,
    "num_layers": 3,
    "lr": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "ppo_epochs": 4,
    "batch_size": 64,
    "max_grad_norm": 0.5,
    "games_per_iter": 256,
    "opponent_pool_max": 20,
    "snapshot_interval": 10,
    "opponent_sampling": "uniform",
    "draw_reward": 0.5,
    "num_iterations": 300,
    "eval_interval": 10,
}


def experiment_1_baseline():
    """Experiment 1: Baseline — verify training works with default config."""
    config = copy.deepcopy(BASE_CONFIG)
    return {"baseline": config}


def experiment_2_learning_rate():
    """Experiment 2: Learning rate sweep."""
    configs = {}
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        c = copy.deepcopy(BASE_CONFIG)
        c["lr"] = lr
        configs[f"lr={lr}"] = c
    return configs


def experiment_3_entropy_coef():
    """Experiment 3: Entropy coefficient sweep."""
    configs = {}
    for ent in [0.0, 0.005, 0.01, 0.05, 0.1]:
        c = copy.deepcopy(BASE_CONFIG)
        c["ent_coef"] = ent
        configs[f"ent={ent}"] = c
    return configs


def experiment_4_opponent_sampling():
    """Experiment 4: Opponent sampling strategy."""
    configs = {}
    for strategy in ["uniform", "latest", "weighted_recent"]:
        c = copy.deepcopy(BASE_CONFIG)
        c["opponent_sampling"] = strategy
        configs[f"opp={strategy}"] = c
    return configs


def experiment_5_network_size():
    """Experiment 5: Network architecture."""
    configs = {}
    for hs, nl in [(64, 2), (128, 3), (256, 4)]:
        c = copy.deepcopy(BASE_CONFIG)
        c["hidden_size"] = hs
        c["num_layers"] = nl
        configs[f"h{hs}_l{nl}"] = c
    return configs


def experiment_6_games_per_iter():
    """Experiment 6: Games per iteration (sample efficiency)."""
    configs = {}
    for gpi in [64, 128, 256, 512]:
        c = copy.deepcopy(BASE_CONFIG)
        c["games_per_iter"] = gpi
        configs[f"gpi={gpi}"] = c
    return configs


def experiment_7_clip_epsilon():
    """Experiment 7: PPO clip epsilon."""
    configs = {}
    for eps in [0.1, 0.2, 0.3]:
        c = copy.deepcopy(BASE_CONFIG)
        c["clip_eps"] = eps
        configs[f"clip={eps}"] = c
    return configs


def experiment_8_snapshot_interval():
    """Experiment 8: How often to snapshot to opponent pool."""
    configs = {}
    for si in [5, 10, 25, 50]:
        c = copy.deepcopy(BASE_CONFIG)
        c["snapshot_interval"] = si
        configs[f"snap={si}"] = c
    return configs


def experiment_9_final_tuned():
    """Experiment 9: Final tuned config with best params from prior experiments.
    Best findings: lr=3e-3, ent=0.05, h256_l4, gpi=512, clip=0.1, snap=25, uniform sampling.
    """
    configs = {}

    # Tuned config combining best hyperparameters
    c = copy.deepcopy(BASE_CONFIG)
    c["lr"] = 3e-3
    c["ent_coef"] = 0.05
    c["hidden_size"] = 256
    c["num_layers"] = 4
    c["games_per_iter"] = 512
    c["clip_eps"] = 0.1
    c["snapshot_interval"] = 25
    c["opponent_sampling"] = "uniform"
    c["num_iterations"] = 500
    configs["tuned"] = c

    # Also run baseline at 500 iters for comparison
    b = copy.deepcopy(BASE_CONFIG)
    b["num_iterations"] = 500
    configs["baseline_500"] = b

    return configs


def run_experiment_suite(experiment_fn, suite_name, report_title):
    """Run all configs in an experiment and generate a PDF report."""
    configs = experiment_fn()
    all_metrics = []
    all_labels = []
    all_final = []

    suite_dir = os.path.join(RESULTS_DIR, suite_name)
    os.makedirs(suite_dir, exist_ok=True)

    for name, config in configs.items():
        exp_dir = os.path.join(suite_dir, name)
        metrics, trainer = run_single_experiment(config, name)
        save_experiment(metrics, config, name, exp_dir)
        all_metrics.append(metrics)
        all_labels.append(name)
        all_final.append(metrics["final_eval"])

    # Generate comparison plots
    plot_comparison(all_metrics, all_labels, suite_dir,
                    "vs_random_win_rate", "Win Rate", f"{suite_name}: Win Rate vs Random")
    plot_comparison(all_metrics, all_labels, suite_dir,
                    "vs_optimal_draw_rate", "Draw Rate", f"{suite_name}: Draw Rate vs Optimal")
    plot_comparison(all_metrics, all_labels, suite_dir,
                    "vs_optimal_loss_rate", "Loss Rate", f"{suite_name}: Loss Rate vs Optimal (Exploitability)")
    plot_comparison(all_metrics, all_labels, suite_dir,
                    "entropy", "Entropy", f"{suite_name}: Policy Entropy")

    # Build report sections
    sections = []

    # Summary table
    headers = ["Config", "vs Random Win%", "vs Random Draw%",
               "vs Optimal Draw%", "vs Optimal Loss%", "Time(s)"]
    rows = []
    for label, m, fe in zip(all_labels, all_metrics, all_final):
        rows.append([
            label,
            f"{fe['vs_random_win_rate']*100:.1f}",
            f"{fe['vs_random_draw_rate']*100:.1f}",
            f"{fe['vs_optimal_draw_rate']*100:.1f}",
            f"{fe['vs_optimal_loss_rate']*100:.1f}",
            f"{m['training_time']:.1f}",
        ])

    sections.append({
        "heading": "Summary",
        "text": "Final evaluation results for each configuration.",
        "table": {"headers": headers, "rows": rows},
    })

    # Comparison plots
    sections.append({
        "heading": "Training Curves — Comparison",
        "text": "Comparison of key metrics across configurations.",
        "plots": [
            (f"comparison_vs_random_win_rate.png", "Win rate vs random opponent across configs"),
            (f"comparison_vs_optimal_draw_rate.png", "Draw rate vs optimal opponent across configs"),
            (f"comparison_vs_optimal_loss_rate.png", "Loss rate vs optimal (exploitability) across configs"),
            (f"comparison_entropy.png", "Policy entropy across configs"),
        ],
        "page_break": True,
    })

    # Individual run plots
    for label, m in zip(all_labels, all_metrics):
        exp_dir = os.path.join(suite_dir, label)
        individual_plots = plot_training_curves(m, exp_dir, title_prefix=f"{label}: ")
        sections.append({
            "heading": f"Details: {label}",
            "plots": [(os.path.join(label, p[0]), p[1]) for p in individual_plots],
            "page_break": True,
        })

    # Generate PDF
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pdf_path = os.path.join(REPORTS_DIR, f"{suite_name}.pdf")
    generate_pdf_report(pdf_path, report_title, sections, suite_dir)

    return all_metrics, all_labels, all_final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=[
        "baseline", "lr", "entropy", "opponent", "network",
        "games", "clip", "snapshot", "tuned", "all"
    ])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiments = {
        "baseline": (experiment_1_baseline, "01_baseline", "Experiment 1: Baseline Training"),
        "lr": (experiment_2_learning_rate, "02_learning_rate", "Experiment 2: Learning Rate Sweep"),
        "entropy": (experiment_3_entropy_coef, "03_entropy", "Experiment 3: Entropy Coefficient"),
        "opponent": (experiment_4_opponent_sampling, "04_opponent_sampling", "Experiment 4: Opponent Sampling"),
        "network": (experiment_5_network_size, "05_network_size", "Experiment 5: Network Architecture"),
        "games": (experiment_6_games_per_iter, "06_games_per_iter", "Experiment 6: Games Per Iteration"),
        "clip": (experiment_7_clip_epsilon, "07_clip_epsilon", "Experiment 7: PPO Clip Epsilon"),
        "snapshot": (experiment_8_snapshot_interval, "08_snapshot_interval", "Experiment 8: Snapshot Interval"),
        "tuned": (experiment_9_final_tuned, "09_final_tuned", "Experiment 9: Final Tuned Config"),
    }

    if args.experiment == "all":
        for name, (fn, suite, title) in experiments.items():
            if name == "tuned":
                continue  # Skip until we have results
            run_experiment_suite(fn, suite, title)
    else:
        fn, suite, title = experiments[args.experiment]
        run_experiment_suite(fn, suite, title)
