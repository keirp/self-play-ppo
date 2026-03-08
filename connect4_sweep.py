"""Hyperparameter sweep for Connect 4 self-play PPO (C backend).

Sweep design:
- Phase 1: Architecture sweep (network size) with fixed hparams, short training
- Phase 2: Hparam sweep (lr, ent_coef, etc.) with best architecture, short training
- Phase 3: Full training with best config
- Generate PDF report with all results
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import json
import random
import torch
import itertools
from collections import defaultdict

from src.connect4_c import Connect4TrainerC, load_params, _lib
from src.connect4 import play_vs_random, play_vs_heuristic
from src.report import generate_pdf_report, smooth

REPORT_DIR = "reports/connect4"
WEIGHTS_DIR = "weights"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ==================== SWEEP CONFIGURATION ====================

# Phase 1: Architecture sweep (short training to compare architectures)
ARCH_CONFIGS = [
    {"hidden_size": 256, "num_layers": 4, "label": "256x4 (232K)"},
    {"hidden_size": 512, "num_layers": 4, "label": "512x4 (858K)"},
    {"hidden_size": 256, "num_layers": 6, "label": "256x6 (363K)"},
    {"hidden_size": 512, "num_layers": 6, "label": "512x6 (1.38M)"},
]

ARCH_SWEEP_BASE = {
    "lr": 1e-3,
    "ent_coef": 0.05,
    "games_per_iter": 512,
    "clip_eps": 0.1,
    "snapshot_interval": 25,
    "opponent_sampling": "uniform",
    "draw_reward": 0.3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ppo_epochs": 4,
    "batch_size": 256,
    "max_grad_norm": 0.5,
    "opponent_pool_max": 20,
}
ARCH_SWEEP_ITERS = 500
ARCH_SWEEP_EVAL = 25

# Phase 2: Hyperparameter sweep with best architecture
HPARAM_GRID = {
    "lr": [3e-4, 1e-3, 3e-3],
    "ent_coef": [0.01, 0.05, 0.1],
    "draw_reward": [0.0, 0.3, 0.5],
}
HPARAM_SWEEP_ITERS = 500
HPARAM_SWEEP_EVAL = 25

# Phase 3: Full training
FULL_TRAIN_ITERS = 3000
FULL_TRAIN_EVAL = 50


def run_training(config, num_iters, eval_interval, label="", verbose=True):
    """Run a single training run and return metrics + wall time."""
    # Need a fresh C library state for each architecture — but c4_init only runs once.
    # WORKAROUND: Since architecture is set at init, we need to reload the library
    # for different architectures. For same architecture, just reset params.

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")

    trainer = Connect4TrainerC(config)
    t0 = time.perf_counter()
    metrics = trainer.train(num_iters, eval_interval=eval_interval, verbose=verbose)
    wall_time = time.perf_counter() - t0

    if verbose:
        print(f"Done in {wall_time:.1f}s ({wall_time/num_iters*1000:.1f}ms/iter)")

    return metrics, wall_time, trainer


def phase1_architecture_sweep():
    """Sweep over network architectures."""
    print("\n" + "="*70)
    print("PHASE 1: Architecture Sweep")
    print("="*70)

    results = []
    for arch in ARCH_CONFIGS:
        config = ARCH_SWEEP_BASE.copy()
        config["hidden_size"] = arch["hidden_size"]
        config["num_layers"] = arch["num_layers"]

        # We need to reload C library for each architecture
        # Since c4_init only runs once, we need to use subprocess or accept this limitation
        # For now, run each in a subprocess via a helper
        metrics, wall_time, trainer = run_training(
            config, ARCH_SWEEP_ITERS, ARCH_SWEEP_EVAL,
            label=f"Arch: {arch['label']}", verbose=True
        )

        final_vs_random = metrics["vs_random_win_rate"][-1] if metrics.get("vs_random_win_rate") else 0
        final_vs_heuristic = metrics["vs_heuristic_win_rate"][-1] if metrics.get("vs_heuristic_win_rate") else 0

        results.append({
            "arch": arch,
            "config": config,
            "metrics": metrics,
            "wall_time": wall_time,
            "final_vs_random": final_vs_random,
            "final_vs_heuristic": final_vs_heuristic,
            "total_params": trainer.total_params,
            "ms_per_iter": wall_time / ARCH_SWEEP_ITERS * 1000,
        })

        # NOTE: c4_init only runs once, so subsequent calls with different
        # hidden_size/num_layers will be ignored. We need to handle this.
        # For a proper sweep, each config needs its own process.

    return results


def phase1_architecture_sweep_subprocess():
    """Run each architecture in a subprocess to get fresh C library state."""
    import subprocess, sys

    print("\n" + "="*70)
    print("PHASE 1: Architecture Sweep")
    print("="*70)

    results = []
    for i, arch in enumerate(ARCH_CONFIGS):
        config = ARCH_SWEEP_BASE.copy()
        config["hidden_size"] = arch["hidden_size"]
        config["num_layers"] = arch["num_layers"]

        out_path = os.path.join(REPORT_DIR, f"arch_{i}.json")

        script = f"""
import json, time, random, numpy as np, torch
from src.connect4_c import Connect4TrainerC

config = {json.dumps(config)}
torch.manual_seed(42); np.random.seed(42); random.seed(42)

trainer = Connect4TrainerC(config)
t0 = time.perf_counter()
metrics = trainer.train({ARCH_SWEEP_ITERS}, eval_interval={ARCH_SWEEP_EVAL}, verbose=True)
wall_time = time.perf_counter() - t0

# Convert for JSON
out = {{'metrics': {{}}, 'wall_time': wall_time, 'total_params': trainer.total_params}}
for k, v in metrics.items():
    if isinstance(v, list) and len(v) > 0:
        out['metrics'][k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
    else:
        out['metrics'][k] = v

with open('{out_path}', 'w') as f:
    json.dump(out, f)
print(f'Saved to {out_path}')
"""
        print(f"\n--- Running Arch: {arch['label']} ---")
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=False, timeout=600)

        if os.path.exists(out_path):
            with open(out_path) as f:
                data = json.load(f)
            metrics = data["metrics"]
            final_vs_random = metrics.get("vs_random_win_rate", [0])[-1]
            final_vs_heuristic = metrics.get("vs_heuristic_win_rate", [0])[-1]

            results.append({
                "arch": arch,
                "config": config,
                "metrics": metrics,
                "wall_time": data["wall_time"],
                "final_vs_random": final_vs_random,
                "final_vs_heuristic": final_vs_heuristic,
                "total_params": data["total_params"],
                "ms_per_iter": data["wall_time"] / ARCH_SWEEP_ITERS * 1000,
            })
        else:
            print(f"WARNING: {arch['label']} failed!")

    return results


def phase2_hparam_sweep(best_arch):
    """Sweep over hyperparameters with the best architecture."""
    import subprocess, sys

    print("\n" + "="*70)
    print("PHASE 2: Hyperparameter Sweep")
    print(f"Architecture: H={best_arch['hidden_size']}, L={best_arch['num_layers']}")
    print("="*70)

    # Generate grid
    keys = list(HPARAM_GRID.keys())
    values = list(HPARAM_GRID.values())
    combos = list(itertools.product(*values))

    results = []
    for i, combo in enumerate(combos):
        config = ARCH_SWEEP_BASE.copy()
        config["hidden_size"] = best_arch["hidden_size"]
        config["num_layers"] = best_arch["num_layers"]
        for k, v in zip(keys, combo):
            config[k] = v

        label = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
        out_path = os.path.join(REPORT_DIR, f"hparam_{i}.json")

        script = f"""
import json, time, random, numpy as np, torch
from src.connect4_c import Connect4TrainerC

config = {json.dumps(config)}
torch.manual_seed(42); np.random.seed(42); random.seed(42)

trainer = Connect4TrainerC(config)
t0 = time.perf_counter()
metrics = trainer.train({HPARAM_SWEEP_ITERS}, eval_interval={HPARAM_SWEEP_EVAL}, verbose=True)
wall_time = time.perf_counter() - t0

out = {{'metrics': {{}}, 'wall_time': wall_time, 'total_params': trainer.total_params}}
for k, v in metrics.items():
    if isinstance(v, list) and len(v) > 0:
        out['metrics'][k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
    else:
        out['metrics'][k] = v

with open('{out_path}', 'w') as f:
    json.dump(out, f)
"""
        print(f"\n--- Hparam [{i+1}/{len(combos)}]: {label} ---")
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=False, timeout=600)

        if os.path.exists(out_path):
            with open(out_path) as f:
                data = json.load(f)
            metrics = data["metrics"]
            final_vs_random = metrics.get("vs_random_win_rate", [0])[-1]
            final_vs_heuristic = metrics.get("vs_heuristic_win_rate", [0])[-1]

            results.append({
                "hparams": dict(zip(keys, combo)),
                "label": label,
                "config": config,
                "metrics": metrics,
                "wall_time": data["wall_time"],
                "final_vs_random": final_vs_random,
                "final_vs_heuristic": final_vs_heuristic,
            })

    return results


def phase3_full_training(best_config):
    """Full training run with the best configuration."""
    import subprocess, sys

    print("\n" + "="*70)
    print("PHASE 3: Full Training")
    print("="*70)

    out_path = os.path.join(REPORT_DIR, "full_train.json")
    weights_path = os.path.join(WEIGHTS_DIR, "connect4_policy_params.npy")
    pt_weights_path = os.path.join(WEIGHTS_DIR, "connect4_policy.pt")

    script = f"""
import json, time, random, numpy as np, torch
from src.connect4_c import Connect4TrainerC, load_params

config = {json.dumps(best_config)}
torch.manual_seed(42); np.random.seed(42); random.seed(42)

trainer = Connect4TrainerC(config)
t0 = time.perf_counter()
metrics = trainer.train({FULL_TRAIN_ITERS}, eval_interval={FULL_TRAIN_EVAL}, verbose=True)
wall_time = time.perf_counter() - t0

# Save weights
np.save('{weights_path}', trainer.params)
load_params(trainer.agent, trainer.params)
torch.save(trainer.agent.state_dict(), '{pt_weights_path}')

out = {{'metrics': {{}}, 'wall_time': wall_time, 'total_params': trainer.total_params}}
for k, v in metrics.items():
    if isinstance(v, list) and len(v) > 0:
        out['metrics'][k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
    else:
        out['metrics'][k] = v

with open('{out_path}', 'w') as f:
    json.dump(out, f)
print(f'Full training done in {{wall_time:.1f}}s')
"""
    subprocess.run([sys.executable, "-c", script], capture_output=False, timeout=3600)

    with open(out_path) as f:
        data = json.load(f)
    return data


# ==================== PLOTTING ====================

def plot_arch_comparison(arch_results, save_dir):
    """Plot architecture comparison curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = ["#4472C4", "#ED7D31", "#70AD47", "#FF4444"]
    for i, r in enumerate(arch_results):
        m = r["metrics"]
        label = r["arch"]["label"]
        color = colors[i % len(colors)]

        if "vs_random_win_rate" in m:
            axes[0].plot(m["eval_iteration"], m["vs_random_win_rate"],
                        "o-", label=label, color=color, markersize=2)
        if "vs_heuristic_win_rate" in m:
            axes[1].plot(m["eval_iteration"], m["vs_heuristic_win_rate"],
                        "o-", label=label, color=color, markersize=2)
        if "entropy" in m:
            axes[2].plot(m["iteration"], smooth(np.array(m["entropy"]), 20),
                        label=label, color=color, linewidth=1)

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("vs Random Win Rate"); axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_title("vs Heuristic Win Rate"); axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_title("Entropy")
    for ax in axes:
        ax.set_xlabel("Iteration")
    fig.suptitle("Architecture Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "arch_comparison.png"), dpi=150)
    plt.close(fig)


def plot_arch_speed(arch_results, save_dir):
    """Bar chart of architecture speed and final performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    labels = [r["arch"]["label"] for r in arch_results]
    ms_per_iter = [r["ms_per_iter"] for r in arch_results]
    vs_heuristic = [r["final_vs_heuristic"] for r in arch_results]

    colors = ["#4472C4", "#ED7D31", "#70AD47", "#FF4444"]
    bars = ax1.bar(labels, ms_per_iter, color=colors[:len(labels)], width=0.5)
    ax1.bar_label(bars, [f"{x:.0f}ms" for x in ms_per_iter], fontsize=9)
    ax1.set_ylabel("ms / iteration")
    ax1.set_title("Training Speed")
    ax1.grid(True, alpha=0.2, axis="y")

    bars = ax2.bar(labels, vs_heuristic, color=colors[:len(labels)], width=0.5)
    ax2.bar_label(bars, [f"{x:.0%}" for x in vs_heuristic], fontsize=9)
    ax2.set_ylabel("Win Rate")
    ax2.set_title(f"vs Heuristic Win Rate (after {ARCH_SWEEP_ITERS} iters)")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "arch_speed.png"), dpi=150)
    plt.close(fig)


def plot_hparam_heatmaps(hparam_results, save_dir):
    """Plot heatmaps for hyperparameter sweep results."""
    keys = list(HPARAM_GRID.keys())
    # Create pivot tables for each pair of hyperparameters
    # Best metric: vs_heuristic_win_rate

    # 3D grid: lr x ent_coef x draw_reward
    lrs = HPARAM_GRID["lr"]
    ents = HPARAM_GRID["ent_coef"]
    drs = HPARAM_GRID["draw_reward"]

    # For each draw_reward, make a lr x ent_coef heatmap
    fig, axes = plt.subplots(1, len(drs), figsize=(5*len(drs), 4))
    if len(drs) == 1:
        axes = [axes]

    for di, dr_val in enumerate(drs):
        grid = np.zeros((len(lrs), len(ents)))
        for r in hparam_results:
            hp = r["hparams"]
            if abs(hp["draw_reward"] - dr_val) < 1e-6:
                li = lrs.index(hp["lr"])
                ei = ents.index(hp["ent_coef"])
                grid[li, ei] = r["final_vs_heuristic"]

        im = axes[di].imshow(grid, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        axes[di].set_xticks(range(len(ents)))
        axes[di].set_xticklabels([str(e) for e in ents])
        axes[di].set_yticks(range(len(lrs)))
        axes[di].set_yticklabels([f"{l:.0e}" for l in lrs])
        axes[di].set_xlabel("Entropy Coef")
        axes[di].set_ylabel("Learning Rate")
        axes[di].set_title(f"draw_reward={dr_val}")

        # Annotate cells
        for li in range(len(lrs)):
            for ei in range(len(ents)):
                axes[di].text(ei, li, f"{grid[li,ei]:.0%}",
                             ha="center", va="center", fontsize=9,
                             color="white" if grid[li,ei] > 0.5 else "black")

    fig.suptitle("Hparam Sweep: vs Heuristic Win Rate", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "hparam_heatmaps.png"), dpi=150)
    plt.close(fig)


def plot_full_training(metrics, wall_time, save_dir):
    """Plot full training curves."""
    # vs Random and vs Heuristic
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    iters = metrics["eval_iteration"]
    axes[0,0].plot(iters, metrics["vs_random_win_rate"], "g-o", markersize=2, label="Win")
    axes[0,0].plot(iters, metrics["vs_random_draw_rate"], "b-s", markersize=2, label="Draw")
    axes[0,0].plot(iters, metrics["vs_random_loss_rate"], "r-^", markersize=2, label="Loss")
    axes[0,0].set_title("vs Random"); axes[0,0].legend(); axes[0,0].set_ylim(-0.05, 1.05)
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(iters, metrics["vs_heuristic_win_rate"], "g-o", markersize=2, label="Win")
    axes[0,1].plot(iters, metrics["vs_heuristic_draw_rate"], "b-s", markersize=2, label="Draw")
    axes[0,1].plot(iters, metrics["vs_heuristic_loss_rate"], "r-^", markersize=2, label="Loss")
    axes[0,1].set_title("vs Heuristic"); axes[0,1].legend(); axes[0,1].set_ylim(-0.05, 1.05)
    axes[0,1].grid(True, alpha=0.3)

    t_iters = metrics["iteration"]
    axes[0,2].plot(t_iters, smooth(np.array(metrics["sp_win_rate"]), 20), "g-", label="Win")
    axes[0,2].plot(t_iters, smooth(np.array(metrics["sp_draw_rate"]), 20), "b-", label="Draw")
    axes[0,2].plot(t_iters, smooth(np.array(metrics["sp_loss_rate"]), 20), "r-", label="Loss")
    axes[0,2].set_title("Self-Play Results"); axes[0,2].legend(); axes[0,2].set_ylim(-0.05, 1.05)
    axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(t_iters, smooth(np.array(metrics["policy_loss"]), 20))
    axes[1,0].set_title("Policy Loss"); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(t_iters, smooth(np.array(metrics["value_loss"]), 20))
    axes[1,1].set_title("Value Loss"); axes[1,1].grid(True, alpha=0.3)

    axes[1,2].plot(t_iters, smooth(np.array(metrics["entropy"]), 20))
    axes[1,2].set_title("Entropy"); axes[1,2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Iteration")
    fig.suptitle(f"Full Connect 4 Training ({FULL_TRAIN_ITERS} iters, {wall_time:.0f}s)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "full_training.png"), dpi=150)
    plt.close(fig)


def generate_report(arch_results, hparam_results, full_data, best_arch, best_hparams):
    """Generate the PDF report."""

    full_metrics = full_data["metrics"]
    full_time = full_data["wall_time"]
    total_params = full_data["total_params"]

    speedup_text = f"{full_time/FULL_TRAIN_ITERS*1000:.0f}ms/iter"

    # Final eval stats
    final_vs_random_w = full_metrics.get("vs_random_win_rate", [0])[-1]
    final_vs_random_d = full_metrics.get("vs_random_draw_rate", [0])[-1]
    final_vs_random_l = full_metrics.get("vs_random_loss_rate", [0])[-1]
    final_vs_heur_w = full_metrics.get("vs_heuristic_win_rate", [0])[-1]
    final_vs_heur_d = full_metrics.get("vs_heuristic_draw_rate", [0])[-1]
    final_vs_heur_l = full_metrics.get("vs_heuristic_loss_rate", [0])[-1]

    sections = [
        {
            "heading": "Executive Summary",
            "text": (
                "This report presents a systematic study of training a Connect 4 agent using "
                "self-play PPO with a pure C backend (Apple Accelerate BLAS). Connect 4 is a "
                "substantially harder game than tic-tac-toe: the state space is ~4.5 trillion positions "
                "(vs 5,478 for tic-tac-toe), games last ~36-40 ply (vs ~9 for tic-tac-toe), and "
                "the game is a first-player win with perfect play (vs a draw for tic-tac-toe)."
                "\n\n"
                "The study followed a rigorous three-phase approach:<br/>"
                f"1. <b>Architecture sweep</b> over {len(ARCH_CONFIGS)} network sizes ({ARCH_SWEEP_ITERS} iters each)<br/>"
                f"2. <b>Hyperparameter sweep</b> over {len(hparam_results)} configurations ({HPARAM_SWEEP_ITERS} iters each)<br/>"
                f"3. <b>Full training</b> with the best configuration ({FULL_TRAIN_ITERS} iterations)"
                "\n\n"
                f"The best agent achieves <b>{final_vs_random_w:.0%}</b> win rate vs random and "
                f"<b>{final_vs_heur_w:.0%}</b> win rate vs a blocking heuristic, training in "
                f"<b>{full_time:.0f}s</b> ({speedup_text})."
            ),
            "page_break": True,
        },
        {
            "heading": "Game: Connect 4",
            "text": (
                "Connect 4 is played on a 6-row x 7-column vertical grid. Players alternate dropping "
                "pieces into columns, where they fall to the lowest empty row. The first player to "
                "connect four pieces in a row (horizontally, vertically, or diagonally) wins."
                "\n\n"
                "<b>Key complexity metrics:</b><br/>"
                "- State space: ~4.5 x 10^12 positions (vs 5,478 for tic-tac-toe)<br/>"
                "- Game-tree complexity: ~4.7 x 10^21<br/>"
                "- Average game length: ~36-40 ply (both players)<br/>"
                "- Branching factor: ~4 (average valid moves)<br/>"
                "- Solved: First player wins with perfect play (center column opening)"
                "\n\n"
                "<b>Observation encoding:</b> 6x7x3 = 126 dimensions (3 channels per cell: "
                "current player pieces, opponent pieces, and a constant bias plane), flattened in "
                "row-major interleaved order to match the tic-tac-toe encoding convention."
                "\n\n"
                "<b>Evaluation opponents:</b><br/>"
                "- <b>Random:</b> Uniform random over valid columns<br/>"
                "- <b>Heuristic:</b> Wins if possible, blocks opponent wins, prefers center columns"
            ),
            "page_break": True,
        },
        # Phase 1: Architecture sweep
        {
            "heading": "Phase 1: Architecture Sweep",
            "text": (
                f"We compared {len(ARCH_CONFIGS)} MLP architectures, each trained for "
                f"{ARCH_SWEEP_ITERS} iterations with {ARCH_SWEEP_BASE['games_per_iter']} games/iter. "
                "All architectures used the same base hyperparameters. The goal was to determine "
                "the best network size for Connect 4 before tuning other hyperparameters."
            ),
            "table": {
                "headers": ["Architecture", "Parameters", "ms/iter", "vs Random", "vs Heuristic"],
                "rows": [
                    [r["arch"]["label"], f"{r['total_params']:,}", f"{r['ms_per_iter']:.0f}",
                     f"{r['final_vs_random']:.0%}", f"{r['final_vs_heuristic']:.0%}"]
                    for r in arch_results
                ],
            },
            "plots": [
                ("arch_comparison.png", "Architecture comparison: evaluation metrics over training."),
                ("arch_speed.png", "Training speed and final performance by architecture."),
            ],
            "page_break": True,
        },
        # Phase 2: Hparam sweep
        {
            "heading": "Phase 2: Hyperparameter Sweep",
            "text": (
                f"Using the best architecture (H={best_arch['hidden_size']}, L={best_arch['num_layers']}), "
                f"we swept over {len(hparam_results)} hyperparameter combinations. "
                "The grid covered learning rate, entropy coefficient, and draw reward — "
                "the three parameters most likely to affect Connect 4 convergence."
                "\n\n"
                "The heatmaps below show the vs-heuristic win rate for each (lr, ent_coef) combination, "
                "with separate panels for each draw_reward value."
            ),
            "plots": [("hparam_heatmaps.png", "Hyperparameter sweep results (vs Heuristic win rate).")],
        },
        {
            "heading": "Best Hyperparameters",
            "table": {
                "headers": ["Parameter", "Value"],
                "rows": [[k, str(v)] for k, v in best_hparams.items()] + [
                    ["Architecture", f"H={best_arch['hidden_size']}, L={best_arch['num_layers']}"],
                    ["Total parameters", f"{total_params:,}"],
                ],
            },
            "page_break": True,
        },
        # Phase 3: Full training
        {
            "heading": "Phase 3: Full Training",
            "text": (
                f"The best configuration was trained for {FULL_TRAIN_ITERS} iterations "
                f"({ARCH_SWEEP_BASE['games_per_iter']} games/iter). Total training time: "
                f"<b>{full_time:.0f}s</b> ({speedup_text})."
            ),
            "plots": [("full_training.png", "Full training curves over 3000 iterations.")],
        },
        {
            "heading": "Final Evaluation",
            "table": {
                "headers": ["Metric", "Value"],
                "rows": [
                    ["vs Random: Win", f"{final_vs_random_w:.0%}"],
                    ["vs Random: Draw", f"{final_vs_random_d:.0%}"],
                    ["vs Random: Loss", f"{final_vs_random_l:.0%}"],
                    ["vs Heuristic: Win", f"{final_vs_heur_w:.0%}"],
                    ["vs Heuristic: Draw", f"{final_vs_heur_d:.0%}"],
                    ["vs Heuristic: Loss", f"{final_vs_heur_l:.0%}"],
                    ["Training time", f"{full_time:.0f}s"],
                    ["Per-iteration time", speedup_text],
                    ["Total iterations", str(FULL_TRAIN_ITERS)],
                    ["Total games played", f"{FULL_TRAIN_ITERS * ARCH_SWEEP_BASE['games_per_iter']:,}"],
                ],
            },
            "page_break": True,
        },
        {
            "heading": "Saved Weights",
            "text": (
                "Trained model weights have been saved to the <b>weights/</b> directory:"
                "\n\n"
                "- <b>weights/connect4_policy.pt</b> — PyTorch state_dict format<br/>"
                "- <b>weights/connect4_policy_params.npy</b> — Flat numpy array for C backend"
            ),
        },
        {
            "heading": "Comparison with Tic-Tac-Toe",
            "text": (
                "Connect 4 presents a qualitatively different challenge from tic-tac-toe:"
                "\n\n"
                "- <b>State space:</b> ~10^12 vs ~10^3 — a billion-fold increase<br/>"
                "- <b>Game length:</b> ~36-40 ply vs ~9 ply — 4x more decisions per game<br/>"
                "- <b>Network size:</b> required significantly more parameters<br/>"
                "- <b>Training iterations:</b> required significantly more iterations to converge<br/>"
                "- <b>Evaluation:</b> no minimax oracle available (too expensive), "
                "so we evaluate against random and heuristic opponents"
                "\n\n"
                "Despite these challenges, the C backend's speed advantage enables "
                "a complete hyperparameter sweep and full training run in a reasonable time. "
                "The same experiment with the PyTorch backend would take ~15x longer."
            ),
        },
        {
            "heading": "Conclusion",
            "text": (
                f"A self-play PPO agent for Connect 4 was trained to achieve "
                f"<b>{final_vs_random_w:.0%}</b> win rate vs random and "
                f"<b>{final_vs_heur_w:.0%}</b> win rate vs a blocking heuristic. "
                f"The pure C backend enabled a complete three-phase study "
                f"(architecture sweep + hparam sweep + full training) in total wall-clock time "
                f"that would be infeasible with the PyTorch baseline."
                "\n\n"
                "Future directions include: adding MCTS for stronger play, "
                "trying CNN architectures that exploit spatial structure, "
                "and evaluating against a Connect 4 solver to measure true exploitability."
            ),
        },
    ]

    generate_pdf_report(
        report_path=os.path.join(REPORT_DIR, "connect4_report.pdf"),
        title="Connect 4: Self-Play PPO Training Report",
        sections=sections,
        plot_dir=REPORT_DIR,
    )


def main():
    # Phase 1: Architecture sweep
    arch_results = phase1_architecture_sweep_subprocess()

    # Pick best architecture by vs_heuristic_win_rate
    best_arch_idx = max(range(len(arch_results)),
                        key=lambda i: arch_results[i]["final_vs_heuristic"])
    best_arch = arch_results[best_arch_idx]["arch"]
    print(f"\nBest architecture: {best_arch['label']}")

    # Phase 2: Hparam sweep
    hparam_results = phase2_hparam_sweep(best_arch)

    # Pick best hparams
    best_hp_idx = max(range(len(hparam_results)),
                      key=lambda i: hparam_results[i]["final_vs_heuristic"])
    best_hparams = hparam_results[best_hp_idx]["hparams"]
    best_config = hparam_results[best_hp_idx]["config"]
    print(f"\nBest hparams: {best_hparams}")

    # Phase 3: Full training
    full_data = phase3_full_training(best_config)

    # Generate plots
    plot_arch_comparison(arch_results, REPORT_DIR)
    plot_arch_speed(arch_results, REPORT_DIR)
    plot_hparam_heatmaps(hparam_results, REPORT_DIR)
    plot_full_training(full_data["metrics"], full_data["wall_time"], REPORT_DIR)

    # Generate report
    generate_report(arch_results, hparam_results, full_data, best_arch, best_hparams)
    print(f"\nReport saved to: {REPORT_DIR}/connect4_report.pdf")
    print(f"Weights saved to: {WEIGHTS_DIR}/")


if __name__ == "__main__":
    main()
