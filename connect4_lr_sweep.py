"""Connect 4 LR sweep with Elo tracking.

Sweeps learning rates while keeping all other hparams fixed at v1 best config.
Uses Elo from the fixed reference pool as the primary metric.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import subprocess
import time

from src.report import generate_pdf_report, smooth

REPORT_DIR = "reports/connect4_lr_sweep"
os.makedirs(REPORT_DIR, exist_ok=True)

# Base config (v1 best, unchanged)
BASE_CONFIG = {
    "hidden_size": 256,
    "num_layers": 6,
    "ent_coef": 0.01,
    "games_per_iter": 512,
    "clip_eps": 0.1,
    "snapshot_interval": 25,
    "opponent_sampling": "uniform",
    "draw_reward": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ppo_epochs": 4,
    "batch_size": 256,
    "max_grad_norm": 0.5,
    "opponent_pool_max": 20,
}

LR_VALUES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SWEEP_ITERS = 500
EVAL_INTERVAL = 25
MAX_PARALLEL = 4  # Sweet spot for 12-core M2: ~3x throughput vs sequential


def _make_script(config, lr, out_path, num_iters, eval_interval):
    """Generate training script string for subprocess."""
    return f"""
import json, time, random, numpy as np, torch
from src.connect4_c import Connect4TrainerC

config = {json.dumps(config)}
torch.manual_seed(42); np.random.seed(42); random.seed(42)

trainer = Connect4TrainerC(config)
t0 = time.perf_counter()
metrics = trainer.train({num_iters}, eval_interval={eval_interval}, verbose=True)
wall_time = time.perf_counter() - t0

out = {{'metrics': {{}}, 'wall_time': wall_time, 'total_params': trainer.total_params, 'lr': {lr}}}
for k, v in metrics.items():
    if isinstance(v, list) and len(v) > 0:
        out['metrics'][k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
    else:
        out['metrics'][k] = v

with open('{out_path}', 'w') as f:
    json.dump(out, f)
print(f'Done. Wall time: {{wall_time:.1f}}s')
"""


def run_sweep():
    """Run all LR values in parallel batches."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    results = [None] * len(LR_VALUES)

    # Process in batches of MAX_PARALLEL
    for batch_start in range(0, len(LR_VALUES), MAX_PARALLEL):
        batch_end = min(batch_start + MAX_PARALLEL, len(LR_VALUES))
        batch = list(range(batch_start, batch_end))

        print(f"\n{'='*60}")
        print(f"Launching batch: LR = {[LR_VALUES[i] for i in batch]}")
        print(f"({'='*60})")

        procs = []
        for idx in batch:
            lr = LR_VALUES[idx]
            config = BASE_CONFIG.copy()
            config["lr"] = lr
            out_path = os.path.join(REPORT_DIR, f"lr_{idx}.json")
            script = _make_script(config, lr, out_path, SWEEP_ITERS, EVAL_INTERVAL)
            p = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=cwd,
            )
            procs.append((idx, lr, p, out_path))

        # Wait for batch to finish, print output as each completes
        for idx, lr, p, out_path in procs:
            stdout, stderr = p.communicate(timeout=600)
            # Print last few lines of output
            lines = stdout.decode().strip().split('\n')
            for line in lines[-3:]:
                print(f"  [lr={lr}] {line}")
            if p.returncode != 0:
                print(f"  [lr={lr}] FAILED: {stderr.decode()[-200:]}")
                continue
            with open(out_path) as f:
                results[idx] = json.load(f)

    return [r for r in results if r is not None]


def main():
    # Run sweep
    t0 = time.time()
    results = run_sweep()
    total_time = time.time() - t0
    print(f"\nSweep done in {total_time:.0f}s")

    # Save combined results
    with open(os.path.join(REPORT_DIR, "all_results.json"), "w") as f:
        json.dump({"results": results, "lr_values": LR_VALUES, "total_time": total_time}, f)

    print(f"Results saved to {REPORT_DIR}/all_results.json")
    print("Run with --report to generate plots and PDF.")


def generate_report_from_results():
    """Generate plots and report from saved results."""
    with open(os.path.join(REPORT_DIR, "all_results.json")) as f:
        data = json.load(f)
    results = data["results"]
    lr_values = data["lr_values"]
    total_time = data["total_time"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Plot 1: Elo curves for all LRs
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    for i, (r, lr) in enumerate(zip(results, lr_values)):
        m = r["metrics"]
        if "elo" in m:
            ax.plot(m["eval_iteration"], m["elo"], "-o", color=colors[i],
                    markersize=3, linewidth=2, label=f"lr={lr}")
    ax.axhline(y=1584, color="red", linestyle="--", alpha=0.5, label="Heuristic (1584)")
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Random (1000)")
    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Elo Rating", fontsize=12)
    ax.set_title("Elo vs Training Iteration by Learning Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "elo_by_lr.png"), dpi=150)
    plt.close(fig)

    # Plot 2: Entropy curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    for i, (r, lr) in enumerate(zip(results, lr_values)):
        m = r["metrics"]
        ax.plot(m["iteration"], smooth(np.array(m["entropy"]), 20), "-",
                color=colors[i], linewidth=1.5, label=f"lr={lr}")
    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Entropy", fontsize=12)
    ax.set_title("Entropy Over Training by Learning Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "entropy_by_lr.png"), dpi=150)
    plt.close(fig)

    # Plot 3: Policy loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (r, lr) in enumerate(zip(results, lr_values)):
        m = r["metrics"]
        axes[0].plot(m["iteration"], smooth(np.array(m["policy_loss"]), 20), "-",
                     color=colors[i], linewidth=1.5, label=f"lr={lr}")
        axes[1].plot(m["iteration"], smooth(np.array(m["value_loss"]), 20), "-",
                     color=colors[i], linewidth=1.5, label=f"lr={lr}")
    axes[0].set_title("Policy Loss")
    axes[1].set_title("Value Loss")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Loss Curves by Learning Rate", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "losses_by_lr.png"), dpi=150)
    plt.close(fig)

    # Plot 4: Final Elo bar chart
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    final_elos = []
    labels = []
    for r, lr in zip(results, lr_values):
        m = r["metrics"]
        final_elo = m["elo"][-1] if "elo" in m else 0
        final_elos.append(final_elo)
        labels.append(f"{lr}")
    bar_colors = [colors[i] for i in range(len(lr_values))]
    bars = ax.bar(labels, final_elos, color=bar_colors, width=0.5)
    ax.bar_label(bars, [f"{e:.0f}" for e in final_elos], fontsize=10)
    ax.axhline(y=1584, color="red", linestyle="--", alpha=0.5, label="Heuristic")
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Final Elo (iter 500)", fontsize=12)
    ax.set_title("Final Elo by Learning Rate", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "final_elo_bar.png"), dpi=150)
    plt.close(fig)

    # Plot 5: vs Heuristic win rate (for comparison with Elo)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (r, lr) in enumerate(zip(results, lr_values)):
        m = r["metrics"]
        ax1.plot(m["eval_iteration"], m["vs_heuristic_win_rate"], "-o",
                 color=colors[i], markersize=2, linewidth=1.5, label=f"lr={lr}")
        if "elo" in m:
            ax2.plot(m["eval_iteration"], m["elo"], "-o",
                     color=colors[i], markersize=2, linewidth=1.5, label=f"lr={lr}")
    ax1.set_title("vs Heuristic Win Rate (noisy)")
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_title("Elo Rating (smooth)")
    ax2.axhline(y=1584, color="red", linestyle="--", alpha=0.5)
    for ax in (ax1, ax2):
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Win Rate vs Elo: LR Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "winrate_vs_elo.png"), dpi=150)
    plt.close(fig)

    print("Plots saved.")

    # Build report
    # Gather final stats
    rows = []
    for r, lr in zip(results, lr_values):
        m = r["metrics"]
        final_elo = m["elo"][-1] if "elo" in m else 0
        peak_elo = max(m["elo"]) if "elo" in m else 0
        final_ent = m["entropy"][-1]
        final_vr = m["vs_random_win_rate"][-1]
        final_vh = m["vs_heuristic_win_rate"][-1]
        rows.append([f"{lr}", f"{final_elo:.0f}", f"{peak_elo:.0f}",
                      f"{final_ent:.3f}", f"{final_vr:.0%}", f"{final_vh:.0%}",
                      f"{r['wall_time']:.0f}s"])

    best_idx = max(range(len(results)),
                   key=lambda i: results[i]["metrics"]["elo"][-1] if "elo" in results[i]["metrics"] else 0)
    best_lr = lr_values[best_idx]
    best_elo = results[best_idx]["metrics"]["elo"][-1]

    sections = [
        {
            "heading": "Executive Summary",
            "text": (
                "This report presents a learning rate sweep for Connect 4 self-play PPO, using "
                "<b>Elo rating</b> against a fixed reference pool as the primary evaluation metric. "
                "All other hyperparameters are held constant at the v1 best configuration "
                "(256x6 arch, ent_coef=0.01, draw_reward=0.5)."
                "\n\n"
                f"<b>LR values tested:</b> {', '.join(str(lr) for lr in lr_values)}<br/>"
                f"<b>Training:</b> {SWEEP_ITERS} iterations, {BASE_CONFIG['games_per_iter']} games/iter, "
                f"eval every {EVAL_INTERVAL} iters<br/>"
                f"<b>Total sweep time:</b> {total_time:.0f}s"
                "\n\n"
                f"<b>Best LR:</b> {best_lr} with final Elo <b>{best_elo:.0f}</b>"
            ),
            "table": {
                "headers": ["LR", "Final Elo", "Peak Elo", "Final Entropy",
                            "vs Random", "vs Heuristic", "Time"],
                "rows": rows,
            },
            "page_break": True,
        },
        {
            "heading": "Elo Curves by Learning Rate",
            "text": (
                "The Elo curve is the key metric. It shows how agent strength evolves over training "
                "for each learning rate. The heuristic baseline is at Elo 1584, random at 1000."
            ),
            "plots": [
                ("elo_by_lr.png", "Elo rating over training for each learning rate."),
            ],
            "page_break": True,
        },
        {
            "heading": "Win Rate vs Elo Comparison",
            "text": (
                "Side-by-side comparison of the noisy win-rate metric and the smooth Elo metric. "
                "The Elo curves make it much easier to distinguish which LR is performing best."
            ),
            "plots": [
                ("winrate_vs_elo.png", "Left: noisy win rate. Right: smooth Elo."),
            ],
            "page_break": True,
        },
        {
            "heading": "Training Dynamics",
            "text": (
                "Entropy and loss curves reveal how different learning rates affect training dynamics."
            ),
            "plots": [
                ("entropy_by_lr.png", "Entropy over training by learning rate."),
                ("losses_by_lr.png", "Policy and value loss by learning rate."),
            ],
            "page_break": True,
        },
        {
            "heading": "Final Elo Comparison",
            "plots": [
                ("final_elo_bar.png", "Final Elo at iteration 500 for each learning rate."),
            ],
        },
    ]

    report_path = os.path.join(REPORT_DIR, "lr_sweep_report.pdf")
    generate_pdf_report(
        report_path=report_path,
        title="Connect 4 Self-Play PPO: Learning Rate Sweep",
        sections=sections,
        plot_dir=REPORT_DIR,
    )
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        generate_report_from_results()
    else:
        main()
        generate_report_from_results()
