"""Connect 4 Pool Config Sweep with 3 seeds per config and PDF report.

Sweeps opponent pool configurations (pool size, snapshot interval, sampling strategy)
with reservoir sampling. Uses Elo from fixed reference pool as primary metric.
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

REPORT_DIR = "reports/connect4_pool_sweep"
os.makedirs(REPORT_DIR, exist_ok=True)

# Base config (best from hparam sweep)
BASE_CONFIG = {
    "hidden_size": 256,
    "num_layers": 6,
    "lr": 1e-4,
    "ent_coef": 0.005,
    "games_per_iter": 512,
    "clip_eps": 0.1,
    "draw_reward": 0.3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ppo_epochs": 4,
    "batch_size": 256,
    "max_grad_norm": 0.5,
}

CONFIGS = [
    {"name": "p20_s25_uniform",   "opponent_pool_max": 20,  "snapshot_interval": 25, "opponent_sampling": "uniform"},
    {"name": "p50_s25_uniform",   "opponent_pool_max": 50,  "snapshot_interval": 25, "opponent_sampling": "uniform"},
    {"name": "p100_s25_uniform",  "opponent_pool_max": 100, "snapshot_interval": 25, "opponent_sampling": "uniform"},
    {"name": "p20_s10_uniform",   "opponent_pool_max": 20,  "snapshot_interval": 10, "opponent_sampling": "uniform"},
    {"name": "p50_s10_uniform",   "opponent_pool_max": 50,  "snapshot_interval": 10, "opponent_sampling": "uniform"},
    {"name": "p20_s25_latest",    "opponent_pool_max": 20,  "snapshot_interval": 25, "opponent_sampling": "latest"},
    {"name": "p50_s25_weighted",  "opponent_pool_max": 50,  "snapshot_interval": 25, "opponent_sampling": "weighted_recent"},
]

NUM_SEEDS = 3
SWEEP_ITERS = 1000
EVAL_INTERVAL = 25
MAX_PARALLEL = 4


def _make_script(config, out_path, num_iters, eval_interval, seed):
    return f"""
import json, time, random, numpy as np, torch
from src.connect4_c import Connect4TrainerC

config = {json.dumps(config)}
torch.manual_seed({seed}); np.random.seed({seed}); random.seed({seed})

trainer = Connect4TrainerC(config)
t0 = time.perf_counter()
metrics = trainer.train({num_iters}, eval_interval={eval_interval}, verbose=True)
wall_time = time.perf_counter() - t0

out = {{'metrics': {{}}, 'wall_time': wall_time, 'total_params': trainer.total_params, 'seed': {seed}}}
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
    cwd = os.path.dirname(os.path.abspath(__file__))
    # Build list of all (config_idx, seed) jobs
    jobs = []
    for ci, cfg in enumerate(CONFIGS):
        for seed in range(NUM_SEEDS):
            jobs.append((ci, seed))

    results = {}  # (ci, seed) -> result dict

    for batch_start in range(0, len(jobs), MAX_PARALLEL):
        batch = jobs[batch_start:batch_start + MAX_PARALLEL]
        print(f"\n{'='*60}")
        print(f"Launching batch {batch_start//MAX_PARALLEL + 1}/{(len(jobs)+MAX_PARALLEL-1)//MAX_PARALLEL}: "
              f"{[(CONFIGS[ci]['name'], f'seed={s}') for ci, s in batch]}")
        print(f"{'='*60}")

        procs = []
        for ci, seed in batch:
            cfg = CONFIGS[ci]
            config = BASE_CONFIG.copy()
            config.update({k: v for k, v in cfg.items() if k != "name"})
            out_path = os.path.join(REPORT_DIR, f"{cfg['name']}_seed{seed}.json")
            script = _make_script(config, out_path, SWEEP_ITERS, EVAL_INTERVAL, seed + 42)
            p = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=cwd,
            )
            procs.append((ci, seed, p, out_path, cfg["name"]))

        for ci, seed, p, out_path, name in procs:
            stdout, stderr = p.communicate(timeout=1200)
            lines = stdout.decode().strip().split('\n')
            for line in lines[-3:]:
                print(f"  [{name} s{seed}] {line}")
            if p.returncode != 0:
                print(f"  [{name} s{seed}] FAILED: {stderr.decode()[-300:]}")
                continue
            with open(out_path) as f:
                results[(ci, seed)] = json.load(f)

    return results


def main():
    t0 = time.time()
    results = run_sweep()
    total_time = time.time() - t0
    print(f"\nSweep done in {total_time:.0f}s")

    # Save combined
    serializable = {}
    for (ci, seed), r in results.items():
        serializable[f"{ci}_{seed}"] = r
    with open(os.path.join(REPORT_DIR, "all_results_3seed.json"), "w") as f:
        json.dump({"results": serializable, "configs": CONFIGS, "total_time": total_time,
                    "num_seeds": NUM_SEEDS, "sweep_iters": SWEEP_ITERS}, f)

    print(f"Results saved. Run with --report to generate PDF.")


def generate_report_from_results():
    with open(os.path.join(REPORT_DIR, "all_results_3seed.json")) as f:
        data = json.load(f)

    raw = data["results"]
    configs = data["configs"]
    total_time = data["total_time"]
    num_seeds = data["num_seeds"]

    # Organize: config_name -> list of metrics dicts
    by_config = {}
    for key, r in raw.items():
        ci, seed = key.split("_", 1)
        ci = int(ci)
        seed = int(seed)
        name = configs[ci]["name"]
        if name not in by_config:
            by_config[name] = []
        by_config[name].append(r)

    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    config_names = [c["name"] for c in configs]
    config_colors = {name: colors_list[i % len(colors_list)] for i, name in enumerate(config_names)}

    # Helper: get mean and std of a metric across seeds at eval points
    def get_eval_stats(config_name, metric_key):
        runs = by_config.get(config_name, [])
        if not runs:
            return None, None, None
        arrays = []
        iters = None
        for r in runs:
            m = r["metrics"]
            if metric_key in m:
                arrays.append(np.array(m[metric_key], dtype=float))
                if iters is None:
                    iters = np.array(m["eval_iteration"])
        if not arrays:
            return None, None, None
        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]
        if iters is not None:
            iters = iters[:min_len]
        stacked = np.stack(arrays)
        return iters, stacked.mean(axis=0), stacked.std(axis=0)

    def get_iter_stats(config_name, metric_key, smooth_w=20):
        runs = by_config.get(config_name, [])
        if not runs:
            return None, None, None
        arrays = []
        iters = None
        for r in runs:
            m = r["metrics"]
            if metric_key in m:
                arr = smooth(np.array(m[metric_key], dtype=float), smooth_w)
                arrays.append(arr)
                if iters is None:
                    iters = np.array(m["iteration"])
        if not arrays:
            return None, None, None
        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]
        if iters is not None:
            iters = iters[:min_len]
        stacked = np.stack(arrays)
        return iters, stacked.mean(axis=0), stacked.std(axis=0)

    # Plot 1: Elo curves with mean +/- std
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    for name in config_names:
        iters, mean, std = get_eval_stats(name, "elo")
        if iters is None:
            continue
        c = config_colors[name]
        ax.plot(iters, mean, "-o", color=c, markersize=2, linewidth=2, label=name)
        ax.fill_between(iters, mean - std, mean + std, color=c, alpha=0.15)
    ax.axhline(y=1584, color="red", linestyle="--", alpha=0.5, label="Heuristic (1584)")
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Random (1000)")
    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Elo Rating", fontsize=12)
    ax.set_title(f"Elo vs Training Iteration by Pool Config (mean +/- std, {num_seeds} seeds)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "elo_by_config.png"), dpi=150)
    plt.close(fig)

    # Plot 2: Entropy curves
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    for name in config_names:
        iters, mean, std = get_iter_stats(name, "entropy")
        if iters is None:
            continue
        c = config_colors[name]
        ax.plot(iters, mean, "-", color=c, linewidth=1.5, label=name)
        ax.fill_between(iters, mean - std, mean + std, color=c, alpha=0.1)
    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Entropy", fontsize=12)
    ax.set_title("Entropy Over Training by Pool Config", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "entropy_by_config.png"), dpi=150)
    plt.close(fig)

    # Plot 3: Policy + Value loss
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in config_names:
        c = config_colors[name]
        iters, mean, std = get_iter_stats(name, "policy_loss")
        if iters is not None:
            axes[0].plot(iters, mean, "-", color=c, linewidth=1.5, label=name)
            axes[0].fill_between(iters, mean - std, mean + std, color=c, alpha=0.1)
        iters, mean, std = get_iter_stats(name, "value_loss")
        if iters is not None:
            axes[1].plot(iters, mean, "-", color=c, linewidth=1.5, label=name)
            axes[1].fill_between(iters, mean - std, mean + std, color=c, alpha=0.1)
    axes[0].set_title("Policy Loss (smoothed)")
    axes[1].set_title("Value Loss (smoothed)")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Loss Curves by Pool Config", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "losses_by_config.png"), dpi=150)
    plt.close(fig)

    # Plot 4: Final Elo bar chart (mean +/- std)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    final_means = []
    final_stds = []
    peak_means = []
    peak_stds = []
    labels = []
    for name in config_names:
        runs = by_config.get(name, [])
        finals = []
        peaks = []
        for r in runs:
            m = r["metrics"]
            if "elo" in m:
                finals.append(m["elo"][-1])
                peaks.append(max(m["elo"]))
        if finals:
            final_means.append(np.mean(finals))
            final_stds.append(np.std(finals))
            peak_means.append(np.mean(peaks))
            peak_stds.append(np.std(peaks))
        else:
            final_means.append(0)
            final_stds.append(0)
            peak_means.append(0)
            peak_stds.append(0)
        labels.append(name)

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, final_means, w, yerr=final_stds, capsize=4,
                    label="Final Elo @1000", color=[config_colors[n] for n in labels], alpha=0.8)
    bars2 = ax.bar(x + w/2, peak_means, w, yerr=peak_stds, capsize=4,
                    label="Peak Elo", color=[config_colors[n] for n in labels], alpha=0.4,
                    edgecolor=[config_colors[n] for n in labels], linewidth=2)
    ax.bar_label(bars1, [f"{m:.0f}" for m in final_means], fontsize=8, padding=3)
    ax.bar_label(bars2, [f"{m:.0f}" for m in peak_means], fontsize=8, padding=3)
    ax.axhline(y=1584, color="red", linestyle="--", alpha=0.5, label="Heuristic")
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Elo", fontsize=12)
    ax.set_title(f"Final & Peak Elo by Pool Config (mean +/- std, {num_seeds} seeds)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "elo_bar_chart.png"), dpi=150)
    plt.close(fig)

    # Plot 5: vs Heuristic win rate
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    for name in config_names:
        iters, mean, std = get_eval_stats(name, "vs_heuristic_win_rate")
        if iters is None:
            continue
        c = config_colors[name]
        ax.plot(iters, mean, "-o", color=c, markersize=2, linewidth=1.5, label=name)
        ax.fill_between(iters, mean - std, mean + std, color=c, alpha=0.1)
    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title("vs Heuristic Win Rate by Pool Config", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "vs_heuristic_by_config.png"), dpi=150)
    plt.close(fig)

    # Plot 6: Individual seed Elo curves for top configs
    # Find top 3 configs by mean final elo
    sorted_configs = sorted(config_names, key=lambda n: -np.mean([
        r["metrics"]["elo"][-1] for r in by_config.get(n, []) if "elo" in r["metrics"]
    ] or [0]))
    top3 = sorted_configs[:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_idx, name in enumerate(top3):
        ax = axes[ax_idx]
        runs = by_config.get(name, [])
        for si, r in enumerate(runs):
            m = r["metrics"]
            if "elo" in m:
                ax.plot(m["eval_iteration"], m["elo"], "-o", markersize=2,
                        linewidth=1.5, label=f"seed {si}", alpha=0.7)
        ax.axhline(y=1584, color="red", linestyle="--", alpha=0.4)
        ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Elo" if ax_idx == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Individual Seed Elo Curves (Top 3 Configs)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "individual_seeds.png"), dpi=150)
    plt.close(fig)

    print("Plots saved.")

    # Build summary table
    rows = []
    for name in config_names:
        runs = by_config.get(name, [])
        finals = [r["metrics"]["elo"][-1] for r in runs if "elo" in r["metrics"]]
        peaks = [max(r["metrics"]["elo"]) for r in runs if "elo" in r["metrics"]]
        entropies = [r["metrics"]["entropy"][-1] for r in runs if "entropy" in r["metrics"]]
        vr = [r["metrics"]["vs_random_win_rate"][-1] for r in runs if "vs_random_win_rate" in r["metrics"]]
        vh = [r["metrics"]["vs_heuristic_win_rate"][-1] for r in runs if "vs_heuristic_win_rate" in r["metrics"]]
        times = [r["wall_time"] for r in runs]

        cfg = next(c for c in configs if c["name"] == name)
        rows.append([
            name,
            f"{cfg['opponent_pool_max']}",
            f"{cfg['snapshot_interval']}",
            cfg["opponent_sampling"],
            f"{np.mean(finals):.0f} +/- {np.std(finals):.0f}" if finals else "N/A",
            f"{np.mean(peaks):.0f} +/- {np.std(peaks):.0f}" if peaks else "N/A",
            f"{np.mean(entropies):.3f}" if entropies else "N/A",
            f"{np.mean(vr):.0%}" if vr else "N/A",
            f"{np.mean(vh):.0%}" if vh else "N/A",
            f"{np.mean(times):.0f}s" if times else "N/A",
        ])

    # Sort by final elo descending
    rows.sort(key=lambda r: -float(r[4].split(" ")[0]) if r[4] != "N/A" else 0)

    best_name = rows[0][0]
    best_final = rows[0][4]
    best_peak = rows[0][5]

    sections = [
        {
            "heading": "Executive Summary",
            "text": (
                "This report presents a <b>pool configuration sweep</b> for Connect 4 self-play PPO with "
                "<b>reservoir sampling</b>, using Elo against a fixed reference pool as the primary metric. "
                "Each configuration is run with <b>3 random seeds</b> to measure variance.\n\n"
                f"<b>Configurations tested:</b> {len(configs)} pool configs x {num_seeds} seeds = {len(configs)*num_seeds} runs<br/>"
                f"<b>Training:</b> {SWEEP_ITERS} iterations, {BASE_CONFIG['games_per_iter']} games/iter, "
                f"eval every {EVAL_INTERVAL} iters<br/>"
                f"<b>Base config:</b> lr={BASE_CONFIG['lr']}, ent_coef={BASE_CONFIG['ent_coef']}, "
                f"draw_reward={BASE_CONFIG['draw_reward']}, clip_eps={BASE_CONFIG['clip_eps']}<br/>"
                f"<b>Total sweep time:</b> {total_time:.0f}s\n\n"
                f"<b>Best config:</b> {best_name} with final Elo {best_final}, peak Elo {best_peak}"
            ),
            "table": {
                "headers": ["Config", "Pool", "Snap Int", "Sampling", "Final Elo", "Peak Elo",
                            "Entropy", "vs Random", "vs Heur.", "Time"],
                "rows": rows,
            },
            "page_break": True,
        },
        {
            "heading": "Elo Training Curves",
            "text": (
                "Mean Elo +/- 1 standard deviation across 3 seeds. Shaded regions show seed variance. "
                "The heuristic baseline is at Elo 1584, random at 1000."
            ),
            "plots": [
                ("elo_by_config.png", "Elo rating over training for each pool config (mean +/- std)."),
            ],
            "page_break": True,
        },
        {
            "heading": "Final & Peak Elo Comparison",
            "text": (
                "Bar chart comparing final Elo at iteration 1000 and peak Elo across training. "
                "Error bars show standard deviation across seeds."
            ),
            "plots": [
                ("elo_bar_chart.png", "Final and peak Elo by pool config."),
            ],
            "page_break": True,
        },
        {
            "heading": "Individual Seed Curves (Top 3)",
            "text": (
                "Individual Elo curves for each seed of the top 3 configs, showing run-to-run variability."
            ),
            "plots": [
                ("individual_seeds.png", "Elo curves for each seed of the top 3 configurations."),
            ],
            "page_break": True,
        },
        {
            "heading": "vs Heuristic Win Rate",
            "plots": [
                ("vs_heuristic_by_config.png", "Win rate against heuristic opponent (mean +/- std)."),
            ],
            "page_break": True,
        },
        {
            "heading": "Training Dynamics",
            "text": "Entropy and loss curves reveal training stability differences between pool configs.",
            "plots": [
                ("entropy_by_config.png", "Entropy over training by pool config."),
                ("losses_by_config.png", "Policy and value loss by pool config."),
            ],
        },
    ]

    report_path = os.path.join(REPORT_DIR, "pool_sweep_report.pdf")
    generate_pdf_report(
        report_path=report_path,
        title="Connect 4 Self-Play PPO: Pool Config Sweep (3 Seeds)",
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
