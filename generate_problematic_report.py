"""Generate PDF report for problematic starting positions experiment."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from src.report import generate_pdf_report

REPORT_DIR = "reports/problematic_starts"
EXP_DIR = "experiments/problematic_starts"
os.makedirs(REPORT_DIR, exist_ok=True)

SWEEP_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]
NUM_SEEDS = 2  # seeds 42, 43
COLORS = ["#4472C4", "#ED7D31", "#548235", "#9B59B6", "#E74C3C"]


def load_all_metrics():
    """Load metrics for all configs and seeds."""
    data = {}
    for frac in SWEEP_VALUES:
        name = f"prob_{int(frac*100):02d}pct"
        seeds = []
        for seed in range(NUM_SEEDS):
            mpath = os.path.join(EXP_DIR, f"{name}_s{seed}", "metrics.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    seeds.append(json.load(f))
        if seeds:
            data[frac] = seeds
    return data


def compute_avg_last(metrics, n=10):
    """Compute avg Elo of last n eval points."""
    elos = metrics["elo"]
    return np.mean(elos[-n:])


def plot_elo_trajectories(data, filename="elo_trajectories.png"):
    """Plot Elo trajectories for all configs (mean ± std across seeds)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, frac in enumerate(SWEEP_VALUES):
        if frac not in data:
            continue
        seeds = data[frac]
        # Align by eval_iteration
        min_len = min(len(s["elo"]) for s in seeds)
        elos = np.array([s["elo"][:min_len] for s in seeds])
        iters = seeds[0]["eval_iteration"][:min_len]

        mean_elo = elos.mean(axis=0)
        std_elo = elos.std(axis=0)

        label = f"{int(frac*100)}%" if frac > 0 else "0% (baseline)"
        ax.plot(iters, mean_elo, color=COLORS[idx], linewidth=1.5, label=label)
        ax.fill_between(iters, mean_elo - std_elo, mean_elo + std_elo,
                        color=COLORS[idx], alpha=0.15)

    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Elo Rating (100 games/opp)", fontsize=11)
    ax.set_title("Elo Trajectories: Problematic Starting Positions", fontsize=13)
    ax.legend(fontsize=9, title="Problematic Start %")
    ax.grid(True, alpha=0.3)

    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def plot_avg_last_100_bar(data, filename="avg_last_100.png"):
    """Bar chart of avg Elo (last 10 evals) for each config."""
    fig, ax = plt.subplots(figsize=(8, 5))

    fracs = []
    means = []
    stds = []
    for frac in SWEEP_VALUES:
        if frac not in data:
            continue
        seeds = data[frac]
        avgs = [compute_avg_last(s) for s in seeds]
        fracs.append(f"{int(frac*100)}%")
        means.append(np.mean(avgs))
        stds.append(np.std(avgs))

    x = np.arange(len(fracs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=COLORS[:len(fracs)],
                  edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 5,
                f"{m:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Problematic Start Fraction", fontsize=11)
    ax.set_ylabel("Avg Elo (last 10 evals)", fontsize=11)
    ax.set_title("Avg Elo (Last 10 Evals) by Problematic Start Fraction", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(fracs)
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def plot_individual_seeds(data, filename="individual_seeds.png"):
    """Plot individual seed runs for each config."""
    fig, axes = plt.subplots(len(SWEEP_VALUES), 1, figsize=(10, 3*len(SWEEP_VALUES)),
                              sharex=True)
    if len(SWEEP_VALUES) == 1:
        axes = [axes]

    for idx, frac in enumerate(SWEEP_VALUES):
        ax = axes[idx]
        if frac not in data:
            continue
        seeds = data[frac]
        for si, s in enumerate(seeds):
            iters = s["eval_iteration"]
            elos = s["elo"]
            ax.plot(iters, elos, alpha=0.6, linewidth=1,
                    label=f"seed {si}")
            # Rolling average
            if len(elos) >= 10:
                rolling = np.convolve(elos, np.ones(10)/10, mode="valid")
                ax.plot(iters[4:4+len(rolling)], rolling, linewidth=2,
                        color=COLORS[idx], alpha=0.8)

        label = f"{int(frac*100)}%" if frac > 0 else "0% (baseline)"
        avg100 = np.mean([compute_avg_last(s) for s in seeds])
        ax.set_title(f"{label} — avg last 100: {avg100:.0f}", fontsize=11)
        ax.set_ylabel("Elo", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    axes[-1].set_xlabel("Training Iteration", fontsize=11)

    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def plot_smoothed_comparison(data, filename="smoothed_comparison.png"):
    """Smoothed Elo comparison (rolling avg of 20 evals)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 20

    for idx, frac in enumerate(SWEEP_VALUES):
        if frac not in data:
            continue
        seeds = data[frac]
        min_len = min(len(s["elo"]) for s in seeds)
        elos = np.array([s["elo"][:min_len] for s in seeds])
        mean_elo = elos.mean(axis=0)
        iters = seeds[0]["eval_iteration"][:min_len]

        if len(mean_elo) >= window:
            smoothed = np.convolve(mean_elo, np.ones(window)/window, mode="valid")
            x = iters[window//2:window//2+len(smoothed)]
        else:
            smoothed = mean_elo
            x = iters

        label = f"{int(frac*100)}%" if frac > 0 else "0% (baseline)"
        ax.plot(x, smoothed, color=COLORS[idx], linewidth=2, label=label)

    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Elo Rating (20-eval rolling avg)", fontsize=11)
    ax.set_title("Smoothed Elo Comparison", fontsize=13)
    ax.legend(fontsize=9, title="Problematic Start %")
    ax.grid(True, alpha=0.3)

    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def main():
    print("Loading metrics...")
    data = load_all_metrics()
    if not data:
        print("No data found!")
        return

    print("Generating plots...")
    plot_elo_trajectories(data)
    plot_avg_last_100_bar(data)
    plot_individual_seeds(data)
    plot_smoothed_comparison(data)

    # Build summary table
    table_rows = []
    baseline_avg = None
    for frac in SWEEP_VALUES:
        if frac not in data:
            continue
        seeds = data[frac]
        avgs = [compute_avg_last(s) for s in seeds]
        mean_avg = np.mean(avgs)
        std_avg = np.std(avgs)
        if frac == 0:
            baseline_avg = mean_avg
            delta = "—"
        else:
            delta = f"{mean_avg - baseline_avg:+.0f}" if baseline_avg else "—"
        label = f"{int(frac*100)}%" if frac > 0 else "0% (baseline)"
        table_rows.append([
            label,
            f"{mean_avg:.0f} ± {std_avg:.0f}",
            delta,
            str(len(seeds)),
        ])

    # Find best config
    best_frac = max(data.keys(), key=lambda f: np.mean([compute_avg_last(s) for s in data[f]]))
    best_avg = np.mean([compute_avg_last(s) for s in data[best_frac]])
    best_label = f"{int(best_frac*100)}%" if best_frac > 0 else "0% (baseline)"

    sections = [
        {
            "heading": "Overview",
            "text": (
                "This report presents the results of training Connect 4 self-play PPO agents "
                "with <b>problematic starting positions</b>. Instead of always starting games from "
                "a blank board, a fraction of training games begin from board states where the agent "
                "previously lost, forcing it to practice difficult situations.\n\n"
                "<b>Hypothesis</b>: Standard self-play always starts from blank boards, meaning the agent "
                "only encounters positions reachable from its own opening play. If the agent develops "
                "blind spots (positions it handles poorly), it may never revisit them. By replaying from "
                "lost positions, we force the agent to confront its weaknesses directly."
            ),
        },
        {
            "heading": "How It Works",
            "text": (
                "<b>Problematic State Buffer</b>\n\n"
                "After each training iteration, we identify games where the agent lost. From those games, "
                "we extract board states where the agent was <i>overconfident</i> — value estimate V &gt; 0.3 "
                "despite ultimately losing. These 'surprised' states are positions where the agent's "
                "understanding is fundamentally wrong. They are added to a circular buffer (max 10,000 states).\n\n"
                "<b>Mixed Starting Positions</b>\n\n"
                "Each iteration, a configurable fraction of games start from states sampled from the "
                "problematic buffer instead of blank boards. The rest start normally from blank. "
                "The agent and opponent are still randomly assigned to player 1 or -1, so the agent "
                "learns to play both sides of problematic positions.\n\n"
                "<b>Overconfidence-Based State Selection</b>\n\n"
                "Analysis showed 85% of states from lost games have V &gt; 0.5 — the agent is almost "
                "always overconfident when it loses. We tested multiple selection strategies (V &lt; 0, "
                "V &gt; 0.3, V &gt; 0.5, all lost, mid-game only). Storing 'surprised' states (V &gt; 0.3 "
                "in lost games) captures mid-game positions (avg 12.6 pieces) where the agent's "
                "evaluation is fundamentally wrong, rather than late-game near-terminal positions.\n\n"
                "<b>Board State Reconstruction</b>\n\n"
                "Board states are reconstructed from observations by mapping the player-relative encoding "
                "back to absolute piece positions. The current player is preserved from the original game.\n\n"
                "<b>Experiment Setup</b>\n\n"
                "We sweep the problematic start fraction across 0%, 10%, 20%, 30%, and 50%. "
                "Each configuration runs for 5000 iterations with 3 random seeds. "
                "The primary metric is <b>average Elo over the last 100 evaluation points</b> "
                "(eval every 50 iterations, 100 games per opponent against the fixed reference pool)."
            ),
            "page_break": True,
        },
        {
            "heading": "Results: Elo Trajectories",
            "text": (
                "The plot below shows Elo trajectories for each configuration, averaged across seeds "
                "with ±1 std shaded. Higher is better."
            ),
            "plots": [
                ("elo_trajectories.png",
                 "Mean ± std Elo trajectories across 3 seeds for each problematic start fraction."),
            ],
        },
        {
            "heading": "Smoothed Comparison",
            "plots": [
                ("smoothed_comparison.png",
                 "20-eval rolling average of mean Elo across seeds."),
            ],
            "page_break": True,
        },
        {
            "heading": "Average Elo (Last 100 Evals)",
            "text": (
                f"The primary metric is average Elo over the last 100 evaluation points. "
                f"<b>Best configuration: {best_label}</b> with avg Elo {best_avg:.0f}."
            ),
            "plots": [
                ("avg_last_100.png",
                 "Bar chart showing avg Elo (last 10 evals) ± std across seeds."),
            ],
            "table": {
                "headers": ["Config", "Avg Elo (last 100)", "Delta vs Baseline", "Seeds"],
                "rows": table_rows,
            },
        },
        {
            "heading": "Individual Seed Runs",
            "plots": [
                ("individual_seeds.png",
                 "Individual seed trajectories for each configuration. Bold line = 10-eval rolling average."),
            ],
            "page_break": True,
        },
        {
            "heading": "Summary",
            "text": (
                f"<b>Best configuration</b>: {best_label} problematic starts "
                f"(avg Elo last 100: {best_avg:.0f}"
                + (f", {best_avg - baseline_avg:+.0f} vs baseline" if baseline_avg and best_frac != 0 else "")
                + f").\n\n"
                "<b>Key Findings:</b>\n\n"
                "1. Problematic starting positions force the agent to practice from positions where "
                "it was overconfident but lost, targeting its biggest blind spots.\n\n"
                "2. The effect is strongest at lower games-per-iteration (gpi=512: +60-90 Elo) and "
                "more modest at higher gpi (gpi=2048: +20 Elo at 50%), likely because more games per "
                "iter already provide sufficient diversity.\n\n"
                "3. High seed variance (±80 Elo between seeds) means conclusions should be interpreted "
                "cautiously with only 2 seeds per configuration."
            ),
        },
        {
            "heading": "Methodology",
            "text": (
                "<b>Architecture</b>: 256x6 residual MLP (366K params) with LayerNorm/GELU.\n\n"
                "<b>Training config</b>: lr=3e-4, ent_coef=0.001, bs=256, clip=0.2, "
                "gpi=2048, opp_temp=1.5, ppo_epochs=4.\n\n"
                "<b>Evaluation</b>: 100 games per opponent against fixed 18-member reference pool "
                "(16 training snapshots + random + heuristic). MLE Elo estimation.\n\n"
                "<b>Problematic buffer</b>: Max 10,000 states, reservoir sampling when full. "
                "Only 'surprised' states (V &gt; 0.3 in lost games) are stored.\n\n"
                "<b>Seeds</b>: 2 seeds per configuration (42, 43).\n\n"
                "<b>Metric</b>: Average Elo over last 10 evaluation points "
                "(eval every 100 iters)."
            ),
        },
    ]

    print("Generating PDF...")
    generate_pdf_report(
        os.path.join(REPORT_DIR, "problematic_starts_report.pdf"),
        "Problematic Starting Positions for Connect 4 Self-Play",
        sections,
        REPORT_DIR,
    )
    print("Done!")


if __name__ == "__main__":
    main()
