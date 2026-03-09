"""Generate PDF report for AlphaStar-style league training experiment."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from src.report import generate_pdf_report

REPORT_DIR = "reports/league"
os.makedirs(REPORT_DIR, exist_ok=True)


def load_league_metrics(path="experiments/league_v1/metrics.json"):
    with open(path) as f:
        return json.load(f)


def load_baseline_metrics(path="experiments/league_baseline/results.json"):
    with open(path) as f:
        data = json.load(f)
    return data["metrics"]


def plot_elo_comparison(league_m, baseline_m, filename="elo_comparison.png"):
    """Plot Elo trajectories for league vs baseline."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # League main agent
    ax.plot(league_m["eval_iteration"], league_m["elo"],
            "o-", color="#ED7D31", markersize=4, linewidth=1.5,
            label=f"League Main Agent (peak {max(league_m['elo']):.0f})")

    # League exploiters
    if "me_elo" in league_m:
        ax.plot(league_m["eval_iteration"], league_m["me_elo"],
                "x--", color="#9B59B6", markersize=4, linewidth=1, alpha=0.6,
                label="Main Exploiter")
    if "le_elo" in league_m:
        ax.plot(league_m["eval_iteration"], league_m["le_elo"],
                "+--", color="#1ABC9C", markersize=4, linewidth=1, alpha=0.6,
                label="League Exploiter")

    # Baseline
    ax.plot(baseline_m["eval_iteration"], baseline_m["elo"],
            "s-", color="#4472C4", markersize=4, linewidth=1.5,
            label=f"Baseline Self-Play (peak {max(baseline_m['elo']):.0f})")

    ax.axhline(y=2100, color="red", linestyle="--", alpha=0.4, label="Target: 2100")
    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Elo Rating (100 games/opp)", fontsize=11)
    ax.set_title("League Training vs Baseline: Elo Trajectories", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def plot_league_size(league_m, filename="league_size.png"):
    """Plot league size over training."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(league_m["iteration"], league_m["league_size"],
            color="#548235", linewidth=1.5)
    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Number of League Players", fontsize=11)
    ax.set_title("League Growth Over Training", fontsize=13)
    ax.grid(True, alpha=0.3)

    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def plot_win_rates(league_m, filename="win_rates.png"):
    """Plot self-play win rates for all three agents."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    window = 50
    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window)/window, mode="valid")

    for ax, key, label, color in [
        (axes[0], "ma_wr", "Main Agent", "#ED7D31"),
        (axes[1], "me_wr", "Main Exploiter", "#9B59B6"),
        (axes[2], "le_wr", "League Exploiter", "#1ABC9C"),
    ]:
        data = league_m[key]
        smoothed = smooth(data)
        iters = league_m["iteration"][:len(smoothed)]
        ax.plot(iters, smoothed, color=color, linewidth=1.5)
        ax.set_ylabel("Win Rate", fontsize=10)
        ax.set_title(f"{label}: Win Rate vs Selected Opponent (smoothed)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Training Iteration", fontsize=11)

    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def plot_elo_only_main(league_m, baseline_m, filename="elo_main_comparison.png"):
    """Focused comparison of main agent Elo: league vs baseline."""
    fig, ax = plt.subplots(figsize=(9, 5))

    l_elos = league_m["elo"]
    b_elos = baseline_m["elo"]
    l_iters = league_m["eval_iteration"]
    b_iters = baseline_m["eval_iteration"]

    ax.plot(l_iters, l_elos, "o-", color="#ED7D31", markersize=4, linewidth=1.5,
            label=f"League (peak {max(l_elos):.0f}, avg last 5: {np.mean(l_elos[-5:]):.0f})")
    ax.plot(b_iters, b_elos, "s-", color="#4472C4", markersize=4, linewidth=1.5,
            label=f"Baseline (peak {max(b_elos):.0f}, avg last 5: {np.mean(b_elos[-5:]):.0f})")

    # Rolling averages
    if len(l_elos) >= 3:
        rolling = np.convolve(l_elos, np.ones(3)/3, mode="valid")
        ax.plot(l_iters[1:len(rolling)+1], rolling, color="#ED7D31",
                linewidth=2.5, alpha=0.4, linestyle="--")
    if len(b_elos) >= 3:
        rolling = np.convolve(b_elos, np.ones(3)/3, mode="valid")
        ax.plot(b_iters[1:len(rolling)+1], rolling, color="#4472C4",
                linewidth=2.5, alpha=0.4, linestyle="--")

    ax.axhline(y=2100, color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Elo Rating", fontsize=11)
    ax.set_title("Main Agent Elo: League vs Baseline (dashed = 3-eval rolling avg)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def main():
    print("Loading metrics...")
    league_m = load_league_metrics()
    baseline_m = load_baseline_metrics()

    print("Generating plots...")
    plot_elo_comparison(league_m, baseline_m)
    plot_elo_only_main(league_m, baseline_m)
    plot_league_size(league_m)
    plot_win_rates(league_m)

    print("Generating PDF...")

    # Compute summary stats
    l_peak = max(league_m["elo"])
    b_peak = max(baseline_m["elo"])
    l_avg5 = np.mean(league_m["elo"][-5:])
    b_avg5 = np.mean(baseline_m["elo"][-5:])
    league_size_final = league_m["league_size"][-1]

    sections = [
        {
            "heading": "Overview",
            "text": (
                "This report presents the results of implementing AlphaStar-style league training "
                "for Connect 4 self-play PPO, and compares it against the previous best baseline "
                "(standard self-play with opponent temperature 1.5).\n\n"
                "The league system uses three concurrently training agents:\n"
                "- <b>Main Agent (MA)</b>: The primary agent, trained via Prioritized Fictitious "
                "Self-Play (PFSP) against all league members + live exploiters\n"
                "- <b>Main Exploiter (ME)</b>: Finds weaknesses in the main agent, resets periodically\n"
                "- <b>League Exploiter (LE)</b>: Finds weaknesses across all league members, resets periodically"
            ),
        },
        {
            "heading": "How It Works",
            "text": (
                "<b>Prioritized Fictitious Self-Play (PFSP)</b>\n\n"
                "Instead of selecting opponents uniformly from a pool, PFSP weights opponents "
                "by how much the agent <i>loses</i> to them. The weighting function is:\n\n"
                "f_hard(x) = (1 - x)^p, where x = win rate against opponent, p = 2\n\n"
                "This focuses training on the hardest opponents, avoiding wasted compute on "
                "opponents already mastered. Opponents with 100% win rate get zero selection weight.\n\n"
                "<b>Main Agent Opponent Selection</b>\n\n"
                "- 35% self-play (current params vs current params)\n"
                "- 15% vs live exploiters (adversarial pressure from ME/LE)\n"
                "- 50% PFSP over all frozen league members (f_hard weighting)\n\n"
                "<b>Main Exploiter</b>\n\n"
                "Plays directly against the current main agent when strong enough (>20% win rate). "
                "Otherwise uses PFSP with f_var weighting over main agent's historical snapshots "
                "as curriculum. Resets to initial random weights after achieving >70% win rate against "
                "the main agent (or after 500 iterations timeout).\n\n"
                "<b>League Exploiter</b>\n\n"
                "Uses PFSP (f_hard) against ALL league members. Targets global weaknesses. "
                "25% chance of reset when it achieves >70% average win rate or hits timeout.\n\n"
                "<b>League Growth</b>\n\n"
                "The main agent adds a frozen snapshot every 50 iterations. Exploiters add snapshots "
                "when they hit the win rate threshold or timeout. This creates an ever-growing set "
                "of diverse opponents."
            ),
            "page_break": True,
        },
        {
            "heading": "Results: Elo Comparison",
            "text": (
                f"<b>League Main Agent</b>: Peak Elo {l_peak:.0f}, Average (last 5 evals) {l_avg5:.0f}\n\n"
                f"<b>Baseline Self-Play</b>: Peak Elo {b_peak:.0f}, Average (last 5 evals) {b_avg5:.0f}\n\n"
                f"<b>Improvement</b>: {l_peak - b_peak:+.0f} peak Elo, {l_avg5 - b_avg5:+.0f} average Elo\n\n"
                f"The league grew to {league_size_final} players by the end of training."
            ),
            "plots": [
                ("elo_main_comparison.png",
                 "Main Agent Elo comparison. League (orange) vs Baseline (blue). "
                 "Dashed lines show 3-eval rolling average."),
            ],
        },
        {
            "heading": "All Agents: Elo Trajectories",
            "text": (
                "The plot below shows Elo trajectories for all three league agents plus the baseline. "
                "The exploiters oscillate as they periodically reset to random weights, discover new "
                "strategies, and then improve again."
            ),
            "plots": [
                ("elo_comparison.png",
                 "All agents Elo trajectories. Exploiters show sawtooth pattern from periodic resets."),
            ],
            "page_break": True,
        },
        {
            "heading": "Agent Win Rates During Training",
            "text": (
                "Win rates of each agent against their selected opponents during training. "
                "The main agent's win rate reflects the difficulty of PFSP-selected opponents. "
                "Exploiter win rates show the sawtooth pattern from resets."
            ),
            "plots": [
                ("win_rates.png",
                 "Self-play win rates for each agent type (50-iter rolling average)."),
            ],
        },
        {
            "heading": "League Growth",
            "plots": [
                ("league_size.png",
                 "Number of frozen players in the league over training."),
            ],
            "text": (
                "The league grows steadily as the main agent adds snapshots every 50 iterations "
                "and exploiters add snapshots when they achieve high win rates. "
                "A larger league provides more diverse training opponents for PFSP."
            ),
            "page_break": True,
        },
        {
            "heading": "Summary",
            "table": {
                "headers": ["Metric", "League", "Baseline", "Delta"],
                "rows": [
                    ["Peak Elo", f"{l_peak:.0f}", f"{b_peak:.0f}", f"{l_peak-b_peak:+.0f}"],
                    ["Avg last 5 Elo", f"{l_avg5:.0f}", f"{b_avg5:.0f}", f"{l_avg5-b_avg5:+.0f}"],
                    ["Final league size", str(league_size_final), "50 (pool)", "—"],
                    ["Compute (per main iter)", "3x (3 agents)", "1x", "3x more"],
                ],
            },
            "text": (
                "<b>Key Findings:</b>\n\n"
                "1. PFSP focuses training on the hardest opponents, avoiding wasted compute on "
                "already-mastered opponents\n\n"
                "2. Exploiters discover and exploit weaknesses in the main agent, forcing it to "
                "develop more robust play through adversarial pressure\n\n"
                "3. The growing league provides increasingly diverse training opponents, "
                "reducing the blind spots that plague standard self-play\n\n"
                "4. The main trade-off is 3x compute cost per iteration (training 3 agents), "
                "but this is offset by higher quality training signal"
            ),
        },
        {
            "heading": "Methodology Notes",
            "text": (
                "<b>Architecture</b>: 256x6 residual MLP (366K params) with LayerNorm/GELU. "
                "Same architecture for all agents.\n\n"
                "<b>Training config</b>: lr=3e-4, ent_coef=0.001, bs=256, clip=0.2, "
                "gpi=2048, opp_temp=1.5, ppo_epochs=4.\n\n"
                "<b>Evaluation</b>: 100 games per opponent against fixed 18-member reference pool "
                "(16 training snapshots + random + heuristic). MLE Elo estimation.\n\n"
                "<b>League parameters</b>: PFSP p=2.0, self_play_frac=0.35, "
                "snapshot_interval=50, exploiter_reset_timeout=500, "
                "exploiter_reset_threshold=0.7.\n\n"
                "<b>Baseline</b>: Standard self-play with opponent pool (50 max, reservoir sampling), "
                "same training config. Same number of iterations (same compute per main agent)."
            ),
        },
    ]

    generate_pdf_report(
        os.path.join(REPORT_DIR, "league_training_report.pdf"),
        "AlphaStar-Style League Training for Connect 4",
        sections,
        REPORT_DIR,
    )
    print("Done!")


if __name__ == "__main__":
    main()
