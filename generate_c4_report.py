"""Generate PDF report of Connect 4 hyperparameter optimization experiments."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from src.report import generate_pdf_report

REPORT_DIR = "reports/c4_optimization"
os.makedirs(REPORT_DIR, exist_ok=True)


def load_exp_evals(name):
    """Load eval trajectory from train.log (grep EVAL lines)."""
    log_path = f"experiments/{name}/train.log"
    iters, elos, heur_wr = [], [], []
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            if "EVAL @" not in line:
                continue
            parts = line.strip().split("|")
            try:
                it = int(line.split("EVAL @")[1].split(":")[0].strip())
                elo = None
                hw = None
                for p in parts:
                    p = p.strip()
                    if p.startswith("Elo:"):
                        elo = int(p.split(":")[1].strip())
                    elif "vs Heuristic W:" in p:
                        hw = float(p.split("W:")[1].strip().split()[0])
                if elo is not None:
                    iters.append(it)
                    elos.append(elo)
                    heur_wr.append(hw if hw is not None else 0)
            except (ValueError, IndexError):
                continue
    if not iters:
        return None
    return {"iters": iters, "elos": elos, "heur_wr": heur_wr}


def plot_elo_trajectory(experiments, title, filename):
    """Plot Elo trajectories for multiple experiments."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, label, color in experiments:
        data = load_exp_evals(name)
        if data is None:
            continue
        ax.plot(data["iters"], data["elos"], label=f"{label} (peak {max(data['elos'])})",
                color=color, linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Training Iteration", fontsize=11)
    ax.set_ylabel("Elo Rating (100 games/opp)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=2100, color="red", linestyle="--", alpha=0.5)
    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def plot_bar_chart(labels, values, title, ylabel, filename, highlight_idx=None):
    """Bar chart comparing peak Elos."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_list = ["#4472C4"] * len(labels)
    if highlight_idx is not None:
        colors_list[highlight_idx] = "#ED7D31"
    bars = ax.bar(range(len(labels)), values, color=colors_list)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=2100, color="red", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(int(val)), ha="center", va="bottom", fontsize=8)
    path = os.path.join(REPORT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def plot_exp18_trajectory():
    """Detailed plot of the best long run."""
    data = load_exp_evals("exp18_temp15_best")
    if data is None:
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    ax1.plot(data["iters"], data["elos"], "o-", color="#4472C4", markersize=3, linewidth=1.2)
    if len(data["elos"]) >= 5:
        rolling = np.convolve(data["elos"], np.ones(5)/5, mode="valid")
        roll_iters = data["iters"][2:len(rolling)+2]
        ax1.plot(roll_iters, rolling, color="#ED7D31", linewidth=2, label="5-eval rolling avg")
    ax1.axhline(y=2100, color="red", linestyle="--", alpha=0.5, label="Target: 2100")
    peak_idx = np.argmax(data["elos"])
    ax1.annotate(f"Peak: {data['elos'][peak_idx]}",
                 xy=(data["iters"][peak_idx], data["elos"][peak_idx]),
                 xytext=(data["iters"][peak_idx]+500, data["elos"][peak_idx]+30),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=10, fontweight="bold")
    ax1.set_ylabel("Elo Rating", fontsize=11)
    ax1.set_title("Experiment 18: Best Config Long Run (opp_temp=1.5, 10k iters)", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(data["iters"], data["heur_wr"], "s-", color="#548235", markersize=3, linewidth=1.2)
    ax2.set_xlabel("Training Iteration", fontsize=11)
    ax2.set_ylabel("Win Rate vs Heuristic", fontsize=11)
    ax2.set_title("Win Rate vs Heuristic Opponent (Elo ~1584)", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    fname = "exp18_detail.png"
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, fname), dpi=150)
    plt.close(fig)
    return fname


# --- Generate all plots ---

print("Generating plots...")

plot_elo_trajectory([
    ("exp5_gpi2048_5k", "Baseline gpi=2048", "#4472C4"),
    ("exp9b_cosine_lr", "Cosine LR", "#548235"),
    ("exp16a_opp_temp15", "opp_temp=1.5", "#ED7D31"),
    ("exp16b_opp_temp20", "opp_temp=2.0", "#A5A5A5"),
], "Key Experiments: Elo Trajectories", "key_experiments.png")

plot_elo_trajectory([
    ("exp5_gpi2048_5k", "Baseline gpi=2048", "#4472C4"),
    ("exp13_gpi4096_bs256", "gpi=4096 bs=256", "#FF6B6B"),
    ("exp10_stable", "clip=0.1 conservative", "#A5A5A5"),
    ("exp15a_ent_schedule", "Ent schedule", "#9B59B6"),
    ("exp15b_multi_opp", "Multi-opp (4/iter)", "#1ABC9C"),
], "Negative Results: Approaches That Didn't Work", "negative_results.png")

plot_exp18_trajectory()

configs = [
    ("gpi=512\nbaseline", 1722),
    ("gpi=1024", 1677),
    ("gpi=2048", 1903),
    ("gpi=2048\n5k iters", 1879),
    ("bs=512", 1534),
    ("gpi=4096\nbs=256", 1644),
    ("clip=0.1", 1717),
    ("8 PPO\nepochs", 1477),
    ("2 PPO\nepochs", 1605),
    ("cosine LR", 1867),
    ("ent sched\n.01->.001", 1586),
    ("multi-opp\n4/iter", 1723),
    ("temp=1.5", 1970),
    ("temp=2.0", 1760),
    ("temp anneal\n1.5->1.0", 1632),
    ("exp18\nbest run", 1943),
]
labels, values = zip(*configs)
plot_bar_chart(list(labels), list(values), "Peak Elo by Configuration",
               "Peak Elo (100 games/opp)", "peak_elo_comparison.png",
               highlight_idx=12)

plot_elo_trajectory([
    ("exp8b_gpi2048_bs256_15k", "Baseline 15k", "#4472C4"),
    ("exp14_long20k", "Baseline 20k", "#548235"),
    ("exp18_temp15_best", "Best config 10k", "#ED7D31"),
], "Long Run Comparison", "long_runs.png")

print("Generating PDF...")

# --- Build PDF ---

sections = [
    {
        "heading": "Overview",
        "text": (
            "This report documents a systematic hyperparameter optimization campaign for "
            "Connect 4 self-play PPO. The goal was to reach 2100 stable Elo against a fixed "
            "reference pool of 18 opponents (16 training snapshots + random + heuristic).\n\n"
            "The agent uses a residual MLP (256x6, 366K params) with LayerNorm and GELU "
            "activations, trained with PPO via a pure C backend using Apple Accelerate BLAS. "
            "Training speed is ~1 second per iteration (2048 games) on an M2 Max.\n\n"
            "<b>Result: Peak 1970 Elo achieved. Target of 2100 was not reached.</b>"
        ),
    },
    {
        "heading": "Peak Elo by Configuration",
        "text": (
            "The bar chart below shows peak Elo (measured with 100 games per opponent) "
            "across all configurations tested. The orange bar highlights the best result: "
            "opponent temperature 1.5, which achieved 1970 Elo."
        ),
        "plots": [("peak_elo_comparison.png",
                    "Peak Elo across all configurations. Red dashed line = 2100 target.")],
        "page_break": True,
    },
    {
        "heading": "What Worked",
        "text": (
            "<b>1. Games per iteration = 2048 (+180 Elo)</b>\n\n"
            "Increasing from 512 to 2048 games per iteration was the single largest improvement. "
            "More games means better gradient estimates. The agent collects ~20,000 transitions per "
            "iteration, processed with batch size 256 giving ~8 gradient steps per data point.\n\n"
            "<b>2. Opponent temperature = 1.5 (+80 Elo)</b>\n\n"
            "Scaling opponent logits by 1/1.5 before softmax during self-play makes the opponent "
            "play more stochastically. This creates diverse board states the agent wouldn't see "
            "with deterministic self-play, reducing blind spots. This was the breakthrough finding, "
            "pushing peak Elo from ~1890 to 1970.\n\n"
            "<b>3. Batch size 256 (essential)</b>\n\n"
            "Batch size 256 gives the right number of gradient steps per iteration. "
            "bs=512 was significantly worse (fewer updates), and implicit analysis showed "
            "more gradient steps (gpi=4096 with bs=256) caused overshooting."
        ),
        "table": {
            "headers": ["Technique", "Peak Elo", "Improvement", "Status"],
            "rows": [
                ["gpi=2048", "1903", "+181", "Essential"],
                ["opp_temp=1.5", "1970", "+80", "Best result"],
                ["bs=256", "—", "—", "Essential"],
                ["clip_eps=0.2", "—", "—", "Essential"],
                ["Cosine LR", "1867/1908", "+20-30", "Marginal"],
            ],
        },
    },
    {
        "heading": "Key Experiments: Elo Trajectories",
        "text": (
            "The plot below shows Elo trajectories for the most important experiments. "
            "opp_temp=1.5 (orange) clearly dominates, reaching 1970 at iteration 3500. "
            "All configs show significant oscillation (+-70-90 Elo) inherent to self-play PPO."
        ),
        "plots": [("key_experiments.png",
                    "Elo trajectories for key experiments. Red dashed line = 2100 target.")],
        "page_break": True,
    },
    {
        "heading": "What Didn't Work",
        "text": (
            "<b>Higher entropy coefficient (0.01+):</b> Too much randomness in the policy.\n\n"
            "<b>Larger models (512x6, 256x8):</b> Overfit with limited data diversity.\n\n"
            "<b>gpi=4096 with bs=256:</b> ~1875 gradient steps/iter causes policy overshooting.\n\n"
            "<b>clip_eps=0.1:</b> Too conservative — agent stops learning and plateaus at ~1710.\n\n"
            "<b>8 PPO epochs:</b> KL divergence 0.10-0.13, policy diverges.\n\n"
            "<b>Mirror augmentation:</b> Mirrored transitions break importance sampling — "
            "the log-prob ratios explode, causing NaN within 200 iterations.\n\n"
            "<b>Multi-opponent (4/iter):</b> Stable ~1720 but lower ceiling. "
            "More diverse but weaker per-opponent signal.\n\n"
            "<b>opp_temp=2.0:</b> Opponent too random, doesn't challenge agent enough.\n\n"
            "<b>Temperature annealing (1.5->1.0):</b> Curriculum didn't help — constant is better."
        ),
        "plots": [("negative_results.png",
                    "Elo trajectories for approaches that didn't improve over baseline.")],
        "page_break": True,
    },
    {
        "heading": "Best Run: Experiment 18 (Detailed)",
        "text": (
            "The best configuration (gpi=2048, bs=256, opp_temp=1.5) was run for ~9500 iterations "
            "with evaluation every 250 iterations (100 games per opponent).\n\n"
            "Peak Elo: 1943 at iteration 8000. Average of last 10 evaluations: 1869. "
            "The oscillation range was 1766-1943 (+-90 Elo).\n\n"
            "Heuristic win rate stabilized at ~100% after iteration 4000, "
            "indicating the agent fully solved this opponent."
        ),
        "plots": [("exp18_detail.png",
                    "Experiment 18 detailed trajectory. Top: Elo with rolling average. "
                    "Bottom: heuristic win rate.")],
    },
    {
        "heading": "Long Run Comparison",
        "plots": [("long_runs.png",
                    "Comparing baseline and best config over extended training.")],
        "text": (
            "Extended training (10k-20k iters) with the baseline config plateaus at ~1890 peak Elo. "
            "The best config (opp_temp=1.5) consistently runs ~50-80 Elo higher."
        ),
        "page_break": True,
    },
    {
        "heading": "Why 2100 Wasn't Reached",
        "text": (
            "<b>1. Architecture mismatch:</b> The agent uses a residual MLP with LayerNorm/GELU, "
            "but reference pool opponents use a plain MLP with ReLU. The agent develops blind spots "
            "against play patterns from the different architecture, particularly losing to "
            "mid-range opponents (iter_1400, iter_2600) at 78-80% win rate even while beating "
            "the strongest opponents at 90%+.\n\n"
            "<b>2. Self-play oscillation:</b> PPO with self-play inherently oscillates because "
            "each training step changes both the agent and its future opponents. With clip=0.2, "
            "policies shift significantly in one update. Conservative clipping (0.1) stops this "
            "but also stops learning.\n\n"
            "<b>3. Limited pool diversity:</b> 18 opponents provide a narrow evaluation signal. "
            "The agent can overfit to specific opponent patterns rather than learning generally "
            "strong play.\n\n"
            "<b>4. Gradient efficiency ceiling:</b> With bs=256 and gpi=2048, each iteration "
            "does ~8 gradient steps. More steps (gpi=4096) causes overshooting. Fewer steps "
            "(bs=512 or ppo_epochs=2) gives insufficient learning. This appears to be the "
            "optimal operating point for this architecture."
        ),
    },
    {
        "heading": "All Experiments Summary",
        "table": {
            "headers": ["Experiment", "Config", "Peak Elo", "Notes"],
            "rows": [
                ["Exp 1", "Entropy sweep", "1722", "ent=0.001 best"],
                ["Exp 2", "Draw reward", "1726", "No significant effect"],
                ["Exp 3", "gpi + model size", "1903", "gpi=2048 breakthrough"],
                ["Exp 4", "Pinned opponents", "1735", "Pinning hurts"],
                ["Exp 5", "Long gpi=2048 (5k)", "1879", "1837 true Elo"],
                ["Exp 8", "Long runs (15k)", "1901", "Plateau at ~1830"],
                ["Exp 9", "Cosine LR + gpi=4096", "1867", "Det eval: 1908"],
                ["Exp 10", "Conservative (clip=0.1)", "1717", "Stable but stuck"],
                ["Exp 12", "Mirror augmentation", "1805", "NaN with bs>256"],
                ["Exp 13", "gpi=4096, bs=256", "1644", "Too many grad steps"],
                ["Exp 14", "Long 20k + ppo2", "1886", "Confirms ceiling"],
                ["Exp 15", "Ent schedule + multi-opp", "1723", "Both worse"],
                ["Exp 16", "opp_temp=1.5/2.0", "1970", "BEST RESULT"],
                ["Exp 17", "Temp anneal + combos", "1632", "Worse than constant"],
                ["Exp 18", "Best config 10k", "1943", "Sustained ~1870"],
            ],
        },
        "page_break": True,
    },
    {
        "heading": "Untested Ideas",
        "text": (
            "Several promising approaches were not attempted due to time constraints:\n\n"
            "<b>Fine-tune against reference pool:</b> A script (finetune_vs_pool.py) was written "
            "to play games directly against pool opponents in Python, then run PPO updates via "
            "the C backend. This directly addresses the architecture-mismatch blind spots.\n\n"
            "<b>Population-based training:</b> Maintain multiple agents that evolve together, "
            "providing natural curriculum and diverse opponents.\n\n"
            "<b>MCTS-guided policy improvement:</b> Use Monte Carlo tree search to generate "
            "stronger training targets, similar to AlphaZero.\n\n"
            "<b>Architecture matching:</b> Train with the same plain MLP architecture as the "
            "reference pool to eliminate the architectural gap."
        ),
    },
    {
        "heading": "Conclusion",
        "text": (
            "Over 18 experiments and ~40 runs, two key techniques drove the majority of "
            "improvement: increasing games per iteration to 2048 (+180 Elo) and applying "
            "opponent temperature of 1.5 (+80 Elo). Together these pushed peak Elo from "
            "1722 to 1970, a +248 improvement.\n\n"
            "The remaining gap to 2100 (~130 Elo) appears to be limited by the architecture "
            "mismatch between agent and reference pool, inherent self-play oscillation, and "
            "the narrow evaluation pool. Fine-tuning against pool opponents is the most "
            "promising untested approach."
        ),
    },
]

generate_pdf_report(
    os.path.join(REPORT_DIR, "c4_optimization_report.pdf"),
    "Connect 4 Self-Play PPO: Hyperparameter Optimization Report",
    sections,
    REPORT_DIR,
)
print("Done!")
