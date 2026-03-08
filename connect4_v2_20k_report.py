"""Generate PDF report for the 20k-iteration v2 architecture training run."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

RESULTS_PATH = "reports/connect4_v2_20k/results.json"
REPORT_DIR = "reports/connect4_v2_20k"
REPORT_PATH = os.path.join(REPORT_DIR, "v2_20k_report.pdf")

# Also load 3k results for comparison
RESULTS_3K_PATH = "reports/connect4_v2_3k/results.json"

with open(RESULTS_PATH) as f:
    data = json.load(f)

m = data["metrics"]
wall_time = data["wall_time"]
total_params = data["total_params"]
config = data.get("config", {})

iters = np.array(m["iteration"])
eval_iters = np.array(m["eval_iteration"])
elos = np.array(m["elo"])

# Load 3k for comparison
has_3k = os.path.exists(RESULTS_3K_PATH)
if has_3k:
    with open(RESULTS_3K_PATH) as f:
        data_3k = json.load(f)
    m_3k = data_3k["metrics"]
    eval_iters_3k = np.array(m_3k["eval_iteration"])
    elos_3k = np.array(m_3k["elo"])

def smooth(x, w=100):
    x = np.array(x)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

with PdfPages(REPORT_PATH) as pdf:
    # --- Page 1: Title + Summary ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    final_elo = elos[-1]
    peak_elo = np.max(elos)
    peak_elo_iter = eval_iters[np.argmax(elos)]
    final_vs_random = m["vs_random_win_rate"][-1]
    final_vs_heuristic_w = m["vs_heuristic_win_rate"][-1]
    final_vs_heuristic_d = m["vs_heuristic_draw_rate"][-1]
    final_vs_heuristic_l = m["vs_heuristic_loss_rate"][-1]

    title = "Connect 4 Self-Play PPO — v2 Architecture (20,000 Iterations)"
    ax.text(0.5, 0.92, title, transform=ax.transAxes, fontsize=16,
            fontweight="bold", ha="center", va="top")

    summary = (
        f"Architecture: Residual MLP with LayerNorm + GELU (nanoGPT-style)\n"
        f"  Input -> Linear(126, 256) -> 5x [x + GELU(Linear(LN(x)))] -> LN -> Heads\n"
        f"  Total parameters: {total_params:,}\n\n"
        f"Training:\n"
        f"  Iterations: 20,000  |  Games/iter: 512  |  Wall time: {wall_time:.0f}s ({wall_time/60:.1f} min)\n"
        f"  LR: {config.get('lr', 3e-4)}  |  Entropy coef: {config.get('ent_coef', 0.001)}\n"
        f"  Batch size: {config.get('batch_size', 256)}  |  PPO epochs: 4\n"
        f"  Opponent pool: 50 (uniform sampling)\n\n"
        f"Final Results (iter 20,000):\n"
        f"  Elo: {final_elo:.0f}  (peak: {peak_elo:.0f} @ iter {peak_elo_iter})\n"
        f"  vs Random:    {final_vs_random:.0%} win\n"
        f"  vs Heuristic: {final_vs_heuristic_w:.0%} win, "
        f"{final_vs_heuristic_d:.0%} draw, {final_vs_heuristic_l:.0%} loss\n\n"
        f"C Backend Optimization:\n"
        f"  Vectorized GELU (vvtanhf), LayerNorm (vDSP), residual adds, bias broadcast\n"
        f"  3.8x speedup vs unoptimized (606ms -> 161ms/iter)"
    )
    ax.text(0.05, 0.82, summary, transform=ax.transAxes, fontsize=11,
            fontfamily="monospace", va="top", linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: Elo Curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eval_iters, elos, "b-", linewidth=1.2, alpha=0.8, label="20k run")
    if has_3k:
        ax.plot(eval_iters_3k, elos_3k, "g--", linewidth=1.2, alpha=0.6, label="3k run (previous)")
    ax.axhline(y=1584, color="red", linestyle="--", alpha=0.7, label="Heuristic baseline (~1584)")
    ax.axhline(y=1000, color="gray", linestyle=":", alpha=0.5, label="Random baseline (~1000)")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Elo Rating", fontsize=12)
    ax.set_title("Elo Rating vs Training Iteration", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20000)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 3: Elo Zoomed (last 10k) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    mask = eval_iters <= 3000
    ax.plot(eval_iters[mask], elos[mask], "b-o", markersize=2, linewidth=1)
    if has_3k:
        ax.plot(eval_iters_3k, elos_3k, "g--o", markersize=2, linewidth=1, alpha=0.6, label="3k run")
    ax.axhline(y=1584, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Early Training (0-3k)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Elo")
    ax.grid(True, alpha=0.3)
    if has_3k:
        ax.legend()

    ax = axes[1]
    mask = eval_iters >= 10000
    ax.plot(eval_iters[mask], elos[mask], "b-o", markersize=2, linewidth=1)
    ax.axhline(y=1584, color="red", linestyle="--", alpha=0.7, label="Heuristic")
    ax.set_title("Late Training (10k-20k)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Elo")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Elo Rating — Early vs Late Training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 4: Win Rates vs Fixed Opponents ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(eval_iters, m["vs_random_win_rate"], "g-", linewidth=1, label="Win")
    ax.plot(eval_iters, m["vs_random_draw_rate"], "orange", linewidth=1, label="Draw")
    ax.plot(eval_iters, m["vs_random_loss_rate"], "r-", linewidth=1, label="Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rate")
    ax.set_title("vs Random Opponent", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.plot(eval_iters, m["vs_heuristic_win_rate"], "g-", linewidth=1, label="Win")
    ax.plot(eval_iters, m["vs_heuristic_draw_rate"], "orange", linewidth=1, label="Draw")
    ax.plot(eval_iters, m["vs_heuristic_loss_rate"], "r-", linewidth=1, label="Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rate")
    ax.set_title("vs Heuristic Opponent", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Win/Draw/Loss Rates vs Fixed Opponents", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 5: Self-Play Metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    w = 200
    sp_w = smooth(m["sp_win_rate"], w)
    sp_d = smooth(m["sp_draw_rate"], w)
    sp_l = smooth(m["sp_loss_rate"], w)
    sm_iters = iters[len(iters)-len(sp_w):]

    ax = axes[0, 0]
    ax.plot(sm_iters, sp_w, "g-", alpha=0.8, label="Win")
    ax.plot(sm_iters, sp_d, "orange", alpha=0.8, label="Draw")
    ax.plot(sm_iters, sp_l, "r-", alpha=0.8, label="Loss")
    ax.set_title("Self-Play W/D/L (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, m["pool_size"], "b-")
    ax.set_title("Opponent Pool Size", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(sm_iters, smooth(m["n_transitions"], w), "purple")
    ax.set_title("Transitions per Iteration (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    ent_smooth = smooth(m["entropy"], w)
    ax = axes[1, 1]
    ax.plot(sm_iters[:len(ent_smooth)], ent_smooth, "teal")
    ax.set_title("Policy Entropy (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Self-Play Training Dynamics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 6: PPO Loss Curves ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(sm_iters[:len(smooth(m["policy_loss"], w))], smooth(m["policy_loss"], w), "b-")
    ax.set_title("Policy Loss (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(sm_iters[:len(smooth(m["value_loss"], w))], smooth(m["value_loss"], w), "r-")
    ax.set_title("Value Loss (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(sm_iters[:len(ent_smooth)], ent_smooth, "g-")
    ax.set_title("Entropy (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    kl_smooth = smooth(m["approx_kl"], w)
    ax = axes[1, 1]
    ax.plot(sm_iters[:len(kl_smooth)], kl_smooth, "purple")
    ax.set_title("Approx KL Divergence (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    fig.suptitle("PPO Training Losses", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 7: Elo milestone table ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    milestones = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]
    rows = []
    for mi in milestones:
        idx = np.argmin(np.abs(eval_iters - mi))
        ei = eval_iters[idx]
        if abs(ei - mi) <= 100:
            e_idx = list(m["eval_iteration"]).index(ei)
            rows.append([
                f"{ei:,}",
                f"{elos[idx]:.0f}",
                f"{m['vs_random_win_rate'][e_idx]:.0%}",
                f"{m['vs_heuristic_win_rate'][e_idx]:.0%}",
                f"{m['vs_heuristic_draw_rate'][e_idx]:.0%}",
                f"{m['vs_heuristic_loss_rate'][e_idx]:.0%}",
            ])

    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=["Iteration", "Elo", "vs Rand W%", "vs Heur W%", "vs Heur D%", "vs Heur L%"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Performance Milestones", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Report saved to {REPORT_PATH}")
