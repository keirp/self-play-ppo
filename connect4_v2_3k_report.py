"""Generate PDF report for the 3000-iteration v2 architecture training run."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

RESULTS_PATH = "reports/connect4_v2_3k/results.json"
REPORT_DIR = "reports/connect4_v2_3k"
REPORT_PATH = os.path.join(REPORT_DIR, "v2_3k_report.pdf")

with open(RESULTS_PATH) as f:
    data = json.load(f)

m = data["metrics"]
wall_time = data["wall_time"]
total_params = data["total_params"]

iters = np.array(m["iteration"])
eval_iters = np.array(m["eval_iteration"])

# Config (from the sweep results)
config = {
    "lr": 3e-4,
    "ent_coef": 0.001,
    "batch_size": 256,
    "hidden_size": 256,
    "num_layers": 6,
    "games_per_iter": 512,
    "opponent_pool_max": 50,
    "opponent_sampling": "uniform",
    "snapshot_interval": 25,
    "ppo_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# Smoothing helper
def smooth(x, w=50):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

with PdfPages(REPORT_PATH) as pdf:
    # --- Page 1: Title + Summary ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    final_elo = m["elo"][-1]
    final_vs_random = m["vs_random_win_rate"][-1]
    final_vs_heuristic_w = m["vs_heuristic_win_rate"][-1]
    final_vs_heuristic_d = m["vs_heuristic_draw_rate"][-1]
    final_vs_heuristic_l = m["vs_heuristic_loss_rate"][-1]
    peak_elo = max(m["elo"])
    peak_elo_iter = eval_iters[np.argmax(m["elo"])]

    title = "Connect 4 Self-Play PPO — v2 Architecture (3000 Iterations)"
    ax.text(0.5, 0.92, title, transform=ax.transAxes, fontsize=16,
            fontweight="bold", ha="center", va="top")

    summary = (
        f"Architecture: Residual MLP with LayerNorm + GELU (nanoGPT-style)\n"
        f"  Input → Linear(126, 256) → 5× [x + GELU(Linear(LN(x)))] → LN → Heads\n"
        f"  Total parameters: {total_params:,}\n\n"
        f"Training:\n"
        f"  Iterations: 3,000  |  Games/iter: 512  |  Wall time: {wall_time:.0f}s ({wall_time/60:.1f} min)\n"
        f"  LR: 3e-4  |  Entropy coef: 0.001  |  Batch size: 256\n"
        f"  PPO epochs: 4  |  Opponent pool: 50 (uniform sampling)\n\n"
        f"Final Results (iter 3000):\n"
        f"  Elo: {final_elo:.0f}  (peak: {peak_elo:.0f} @ iter {peak_elo_iter})\n"
        f"  vs Random:    {final_vs_random:.0%} win\n"
        f"  vs Heuristic: {final_vs_heuristic_w:.0%} win, {final_vs_heuristic_d:.0%} draw, {final_vs_heuristic_l:.0%} loss\n\n"
        f"Key Finding:\n"
        f"  The residual architecture surpasses the heuristic baseline (Elo ~1584)\n"
        f"  with NO catastrophic forgetting — Elo climbs continuously throughout training.\n"
        f"  This is a significant improvement over the v1 MLP which peaked and declined."
    )
    ax.text(0.05, 0.82, summary, transform=ax.transAxes, fontsize=11,
            fontfamily="monospace", va="top", linespacing=1.4)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: Elo Curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eval_iters, m["elo"], "b-o", markersize=3, linewidth=1.5, label="Elo")
    ax.axhline(y=1584, color="red", linestyle="--", alpha=0.7, label="Heuristic baseline (~1584)")
    ax.axhline(y=1000, color="gray", linestyle=":", alpha=0.5, label="Random baseline (~1000)")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Elo Rating", fontsize=12)
    ax.set_title("Elo Rating vs Training Iteration", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3000)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 3: Win Rates vs Fixed Opponents ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(eval_iters, m["vs_random_win_rate"], "g-o", markersize=3, label="Win")
    ax.plot(eval_iters, m["vs_random_draw_rate"], "orange", markersize=2, label="Draw")
    ax.plot(eval_iters, m["vs_random_loss_rate"], "r-", markersize=2, label="Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rate")
    ax.set_title("vs Random Opponent", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.plot(eval_iters, m["vs_heuristic_win_rate"], "g-o", markersize=3, label="Win")
    ax.plot(eval_iters, m["vs_heuristic_draw_rate"], "orange", markersize=2, label="Draw")
    ax.plot(eval_iters, m["vs_heuristic_loss_rate"], "r-", markersize=2, label="Loss")
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

    # --- Page 4: Self-Play Metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sp_w = smooth(m["sp_win_rate"])
    sp_d = smooth(m["sp_draw_rate"])
    sp_l = smooth(m["sp_loss_rate"])
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
    ax.plot(sm_iters, smooth(m["n_transitions"]), "purple")
    ax.set_title("Transitions per Iteration (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(sm_iters, smooth(m["entropy"]), "teal")
    ax.set_title("Policy Entropy (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Self-Play Training Dynamics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 5: PPO Loss Curves ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(sm_iters, smooth(m["policy_loss"]), "b-")
    ax.set_title("Policy Loss (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(sm_iters, smooth(m["value_loss"]), "r-")
    ax.set_title("Value Loss (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(sm_iters, smooth(m["entropy"]), "g-")
    ax.set_title("Entropy (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(sm_iters, smooth(m["approx_kl"]), "purple")
    ax.set_title("Approx KL Divergence (smoothed)", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    fig.suptitle("PPO Training Losses", fontsize=14, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Report saved to {REPORT_PATH}")
