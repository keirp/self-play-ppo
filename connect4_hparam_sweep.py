"""Connect 4 full hparam sweep with Elo tracking.

Sweeps ent_coef, draw_reward, clip_eps sequentially (each phase uses best from prior).
Uses parallel subprocess execution and Elo as primary metric.
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

REPORT_DIR = "reports/connect4_hparam_sweep"
os.makedirs(REPORT_DIR, exist_ok=True)

# Base config with optimal LR from LR sweep
BASE_CONFIG = {
    "hidden_size": 256,
    "num_layers": 6,
    "lr": 1e-4,          # optimal from LR sweep
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

SWEEP_ITERS = 500
EVAL_INTERVAL = 25
MAX_PARALLEL = 4

FULL_TRAIN_ITERS = 3000
FULL_TRAIN_EVAL = 50


def _make_script(config, out_path, num_iters, eval_interval):
    return f"""
import json, time, random, numpy as np, torch
from src.connect4_c import Connect4TrainerC

config = {json.dumps(config)}
torch.manual_seed(42); np.random.seed(42); random.seed(42)

trainer = Connect4TrainerC(config)
t0 = time.perf_counter()
metrics = trainer.train({num_iters}, eval_interval={eval_interval}, verbose=True)
wall_time = time.perf_counter() - t0

out = {{'metrics': {{}}, 'wall_time': wall_time, 'total_params': trainer.total_params}}
for k, v in metrics.items():
    if isinstance(v, list) and len(v) > 0:
        out['metrics'][k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
    else:
        out['metrics'][k] = v

with open('{out_path}', 'w') as f:
    json.dump(out, f)
print(f'Done. Wall time: {{wall_time:.1f}}s')
"""


def run_sweep_phase(base_config, param_name, values, phase_name):
    """Run a single-parameter sweep in parallel. Returns list of (value, data) tuples."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    results = [None] * len(values)

    for batch_start in range(0, len(values), MAX_PARALLEL):
        batch_end = min(batch_start + MAX_PARALLEL, len(values))
        batch_indices = list(range(batch_start, batch_end))

        print(f"\n  Launching: {param_name} = {[values[i] for i in batch_indices]}")

        procs = []
        for idx in batch_indices:
            val = values[idx]
            config = base_config.copy()
            config[param_name] = val
            out_path = os.path.join(REPORT_DIR, f"{phase_name}_{idx}.json")
            script = _make_script(config, out_path, SWEEP_ITERS, EVAL_INTERVAL)
            p = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
            )
            procs.append((idx, val, p, out_path))

        for idx, val, p, out_path in procs:
            stdout, stderr = p.communicate(timeout=600)
            lines = stdout.decode().strip().split('\n')
            for line in lines[-2:]:
                print(f"    [{param_name}={val}] {line}")
            if p.returncode != 0:
                print(f"    [{param_name}={val}] FAILED")
                continue
            with open(out_path) as f:
                results[idx] = json.load(f)

    return [(values[i], results[i]) for i in range(len(values)) if results[i] is not None]


def pick_best(results, param_name):
    """Pick best value by final Elo."""
    best_val, best_elo = None, -9999
    for val, data in results:
        elos = data["metrics"].get("elo", [])
        final_elo = elos[-1] if elos else 0
        if final_elo > best_elo:
            best_elo = final_elo
            best_val = val
    print(f"\n  Best {param_name}: {best_val} (Elo {best_elo:.0f})")
    return best_val, best_elo


def run_full_training(config):
    """Run full training with best config."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(REPORT_DIR, "full_train.json")
    snap_dir = os.path.join(REPORT_DIR, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    # Full training script with snapshot saving
    script = f"""
import json, time, random, numpy as np, torch, ctypes, os
from collections import defaultdict
from src.connect4_c import (
    _lib, _fp, _llp, _ip, INPUT_DIM, POLICY_DIM,
    Connect4Net, extract_params, load_params
)
from src.connect4 import play_vs_random, play_vs_heuristic
from src.elo import compute_elo

torch.manual_seed(42); np.random.seed(42); random.seed(42)

config = {json.dumps(config)}
H = config["hidden_size"]
L = config["num_layers"]

_lib.c4_init(H, L)
total_params = _lib.c4_total_params()

model = Connect4Net(hidden_size=H, num_layers=L)
params = extract_params(model)
assert len(params) == total_params

adam_m = np.zeros(total_params, dtype=np.float32)
adam_v = np.zeros(total_params, dtype=np.float32)
adam_t = np.array([0], dtype=np.int32)

opponent_pool = [params.copy()]
pool_max = config["opponent_pool_max"]
snap_interval = config["snapshot_interval"]

max_trans = config["games_per_iter"] * 22
buf_obs = np.zeros((max_trans, INPUT_DIM), dtype=np.float32)
buf_actions = np.zeros(max_trans, dtype=np.int64)
buf_log_probs = np.zeros(max_trans, dtype=np.float32)
buf_values = np.zeros(max_trans, dtype=np.float32)
buf_valid_masks = np.zeros((max_trans, POLICY_DIM), dtype=np.float32)
buf_rewards = np.zeros(max_trans, dtype=np.float32)
buf_dones = np.zeros(max_trans, dtype=np.float32)
game_results = np.zeros(3, dtype=np.int32)
stats_out = np.zeros(4, dtype=np.float32)

metrics = defaultdict(list)
snap_dir = "{snap_dir}"
TOTAL = {FULL_TRAIN_ITERS}
EVAL_INT = {FULL_TRAIN_EVAL}
gpi = config["games_per_iter"]

np.save(os.path.join(snap_dir, "snapshot_0.npy"), params.copy())

print(f"Full training: {{TOTAL}} iters, {{gpi}} games/iter, H={{H}}, L={{L}}, params={{total_params}}")
print(f"Config: lr={{config['lr']}}, ent={{config['ent_coef']}}, clip={{config['clip_eps']}}, dr={{config['draw_reward']}}")
t0 = time.perf_counter()

for iteration in range(1, TOTAL + 1):
    iter_start = time.time()
    opp_idx = random.randint(0, len(opponent_pool) - 1)
    opp_params = opponent_pool[opp_idx]
    _lib.c4_seed(ctypes.c_ulong(random.getrandbits(64)))

    n_trans = _lib.c4_collect_games(
        _fp(params), _fp(opp_params), gpi, config["draw_reward"],
        _fp(buf_obs), _llp(buf_actions), _fp(buf_log_probs),
        _fp(buf_values), _fp(buf_valid_masks),
        _fp(buf_rewards), _fp(buf_dones), _ip(game_results),
    )
    _lib.c4_ppo_update(
        _fp(params), _fp(adam_m), _fp(adam_v), _ip(adam_t),
        _fp(buf_obs), _llp(buf_actions), _fp(buf_log_probs),
        _fp(buf_values), _fp(buf_valid_masks),
        _fp(buf_rewards), _fp(buf_dones), n_trans,
        config["gamma"], config["gae_lambda"], config["clip_eps"], config["vf_coef"],
        config["ent_coef"], config["lr"], config["max_grad_norm"], config["ppo_epochs"],
        config["batch_size"], _fp(stats_out),
    )

    gr = game_results
    metrics["iteration"].append(iteration)
    metrics["sp_win_rate"].append(int(gr[0]) / gpi)
    metrics["sp_draw_rate"].append(int(gr[1]) / gpi)
    metrics["sp_loss_rate"].append(int(gr[2]) / gpi)
    metrics["policy_loss"].append(float(stats_out[0]))
    metrics["value_loss"].append(float(stats_out[1]))
    metrics["entropy"].append(float(stats_out[2]))
    metrics["approx_kl"].append(float(stats_out[3]))
    metrics["n_transitions"].append(n_trans)

    if iteration % 100 == 0 or iteration == 1:
        print(f"[iter {{iteration:4d}}/{{TOTAL}}] SP W/D/L: {{gr[0]:3d}}/{{gr[1]:3d}}/{{gr[2]:3d}} | "
              f"ploss: {{stats_out[0]:.4f}} vloss: {{stats_out[1]:.4f}} "
              f"ent: {{stats_out[2]:.3f}} kl: {{stats_out[3]:.4f}} | "
              f"{{(time.time()-iter_start)*1000:.0f}}ms", flush=True)

    if iteration % snap_interval == 0:
        opponent_pool.append(params.copy())
        if len(opponent_pool) > pool_max:
            keep = [0] + list(range(len(opponent_pool) - pool_max + 1, len(opponent_pool)))
            opponent_pool = [opponent_pool[i] for i in keep]

    if iteration % 100 == 0:
        np.save(os.path.join(snap_dir, f"snapshot_{{iteration}}.npy"), params.copy())

    if iteration % EVAL_INT == 0:
        load_params(model, params)
        model.eval()
        policy_fn = model.get_policy_fn("cpu", deterministic=False)
        ng = 50
        w1,d1,l1 = play_vs_random(policy_fn, policy_plays_as=1, num_games=ng)
        w2,d2,l2 = play_vs_random(policy_fn, policy_plays_as=-1, num_games=ng)
        tot = ng*2
        metrics["vs_random_win_rate"].append((w1+w2)/tot)
        metrics["vs_random_draw_rate"].append((d1+d2)/tot)
        metrics["vs_random_loss_rate"].append((l1+l2)/tot)
        w1,d1,l1 = play_vs_heuristic(policy_fn, policy_plays_as=1, num_games=ng)
        w2,d2,l2 = play_vs_heuristic(policy_fn, policy_plays_as=-1, num_games=ng)
        metrics["vs_heuristic_win_rate"].append((w1+w2)/tot)
        metrics["vs_heuristic_draw_rate"].append((d1+d2)/tot)
        metrics["vs_heuristic_loss_rate"].append((l1+l2)/tot)

        elo_result = compute_elo(params, hidden_size=H, num_layers=L, games_per_opponent=20)
        metrics["elo"].append(elo_result["elo"])
        metrics["eval_iteration"].append(iteration)

        print(f"  EVAL @ {{iteration}}: vsR={{metrics['vs_random_win_rate'][-1]:.2f}} "
              f"vsH={{metrics['vs_heuristic_win_rate'][-1]:.2f}} "
              f"Elo={{metrics['elo'][-1]:.0f}}", flush=True)

wall_time = time.perf_counter() - t0
print(f"\\nDone in {{wall_time:.1f}}s")

out = {{"metrics": {{}}, "wall_time": wall_time, "total_params": total_params, "config": config}}
for k, v in metrics.items():
    if isinstance(v, list) and len(v) > 0:
        out["metrics"][k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
    else:
        out["metrics"][k] = v
with open("{out_path}", "w") as f:
    json.dump(out, f)
"""

    print(f"\nRunning full training ({FULL_TRAIN_ITERS} iters)...")
    subprocess.run([sys.executable, "-c", script], capture_output=False, timeout=3600, cwd=cwd)

    with open(out_path) as f:
        return json.load(f)


# ==================== PLOTTING ====================

def plot_sweep_phase(results, param_name, phase_name, save_dir):
    """Plot Elo curves for a single-parameter sweep."""
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, (val, data) in enumerate(results):
        m = data["metrics"]
        if "elo" in m:
            ax1.plot(m["eval_iteration"], m["elo"], "-o", color=colors[i],
                     markersize=3, linewidth=2, label=f"{param_name}={val}")
        ax2.plot(m["iteration"], smooth(np.array(m["entropy"]), 20), "-",
                 color=colors[i], linewidth=1.5, label=f"{param_name}={val}")

    ax1.axhline(y=1584, color="red", linestyle="--", alpha=0.5, label="Heuristic")
    ax1.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Elo")
    ax1.set_title(f"Elo by {param_name}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Entropy")
    ax2.set_title(f"Entropy by {param_name}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Phase: {phase_name}", fontweight="bold")
    fig.tight_layout()
    fname = f"{phase_name}.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)
    return fname


def plot_full_training(data, save_dir):
    """Plot full training curves."""
    m = data["metrics"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ei = m["eval_iteration"]
    axes[0, 0].plot(ei, m["elo"], "b-o", markersize=3, linewidth=2)
    axes[0, 0].axhline(y=1584, color="red", linestyle="--", alpha=0.5, label="Heuristic")
    axes[0, 0].axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Random")
    axes[0, 0].set_title("Elo Rating")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ei, m["vs_heuristic_win_rate"], "g-o", markersize=2, label="Win")
    axes[0, 1].plot(ei, m["vs_heuristic_loss_rate"], "r-^", markersize=2, label="Loss")
    axes[0, 1].set_title("vs Heuristic")
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(ei, m["vs_random_win_rate"], "g-o", markersize=2, label="Win")
    axes[0, 2].set_title("vs Random")
    axes[0, 2].legend()
    axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].grid(True, alpha=0.3)

    ti = m["iteration"]
    axes[1, 0].plot(ti, smooth(np.array(m["entropy"]), 50))
    axes[1, 0].set_title("Entropy")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ti, smooth(np.array(m["policy_loss"]), 50))
    axes[1, 1].set_title("Policy Loss")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(ti, smooth(np.array(m["value_loss"]), 50))
    axes[1, 2].set_title("Value Loss")
    axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Iteration")

    wt = data["wall_time"]
    fig.suptitle(f"Full Training ({FULL_TRAIN_ITERS} iters, {wt:.0f}s)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "full_training.png"), dpi=150)
    plt.close(fig)


def plot_final_bar(sweep_results, save_dir):
    """Bar chart comparing final Elo of each sweep phase winner vs baselines."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = []
    elos = []
    for label, elo in sweep_results:
        names.append(label)
        elos.append(elo)
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, elos, color=colors, width=0.5)
    ax.bar_label(bars, [f"{e:.0f}" for e in elos], fontsize=10)
    ax.axhline(y=1584, color="red", linestyle="--", alpha=0.5, label="Heuristic (1584)")
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Random (1000)")
    ax.set_ylabel("Final Elo")
    ax.set_title("Elo Progression Through Sweep Phases", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sweep_progression.png"), dpi=150)
    plt.close(fig)


# ==================== MAIN ====================

def main():
    all_data = {}
    config = BASE_CONFIG.copy()
    sweep_progression = []

    t_total = time.time()

    # Phase 1: ent_coef sweep
    print("\n" + "="*60)
    print("PHASE 1: Entropy Coefficient Sweep")
    print("="*60)
    ent_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    ent_results = run_sweep_phase(config, "ent_coef", ent_values, "ent_coef")
    best_ent, best_ent_elo = pick_best(ent_results, "ent_coef")
    config["ent_coef"] = best_ent
    all_data["ent_coef"] = {"values": ent_values, "best": best_ent,
                            "results": [(v, d["metrics"].get("elo", [])[-1] if d["metrics"].get("elo") else 0)
                                        for v, d in ent_results]}
    sweep_progression.append((f"LR only\n(1e-4)", 1388))
    sweep_progression.append((f"+ent_coef\n({best_ent})", best_ent_elo))
    plot_sweep_phase(ent_results, "ent_coef", "ent_coef_sweep", REPORT_DIR)

    # Phase 2: draw_reward sweep
    print("\n" + "="*60)
    print("PHASE 2: Draw Reward Sweep")
    print(f"  (using ent_coef={best_ent})")
    print("="*60)
    dr_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    dr_results = run_sweep_phase(config, "draw_reward", dr_values, "draw_reward")
    best_dr, best_dr_elo = pick_best(dr_results, "draw_reward")
    config["draw_reward"] = best_dr
    all_data["draw_reward"] = {"values": dr_values, "best": best_dr,
                                "results": [(v, d["metrics"].get("elo", [])[-1] if d["metrics"].get("elo") else 0)
                                            for v, d in dr_results]}
    sweep_progression.append((f"+draw_reward\n({best_dr})", best_dr_elo))
    plot_sweep_phase(dr_results, "draw_reward", "draw_reward_sweep", REPORT_DIR)

    # Phase 3: clip_eps sweep
    print("\n" + "="*60)
    print("PHASE 3: Clip Epsilon Sweep")
    print(f"  (using ent_coef={best_ent}, draw_reward={best_dr})")
    print("="*60)
    clip_values = [0.05, 0.1, 0.15, 0.2, 0.3]
    clip_results = run_sweep_phase(config, "clip_eps", clip_values, "clip_eps")
    best_clip, best_clip_elo = pick_best(clip_results, "clip_eps")
    config["clip_eps"] = best_clip
    all_data["clip_eps"] = {"values": clip_values, "best": best_clip,
                             "results": [(v, d["metrics"].get("elo", [])[-1] if d["metrics"].get("elo") else 0)
                                         for v, d in clip_results]}
    sweep_progression.append((f"+clip_eps\n({best_clip})", best_clip_elo))
    plot_sweep_phase(clip_results, "clip_eps", "clip_eps_sweep", REPORT_DIR)

    # Phase 4: batch_size / vf_coef quick sweep
    print("\n" + "="*60)
    print("PHASE 4: Batch Size Sweep")
    print(f"  (using ent_coef={best_ent}, draw_reward={best_dr}, clip_eps={best_clip})")
    print("="*60)
    bs_values = [64, 128, 256, 512]
    bs_results = run_sweep_phase(config, "batch_size", bs_values, "batch_size")
    best_bs, best_bs_elo = pick_best(bs_results, "batch_size")
    config["batch_size"] = best_bs
    all_data["batch_size"] = {"values": bs_values, "best": best_bs,
                               "results": [(v, d["metrics"].get("elo", [])[-1] if d["metrics"].get("elo") else 0)
                                           for v, d in bs_results]}
    sweep_progression.append((f"+batch_size\n({best_bs})", best_bs_elo))
    plot_sweep_phase(bs_results, "batch_size", "batch_size_sweep", REPORT_DIR)

    sweep_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"ALL SWEEPS DONE in {sweep_time:.0f}s")
    print(f"Best config: {json.dumps({k: config[k] for k in ['lr','ent_coef','draw_reward','clip_eps','batch_size']}, indent=2)}")
    print(f"{'='*60}")

    # Phase 5: Full training with best config
    full_data = run_full_training(config)
    plot_full_training(full_data, REPORT_DIR)

    final_elo = full_data["metrics"]["elo"][-1]
    sweep_progression.append((f"Full train\n({FULL_TRAIN_ITERS}it)", final_elo))
    plot_final_bar(sweep_progression, REPORT_DIR)

    # Save all results
    all_data["best_config"] = config
    all_data["sweep_time"] = sweep_time
    all_data["full_train_wall_time"] = full_data["wall_time"]
    all_data["sweep_progression"] = sweep_progression
    with open(os.path.join(REPORT_DIR, "all_results.json"), "w") as f:
        json.dump(all_data, f, indent=2, default=str)

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time:.0f}s")

    # Generate report AFTER seeing results
    generate_report(config, all_data, ent_results, dr_results, clip_results,
                    bs_results, full_data, sweep_progression, total_time)


def generate_report(config, all_data, ent_results, dr_results, clip_results,
                    bs_results, full_data, sweep_progression, total_time):
    """Generate PDF report from results."""
    fm = full_data["metrics"]
    final_elo = fm["elo"][-1]
    peak_elo = max(fm["elo"])
    peak_iter = fm["eval_iteration"][fm["elo"].index(peak_elo)]
    final_vr = fm["vs_random_win_rate"][-1]
    final_vh = fm["vs_heuristic_win_rate"][-1]

    def phase_table(results, param_name):
        rows = []
        for val, data in results:
            m = data["metrics"]
            elo = m.get("elo", [0])[-1]
            ent = m["entropy"][-1]
            rows.append([str(val), f"{elo:.0f}", f"{ent:.3f}"])
        return {"headers": [param_name, "Final Elo", "Final Entropy"], "rows": rows}

    sections = [
        {
            "heading": "Executive Summary",
            "text": (
                "This report presents a systematic hyperparameter sweep for Connect 4 self-play PPO, "
                "using Elo rating against a fixed reference pool as the primary metric. Starting from "
                "the optimal learning rate (1e-4), we sequentially swept entropy coefficient, draw reward, "
                "clip epsilon, and batch size. Each phase uses the best value from all prior phases."
                "\n\n"
                f"<b>Total sweep time:</b> {total_time:.0f}s (parallel execution, 4 workers)<br/>"
                f"<b>Phases:</b> LR (prior) → ent_coef → draw_reward → clip_eps → batch_size → full training<br/>"
                f"<b>Final config:</b> lr={config['lr']}, ent_coef={config['ent_coef']}, "
                f"draw_reward={config['draw_reward']}, clip_eps={config['clip_eps']}, "
                f"batch_size={config['batch_size']}<br/>"
                f"<b>Full training result:</b> Elo <b>{final_elo:.0f}</b> "
                f"(peak {peak_elo:.0f} at iter {peak_iter}), "
                f"vs Random {final_vr:.0%}, vs Heuristic {final_vh:.0%}"
            ),
            "page_break": True,
        },
        {
            "heading": "Sweep Progression",
            "text": (
                "Each bar shows the final Elo (at 500 iterations for sweep phases, "
                f"{FULL_TRAIN_ITERS} for full training) as we accumulate the best hyperparameter from each phase."
            ),
            "plots": [("sweep_progression.png", "Elo improvement through sweep phases.")],
            "page_break": True,
        },
        {
            "heading": "Phase 1: Entropy Coefficient",
            "text": (
                f"Swept ent_coef over {[v for v,_ in ent_results]}. "
                f"Best: <b>{config['ent_coef']}</b>."
            ),
            "table": phase_table(ent_results, "ent_coef"),
            "plots": [("ent_coef_sweep.png", "Elo and entropy curves by ent_coef.")],
            "page_break": True,
        },
        {
            "heading": "Phase 2: Draw Reward",
            "text": (
                f"Swept draw_reward over {[v for v,_ in dr_results]} "
                f"with ent_coef={config['ent_coef']}. "
                f"Best: <b>{config['draw_reward']}</b>."
            ),
            "table": phase_table(dr_results, "draw_reward"),
            "plots": [("draw_reward_sweep.png", "Elo and entropy curves by draw_reward.")],
            "page_break": True,
        },
        {
            "heading": "Phase 3: Clip Epsilon",
            "text": (
                f"Swept clip_eps over {[v for v,_ in clip_results]} "
                f"with ent_coef={config['ent_coef']}, draw_reward={config['draw_reward']}. "
                f"Best: <b>{config['clip_eps']}</b>."
            ),
            "table": phase_table(clip_results, "clip_eps"),
            "plots": [("clip_eps_sweep.png", "Elo and entropy curves by clip_eps.")],
            "page_break": True,
        },
        {
            "heading": "Phase 4: Batch Size",
            "text": (
                f"Swept batch_size over {[v for v,_ in bs_results]} "
                f"with prior best hparams. "
                f"Best: <b>{config['batch_size']}</b>."
            ),
            "table": phase_table(bs_results, "batch_size"),
            "plots": [("batch_size_sweep.png", "Elo and entropy curves by batch_size.")],
            "page_break": True,
        },
        {
            "heading": f"Full Training ({FULL_TRAIN_ITERS} iterations)",
            "text": (
                f"Full training with optimal config: lr={config['lr']}, ent_coef={config['ent_coef']}, "
                f"draw_reward={config['draw_reward']}, clip_eps={config['clip_eps']}, "
                f"batch_size={config['batch_size']}."
                "\n\n"
                f"<b>Final Elo:</b> {final_elo:.0f} (peak: {peak_elo:.0f} at iter {peak_iter})<br/>"
                f"<b>vs Random:</b> {final_vr:.0%}<br/>"
                f"<b>vs Heuristic:</b> {final_vh:.0%}<br/>"
                f"<b>Wall time:</b> {full_data['wall_time']:.0f}s"
            ),
            "plots": [("full_training.png", "Full training curves.")],
            "page_break": True,
        },
        {
            "heading": "Final Configuration",
            "table": {
                "headers": ["Parameter", "Value"],
                "rows": [[k, str(v)] for k, v in sorted(config.items())],
            },
        },
    ]

    report_path = os.path.join(REPORT_DIR, "hparam_sweep_report.pdf")
    generate_pdf_report(
        report_path=report_path,
        title="Connect 4 Self-Play PPO: Hyperparameter Sweep",
        sections=sections,
        plot_dir=REPORT_DIR,
    )
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
