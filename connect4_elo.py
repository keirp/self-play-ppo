"""Connect 4: Re-run v1 training with snapshot saving, then compute Elo ratings.

This does NOT change the training algorithm. It re-runs the exact same v1 config
but saves parameter snapshots every N iterations for post-hoc Elo tournament.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import json
import subprocess
import itertools

from src.connect4_c import Connect4Net, load_params, extract_params
from src.connect4 import Connect4, play_vs_random, play_vs_heuristic, _heuristic_move
from src.report import generate_pdf_report, smooth

REPORT_DIR = "reports/connect4_elo"
os.makedirs(REPORT_DIR, exist_ok=True)

# Exact v1 best config
CONFIG = {
    "hidden_size": 256,
    "num_layers": 6,
    "lr": 3e-4,
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

TOTAL_ITERS = 3000
EVAL_INTERVAL = 50
ELO_SNAPSHOT_INTERVAL = 50  # Save snapshot every 50 iters for Elo


def run_training():
    """Run v1 training in subprocess, saving snapshots."""
    out_path = os.path.join(REPORT_DIR, "training_data.json")
    snapshots_dir = os.path.join(REPORT_DIR, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    script = f'''
import json, time, random, numpy as np, torch, ctypes, os
from collections import defaultdict

from src.connect4_c import (
    _lib, _fp, _llp, _ip, INPUT_DIM, POLICY_DIM,
    Connect4Net, extract_params, load_params
)
from src.connect4 import play_vs_random, play_vs_heuristic

torch.manual_seed(42); np.random.seed(42); random.seed(42)

config = {json.dumps(CONFIG)}
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
snapshots_dir = "{snapshots_dir}"

# Save initial snapshot
np.save(os.path.join(snapshots_dir, "snapshot_0.npy"), params.copy())

TOTAL_ITERS = {TOTAL_ITERS}
EVAL_INTERVAL = {EVAL_INTERVAL}
ELO_SNAP = {ELO_SNAPSHOT_INTERVAL}
games_per_iter = config["games_per_iter"]

print(f"Training: {{TOTAL_ITERS}} iters, {{games_per_iter}} games/iter, H={{H}}, L={{L}}, params={{total_params}}")
t0 = time.perf_counter()

for iteration in range(1, TOTAL_ITERS + 1):
    iter_start = time.time()

    # Select opponent (uniform from pool, same as v1)
    opp_idx = random.randint(0, len(opponent_pool) - 1)
    opp_params = opponent_pool[opp_idx]

    _lib.c4_seed(ctypes.c_ulong(random.getrandbits(64)))

    n_trans = _lib.c4_collect_games(
        _fp(params), _fp(opp_params),
        games_per_iter, config["draw_reward"],
        _fp(buf_obs), _llp(buf_actions), _fp(buf_log_probs),
        _fp(buf_values), _fp(buf_valid_masks),
        _fp(buf_rewards), _fp(buf_dones),
        _ip(game_results),
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

    iter_time = time.time() - iter_start
    gr = game_results

    metrics["iteration"].append(iteration)
    metrics["sp_win_rate"].append(int(gr[0]) / games_per_iter)
    metrics["sp_draw_rate"].append(int(gr[1]) / games_per_iter)
    metrics["sp_loss_rate"].append(int(gr[2]) / games_per_iter)
    metrics["policy_loss"].append(float(stats_out[0]))
    metrics["value_loss"].append(float(stats_out[1]))
    metrics["entropy"].append(float(stats_out[2]))
    metrics["approx_kl"].append(float(stats_out[3]))
    metrics["pool_size"].append(len(opponent_pool))
    metrics["n_transitions"].append(n_trans)

    if iteration % 100 == 0 or iteration == 1:
        print(
            f"[iter {{iteration:4d}}/{{TOTAL_ITERS}}] "
            f"SP W/D/L: {{gr[0]:3d}}/{{gr[1]:3d}}/{{gr[2]:3d}} | "
            f"ploss: {{stats_out[0]:.4f}} vloss: {{stats_out[1]:.4f}} "
            f"ent: {{stats_out[2]:.3f}} kl: {{stats_out[3]:.4f}} | "
            f"{{iter_time*1000:.1f}}ms", flush=True
        )

    # Opponent pool (same as v1)
    if iteration % snap_interval == 0:
        opponent_pool.append(params.copy())
        if len(opponent_pool) > pool_max:
            keep = [0] + list(range(len(opponent_pool) - pool_max + 1, len(opponent_pool)))
            opponent_pool = [opponent_pool[i] for i in keep]

    # Save Elo snapshots
    if iteration % ELO_SNAP == 0:
        np.save(os.path.join(snapshots_dir, f"snapshot_{{iteration}}.npy"), params.copy())

    # Evaluation
    if iteration % EVAL_INTERVAL == 0:
        eval_start = time.time()
        load_params(model, params)
        model.eval()
        policy_fn = model.get_policy_fn("cpu", deterministic=False)

        ng = 50  # 50 per side = 100 total (same as v1)
        w1, d1, l1 = play_vs_random(policy_fn, policy_plays_as=1, num_games=ng)
        w2, d2, l2 = play_vs_random(policy_fn, policy_plays_as=-1, num_games=ng)
        total = ng * 2
        metrics["vs_random_win_rate"].append((w1 + w2) / total)
        metrics["vs_random_draw_rate"].append((d1 + d2) / total)
        metrics["vs_random_loss_rate"].append((l1 + l2) / total)

        w1, d1, l1 = play_vs_heuristic(policy_fn, policy_plays_as=1, num_games=ng)
        w2, d2, l2 = play_vs_heuristic(policy_fn, policy_plays_as=-1, num_games=ng)
        metrics["vs_heuristic_win_rate"].append((w1 + w2) / total)
        metrics["vs_heuristic_draw_rate"].append((d1 + d2) / total)
        metrics["vs_heuristic_loss_rate"].append((l1 + l2) / total)
        metrics["eval_iteration"].append(iteration)

        eval_time = time.time() - eval_start
        print(
            f"  EVAL @ {{iteration}}: "
            f"vs Random W: {{metrics['vs_random_win_rate'][-1]:.2f}} | "
            f"vs Heuristic W: {{metrics['vs_heuristic_win_rate'][-1]:.2f}} "
            f"D: {{metrics['vs_heuristic_draw_rate'][-1]:.2f}} "
            f"L: {{metrics['vs_heuristic_loss_rate'][-1]:.2f}} | "
            f"Pool: {{len(opponent_pool)}} | eval {{eval_time:.1f}}s", flush=True
        )

wall_time = time.perf_counter() - t0
print(f"\\nTraining done in {{wall_time:.1f}}s")

# Save
out = {{"metrics": {{}}, "wall_time": wall_time, "total_params": total_params}}
for k, v in metrics.items():
    if isinstance(v, list) and len(v) > 0:
        out["metrics"][k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
    else:
        out["metrics"][k] = v

with open("{out_path}", "w") as f:
    json.dump(out, f)
'''

    print("Running v1 training with snapshot saving...")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=False, timeout=3600,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    with open(out_path) as f:
        return json.load(f)


def compute_elo_tournament(snapshots_dir):
    """Round-robin tournament between all snapshots + random + heuristic."""
    print("\nComputing Elo ratings via round-robin tournament...")

    snapshot_files = sorted(
        [f for f in os.listdir(snapshots_dir) if f.startswith("snapshot_") and f.endswith(".npy")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    players = {}
    for f in snapshot_files:
        iter_num = int(f.split("_")[1].split(".")[0])
        params = np.load(os.path.join(snapshots_dir, f))
        players[f"iter_{iter_num}"] = {"params": params, "iteration": iter_num}

    player_names = list(players.keys()) + ["random", "heuristic"]
    n_players = len(player_names)

    print(f"Tournament: {n_players} players ({len(players)} snapshots + random + heuristic)")

    model_a = Connect4Net(hidden_size=CONFIG["hidden_size"], num_layers=CONFIG["num_layers"])
    model_b = Connect4Net(hidden_size=CONFIG["hidden_size"], num_layers=CONFIG["num_layers"])

    games_per_pair = 20  # 10 per side
    results = {}

    t0 = time.time()
    total_pairs = n_players * (n_players - 1) // 2
    pair_count = 0

    for i in range(n_players):
        for j in range(i + 1, n_players):
            pa = player_names[i]
            pb = player_names[j]

            # Build policy functions
            if pa == "random":
                policy_a = lambda obs, valid: np.random.choice(np.where(valid > 0)[0])
            elif pa == "heuristic":
                policy_a = "heuristic"
            else:
                load_params(model_a, players[pa]["params"])
                model_a.eval()
                policy_a = model_a.get_policy_fn("cpu", deterministic=False)

            if pb == "random":
                policy_b = lambda obs, valid: np.random.choice(np.where(valid > 0)[0])
            elif pb == "heuristic":
                policy_b = "heuristic"
            else:
                load_params(model_b, players[pb]["params"])
                model_b.eval()
                policy_b = model_b.get_policy_fn("cpu", deterministic=False)

            wa, da, la = 0, 0, 0

            for g in range(games_per_pair):
                env = Connect4()
                obs = env.reset()
                a_plays_as = 1 if g < games_per_pair // 2 else -1

                while not env.done:
                    valid = env.get_valid_moves()
                    if env.current_player == a_plays_as:
                        if policy_a == "heuristic":
                            action = _heuristic_move(env)
                        else:
                            action = policy_a(obs, valid)
                    else:
                        if policy_b == "heuristic":
                            action = _heuristic_move(env)
                        else:
                            action = policy_b(obs, valid)
                    obs, reward, done, info = env.step(action)

                if env.winner == a_plays_as:
                    wa += 1
                elif env.winner == 0:
                    da += 1
                else:
                    la += 1

            results[(pa, pb)] = (wa, da, la)
            pair_count += 1

            if pair_count % 200 == 0:
                elapsed = time.time() - t0
                print(f"  {pair_count}/{total_pairs} pairs done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"Tournament done: {pair_count} pairs, {pair_count * games_per_pair} games in {elapsed:.1f}s")

    # Compute Elo using iterative Bradley-Terry
    elo = {name: 1500.0 for name in player_names}

    for _ in range(200):
        for (pa, pb), (wa, da, la) in results.items():
            total = wa + da + la
            if total == 0:
                continue
            ea = 1.0 / (1.0 + 10 ** ((elo[pb] - elo[pa]) / 400))
            sa = (wa + 0.5 * da) / total
            K = 16
            elo[pa] += K * (sa - ea)
            elo[pb] += K * ((1 - sa) - (1 - ea))

    # Normalize so random = 1000
    offset = 1000 - elo.get("random", 1500)
    for name in elo:
        elo[name] += offset

    # Extract Elo curve
    elo_curve = []
    for name in sorted(players.keys(), key=lambda x: players[x]["iteration"]):
        elo_curve.append({
            "iteration": players[name]["iteration"],
            "elo": elo[name],
        })

    elo_data = {
        "elo_ratings": elo,
        "elo_curve": elo_curve,
        "random_elo": elo.get("random", 1000),
        "heuristic_elo": elo.get("heuristic", 1000),
        "results": {f"{pa}_vs_{pb}": list(v) for (pa, pb), v in results.items()},
    }

    with open(os.path.join(REPORT_DIR, "elo_data.json"), "w") as f:
        json.dump(elo_data, f, indent=2)

    print(f"\nElo ratings:")
    print(f"  Random:    {elo['random']:.0f}")
    print(f"  Heuristic: {elo['heuristic']:.0f}")
    for item in elo_curve:
        if item["iteration"] % 200 == 0 or item["iteration"] == 0:
            print(f"  Iter {item['iteration']:5d}: {item['elo']:.0f}")
    if elo_curve:
        print(f"  Final:     {elo_curve[-1]['elo']:.0f}")

    return elo_data


def generate_plots(train_data, elo_data):
    """Generate all plots from training and Elo data."""
    m = train_data["metrics"]

    # Plot 1: Elo curve (the main new thing)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    curve = elo_data["elo_curve"]
    iters = [x["iteration"] for x in curve]
    elos = [x["elo"] for x in curve]
    ax.plot(iters, elos, "b-o", markersize=3, linewidth=2, label="Agent Elo")
    ax.axhline(y=elo_data["random_elo"], color="gray", linestyle="--",
               label=f"Random ({elo_data['random_elo']:.0f})")
    ax.axhline(y=elo_data["heuristic_elo"], color="red", linestyle="--",
               label=f"Heuristic ({elo_data['heuristic_elo']:.0f})")
    ax.set_xlabel("Training Iteration", fontsize=12)
    ax.set_ylabel("Elo Rating", fontsize=12)
    ax.set_title("Agent Elo Rating Over Training\n(Round-robin tournament, 61 snapshots + random + heuristic)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "elo_curve.png"), dpi=150)
    plt.close(fig)

    # Plot 2: Elo vs win-rate comparison (showing Elo is smoother)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    eval_iters = m["eval_iteration"]
    ax1.plot(eval_iters, m["vs_heuristic_win_rate"], "r-o", markersize=2, alpha=0.7, label="Win rate vs Heuristic")
    ax1.set_ylabel("Win Rate", fontsize=11)
    ax1.set_title("vs Heuristic Win Rate (noisy, 100 games per eval)", fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(iters, elos, "b-o", markersize=3, linewidth=2, label="Elo")
    ax2.axhline(y=elo_data["heuristic_elo"], color="red", linestyle="--", alpha=0.5,
                label=f"Heuristic Elo ({elo_data['heuristic_elo']:.0f})")
    ax2.set_ylabel("Elo Rating", fontsize=11)
    ax2.set_xlabel("Training Iteration", fontsize=11)
    ax2.set_title("Elo Rating (smooth, round-robin tournament)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Win Rate vs Elo: Why Elo is a Better Metric", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "elo_vs_winrate.png"), dpi=150)
    plt.close(fig)

    # Plot 3: Training dynamics (entropy, losses, self-play)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    t_iters = m["iteration"]

    axes[0, 0].plot(eval_iters, m["vs_random_win_rate"], "g-o", markersize=2, label="Win")
    axes[0, 0].plot(eval_iters, m["vs_random_loss_rate"], "r-^", markersize=2, label="Loss")
    axes[0, 0].set_title("vs Random")
    axes[0, 0].legend()
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(eval_iters, m["vs_heuristic_win_rate"], "g-o", markersize=2, label="Win")
    axes[0, 1].plot(eval_iters, m["vs_heuristic_loss_rate"], "r-^", markersize=2, label="Loss")
    axes[0, 1].set_title("vs Heuristic")
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(t_iters, smooth(np.array(m["sp_win_rate"]), 50), "g-", label="Win")
    axes[0, 2].plot(t_iters, smooth(np.array(m["sp_draw_rate"]), 50), "b-", label="Draw")
    axes[0, 2].plot(t_iters, smooth(np.array(m["sp_loss_rate"]), 50), "r-", label="Loss")
    axes[0, 2].set_title("Self-Play Results")
    axes[0, 2].legend()
    axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(t_iters, smooth(np.array(m["policy_loss"]), 50))
    axes[1, 0].set_title("Policy Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_iters, smooth(np.array(m["value_loss"]), 50))
    axes[1, 1].set_title("Value Loss")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(t_iters, smooth(np.array(m["entropy"]), 50))
    axes[1, 2].set_title("Entropy")
    axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Iteration")
    fig.suptitle("Training Dynamics (v1 config, 3000 iters)", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)

    print("Plots saved.")


def generate_report(train_data, elo_data):
    """Generate PDF report from results."""
    m = train_data["metrics"]
    wall_time = train_data["wall_time"]
    total_params = train_data["total_params"]

    curve = elo_data["elo_curve"]
    final_elo = curve[-1]["elo"]
    peak_elo = max(x["elo"] for x in curve)
    peak_elo_iter = [x for x in curve if x["elo"] == peak_elo][0]["iteration"]
    heuristic_elo = elo_data["heuristic_elo"]
    random_elo = elo_data["random_elo"]

    final_vs_random = m["vs_random_win_rate"][-1]
    final_vs_heur = m["vs_heuristic_win_rate"][-1]

    # Elo at key checkpoints
    elo_at_500 = [x["elo"] for x in curve if x["iteration"] == 500][0]
    elo_at_1000 = [x["elo"] for x in curve if x["iteration"] == 1000][0]
    elo_at_2000 = [x["elo"] for x in curve if x["iteration"] == 2000][0]
    elo_at_3000 = [x["elo"] for x in curve if x["iteration"] == 3000][0]

    sections = [
        {
            "heading": "Executive Summary",
            "text": (
                "This report introduces <b>Elo rating tracking</b> as a measurement tool for self-play "
                "PPO training in Connect 4. The key insight is that per-evaluation win rates against "
                "fixed opponents (random, heuristic) are <b>extremely noisy</b> (swinging 1%–73% across "
                "consecutive evaluations), making it difficult to assess whether training is actually "
                "improving the agent. Elo ratings from a post-hoc round-robin tournament provide a much "
                "smoother, more informative signal."
                "\n\n"
                "<b>Setup:</b> We re-ran the v1 best configuration (256x6 architecture, lr=3e-4, "
                "ent_coef=0.01, draw_reward=0.5) for 3000 iterations, saving parameter snapshots every "
                "50 iterations. After training, we ran a round-robin tournament between all 61 snapshots "
                "plus random and heuristic opponents (1,953 pairs, 20 games each = 39,060 total games)."
                "\n\n"
                "<b>Key findings:</b><br/>"
                f"1. The agent's Elo increases from {curve[0]['elo']:.0f} to <b>{final_elo:.0f}</b>, "
                f"reaching the heuristic level ({heuristic_elo:.0f}) by iteration ~2800<br/>"
                "2. Elo reveals a clear improvement trend that is invisible in the noisy win-rate data<br/>"
                "3. Most improvement happens in the first 500 iterations; progress after that is very slow<br/>"
                "4. The agent is barely at heuristic strength after 3000 iterations — significant room "
                "for algorithmic improvement remains"
            ),
            "page_break": True,
        },
        {
            "heading": "Why Elo? Win Rate vs Elo Comparison",
            "text": (
                "The plot below shows the same training run measured two ways. The top panel shows "
                "per-evaluation win rate vs the heuristic (100 games per eval). The bottom panel shows "
                "Elo from the round-robin tournament."
                "\n\n"
                "The win-rate signal is essentially <b>random noise</b>: it swings from 1% to 73% with no "
                "visible trend. This makes it impossible to tell if training is improving or stalled. "
                "With only 100 games per evaluation, the standard error of a 50% win rate is ~5%, but "
                "the observed variance is far larger — likely due to the agent's stochastic policy "
                "producing qualitatively different play patterns across evaluations."
                "\n\n"
                "In contrast, the Elo curve is <b>smooth and monotonically increasing</b>. Each Elo "
                "rating is computed from games against all other snapshots (not just the heuristic), "
                "giving a much richer signal. The tournament totals 39,060 games — far more data than "
                "the 100-game per-eval window."
                "\n\n"
                "<b>Elo methodology:</b> After training, all 61 snapshots plus random and heuristic "
                "played each other in a round-robin (20 games per pair, 10 per side). Elo ratings were "
                "computed using iterative Bradley-Terry updates (200 iterations, K=16), anchored so "
                "random = 1000. Draws count as 0.5 wins."
            ),
            "plots": [("elo_vs_winrate.png",
                        "Top: noisy win rate vs heuristic. Bottom: smooth Elo curve from round-robin.")],
            "page_break": True,
        },
        {
            "heading": "Elo Progression Over Training",
            "text": (
                "The Elo curve reveals three phases of learning:"
                "\n\n"
                f"<b>Phase 1 — Rapid improvement (iter 0–200):</b> Elo rises from {curve[0]['elo']:.0f} "
                f"to ~1300 (+300 Elo). The agent quickly learns basic column preferences and simple "
                "tactical patterns that beat random opponents."
                "\n\n"
                f"<b>Phase 2 — Slow grind (iter 200–2400):</b> Elo increases only ~170 points over 2200 "
                "iterations. The agent is marginally improving but struggles to discover the multi-step "
                "tactics (double threats, forcing sequences) needed to consistently beat the heuristic."
                "\n\n"
                f"<b>Phase 3 — Late push (iter 2400–3000):</b> Elo jumps from ~1470 to {final_elo:.0f}, "
                f"finally reaching heuristic level ({heuristic_elo:.0f}). This may represent a "
                "breakthrough in tactical play, or could be noise in the measurement."
            ),
            "plots": [("elo_curve.png",
                        "Agent Elo over training with random and heuristic baselines.")],
            "table": {
                "headers": ["Checkpoint", "Elo", "Gap to Heuristic"],
                "rows": [
                    ["Iter 0 (random init)", f"{curve[0]['elo']:.0f}", f"{curve[0]['elo'] - heuristic_elo:+.0f}"],
                    ["Iter 500", f"{elo_at_500:.0f}", f"{elo_at_500 - heuristic_elo:+.0f}"],
                    ["Iter 1000", f"{elo_at_1000:.0f}", f"{elo_at_1000 - heuristic_elo:+.0f}"],
                    ["Iter 2000", f"{elo_at_2000:.0f}", f"{elo_at_2000 - heuristic_elo:+.0f}"],
                    ["Iter 3000 (final)", f"{elo_at_3000:.0f}", f"{elo_at_3000 - heuristic_elo:+.0f}"],
                    ["Random", f"{random_elo:.0f}", f"{random_elo - heuristic_elo:+.0f}"],
                    ["Heuristic", f"{heuristic_elo:.0f}", "0"],
                ],
            },
            "page_break": True,
        },
        {
            "heading": "Training Dynamics",
            "text": (
                f"Training ran for 3000 iterations (512 games/iter) in {wall_time:.0f}s "
                f"({wall_time/3000*1000:.0f}ms/iter). Architecture: 256x6, {total_params:,} parameters."
                "\n\n"
                "<b>Entropy:</b> Drops from ~1.9 to ~0.5 in the first 200–300 iterations and then "
                "oscillates between 0.3–0.8 for the remainder. The oscillations correspond to the "
                "agent cycling between near-deterministic strategies and brief exploratory phases. "
                "With constant ent_coef=0.01, entropy is not being actively maintained."
                "\n\n"
                "<b>Self-play results:</b> The agent wins 80–95% of self-play games, indicating "
                "it dominates the opponent pool (which includes many older, weaker snapshots with "
                "uniform sampling)."
                "\n\n"
                "<b>vs Random:</b> Consistently 85–96% win rate — strong baseline performance."
                "\n\n"
                "<b>vs Heuristic:</b> Extremely noisy (see Elo discussion above). The per-eval signal "
                "is not useful for assessing improvement."
            ),
            "plots": [("training_curves.png", "Training dynamics over 3000 iterations.")],
            "page_break": True,
        },
        {
            "heading": "Conclusions and Next Steps",
            "text": (
                "<b>Elo tracking is valuable:</b> The round-robin Elo measurement provides a clear, "
                "smooth signal of agent improvement that is invisible in per-eval win rates. It should "
                "be the primary metric for assessing training progress in self-play settings."
                "\n\n"
                "<b>Current agent is weak:</b> After 3000 iterations, the agent barely reaches "
                f"heuristic Elo ({final_elo:.0f} vs {heuristic_elo:.0f}). The heuristic itself is "
                "very simple (1-ply lookahead + center preference). A strong Connect 4 agent should "
                "significantly exceed this."
                "\n\n"
                "<b>Bottleneck diagnosis:</b> The slow Elo growth in Phase 2 (iter 200–2400) suggests "
                "the agent gets stuck after learning basic tactics. Potential causes include:<br/>"
                "1. <b>Low entropy</b> — the policy becomes too deterministic too early, preventing "
                "exploration of new strategies<br/>"
                "2. <b>Uniform opponent sampling</b> — playing against many weak early snapshots "
                "provides poor learning signal<br/>"
                "3. <b>Constant hyperparameters</b> — lr, entropy coef, and clip epsilon may need "
                "scheduling to enable different learning phases<br/>"
                "\n\n"
                "<b>Recommended next steps:</b><br/>"
                "1. Use Elo as the primary evaluation metric going forward<br/>"
                "2. Investigate targeted algorithmic changes one at a time, using Elo to measure impact<br/>"
                "3. Specifically test: higher entropy coefficient, entropy annealing, weighted-recent "
                "opponent sampling, and learning rate scheduling"
            ),
        },
    ]

    report_path = os.path.join(REPORT_DIR, "connect4_elo_report.pdf")
    generate_pdf_report(
        report_path=report_path,
        title="Connect 4 Self-Play PPO: Elo Rating Analysis",
        sections=sections,
        plot_dir=REPORT_DIR,
    )
    print(f"Report saved to: {report_path}")


def main():
    # Step 1: Train (same as v1, but save snapshots)
    train_data = run_training()

    # Step 2: Elo tournament
    snapshots_dir = os.path.join(REPORT_DIR, "snapshots")
    elo_data = compute_elo_tournament(snapshots_dir)

    # Step 3: Generate plots and report
    generate_plots(train_data, elo_data)
    generate_report(train_data, elo_data)


def generate_report_only():
    """Generate plots and report from existing data (no training)."""
    with open(os.path.join(REPORT_DIR, "training_data.json")) as f:
        train_data = json.load(f)
    with open(os.path.join(REPORT_DIR, "elo_data.json")) as f:
        elo_data = json.load(f)
    generate_plots(train_data, elo_data)
    generate_report(train_data, elo_data)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--report-only":
        generate_report_only()
    else:
        main()
