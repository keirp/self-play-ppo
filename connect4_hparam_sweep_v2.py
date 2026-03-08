"""Connect 4 hparam sweep v2 — new residual architecture.

Sequential 3-phase sweep: lr → ent_coef → batch_size
2 seeds each, 500 steps, parallel execution.
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

REPORT_DIR = "reports/connect4_hparam_sweep_v2"
os.makedirs(REPORT_DIR, exist_ok=True)

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
    "opponent_pool_max": 50,
    "snapshot_interval": 25,
    "opponent_sampling": "uniform",
}

PHASES = [
    {
        "name": "lr",
        "param": "lr",
        "values": [3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    },
    {
        "name": "ent_coef",
        "param": "ent_coef",
        "values": [0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
    },
    {
        "name": "batch_size",
        "param": "batch_size",
        "values": [64, 128, 256, 512, 1024],
    },
]

NUM_SEEDS = 2
SWEEP_ITERS = 500
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


def run_phase(phase_name, param_name, values, base_config):
    """Run one sweep phase. Returns {value: [result_dict, ...]}."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    jobs = []
    for vi, val in enumerate(values):
        for seed in range(NUM_SEEDS):
            jobs.append((vi, val, seed))

    results = {}
    for batch_start in range(0, len(jobs), MAX_PARALLEL):
        batch = jobs[batch_start:batch_start + MAX_PARALLEL]
        print(f"\n{'='*60}")
        print(f"[{phase_name}] Batch: {[(f'{param_name}={v}', f's{s}') for _, v, s in batch]}")
        print(f"{'='*60}")

        procs = []
        for vi, val, seed in batch:
            config = base_config.copy()
            config[param_name] = val
            tag = f"{phase_name}_{vi}_s{seed}"
            out_path = os.path.join(REPORT_DIR, f"{tag}.json")
            script = _make_script(config, out_path, SWEEP_ITERS, EVAL_INTERVAL, seed + 42)
            p = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=cwd,
            )
            procs.append((vi, val, seed, p, out_path))

        for vi, val, seed, p, out_path in procs:
            stdout, stderr = p.communicate(timeout=600)
            lines = stdout.decode().strip().split('\n')
            for line in lines[-2:]:
                print(f"  [{param_name}={val} s{seed}] {line}")
            if p.returncode != 0:
                print(f"  [{param_name}={val} s{seed}] FAILED: {stderr.decode()[-300:]}")
                continue
            with open(out_path) as f:
                r = json.load(f)
            key = str(val)
            if key not in results:
                results[key] = []
            results[key].append(r)

    return results


def pick_best(results, values):
    """Pick the value with highest mean final Elo."""
    best_val = values[0]
    best_elo = -9999
    for val in values:
        runs = results.get(str(val), [])
        elos = [r["metrics"]["elo"][-1] for r in runs if "elo" in r["metrics"]]
        if elos:
            mean_elo = np.mean(elos)
            if mean_elo > best_elo:
                best_elo = mean_elo
                best_val = val
    return best_val, best_elo


def main():
    t0 = time.time()
    all_phase_results = {}
    config = BASE_CONFIG.copy()

    for phase in PHASES:
        phase_name = phase["name"]
        param = phase["param"]
        values = phase["values"]

        print(f"\n\n{'#'*60}")
        print(f"# Phase: {phase_name} — sweeping {param} over {values}")
        print(f"# Base: {param}={config.get(param)}")
        print(f"{'#'*60}")

        results = run_phase(phase_name, param, values, config)
        all_phase_results[phase_name] = {
            "param": param,
            "values": values,
            "results": results,
        }

        best_val, best_elo = pick_best(results, values)
        print(f"\n>>> Best {param} = {best_val} (mean Elo {best_elo:.0f})")
        config[param] = best_val

    total_time = time.time() - t0
    print(f"\nTotal sweep time: {total_time:.0f}s")
    print(f"Final config: lr={config['lr']}, ent_coef={config['ent_coef']}, batch_size={config['batch_size']}")

    # Save
    save_data = {
        "phases": all_phase_results,
        "final_config": config,
        "total_time": total_time,
        "num_seeds": NUM_SEEDS,
        "sweep_iters": SWEEP_ITERS,
    }
    # Convert values keys for JSON
    for pname, pdata in save_data["phases"].items():
        pdata["values"] = [float(v) if isinstance(v, float) else int(v) for v in pdata["values"]]
    with open(os.path.join(REPORT_DIR, "all_results.json"), "w") as f:
        json.dump(save_data, f)

    print("Results saved. Run with --report to generate PDF.")


def generate_report_from_results():
    with open(os.path.join(REPORT_DIR, "all_results.json")) as f:
        data = json.load(f)

    phases = data["phases"]
    final_config = data["final_config"]
    total_time = data["total_time"]
    num_seeds = data["num_seeds"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    all_sections = []

    # Summary section
    summary_rows = []
    for pname in ["lr", "ent_coef", "batch_size"]:
        pdata = phases[pname]
        param = pdata["param"]
        values = pdata["values"]
        results = pdata["results"]
        best_val = final_config[param]
        for val in values:
            runs = results.get(str(val), [])
            elos = [r["metrics"]["elo"][-1] for r in runs if "elo" in r["metrics"]]
            peaks = [max(r["metrics"]["elo"]) for r in runs if "elo" in r["metrics"]]
            mean_elo = np.mean(elos) if elos else 0
            std_elo = np.std(elos) if elos else 0
            mean_peak = np.mean(peaks) if peaks else 0
            marker = " *" if val == best_val else ""
            summary_rows.append([
                pname, str(val) + marker,
                f"{mean_elo:.0f} +/- {std_elo:.0f}",
                f"{mean_peak:.0f}",
                f"{np.mean([r['wall_time'] for r in runs]):.0f}s" if runs else "N/A",
            ])

    all_sections.append({
        "heading": "Executive Summary",
        "text": (
            "Hyperparameter sweep for Connect 4 self-play PPO with the <b>new residual architecture</b> "
            "(LayerNorm + GELU + residual connections). Sequential 3-phase sweep: "
            "lr → ent_coef → batch_size. Each phase uses the best value from prior phases.\n\n"
            f"<b>Architecture:</b> 256 hidden, 6 layers (1 input proj + 5 residual blocks), "
            f"366,600 params<br/>"
            f"<b>Training:</b> {SWEEP_ITERS} iters, 512 games/iter, {num_seeds} seeds/config<br/>"
            f"<b>Total sweep time:</b> {total_time:.0f}s\n\n"
            f"<b>Best config:</b> lr={final_config['lr']}, ent_coef={final_config['ent_coef']}, "
            f"batch_size={final_config['batch_size']}"
        ),
        "table": {
            "headers": ["Phase", "Value", "Final Elo", "Peak Elo", "Time"],
            "rows": summary_rows,
        },
        "page_break": True,
    })

    # Per-phase plots and sections
    for pname in ["lr", "ent_coef", "batch_size"]:
        pdata = phases[pname]
        param = pdata["param"]
        values = pdata["values"]
        results = pdata["results"]

        # Elo curves
        fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
        for i, val in enumerate(values):
            runs = results.get(str(val), [])
            if not runs:
                continue
            arrays = []
            iters = None
            for r in runs:
                m = r["metrics"]
                if "elo" in m:
                    arrays.append(np.array(m["elo"], dtype=float))
                    if iters is None:
                        iters = np.array(m["eval_iteration"])
            if not arrays:
                continue
            min_len = min(len(a) for a in arrays)
            arrays = [a[:min_len] for a in arrays]
            iters = iters[:min_len]
            stacked = np.stack(arrays)
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            c = colors[i % len(colors)]
            ax.plot(iters, mean, "-o", color=c, markersize=2, linewidth=2,
                    label=f"{param}={val}")
            ax.fill_between(iters, mean - std, mean + std, color=c, alpha=0.15)
        ax.axhline(y=1584, color="red", linestyle="--", alpha=0.5, label="Heuristic (1584)")
        ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Random (1000)")
        ax.set_xlabel("Training Iteration", fontsize=12)
        ax.set_ylabel("Elo Rating", fontsize=12)
        ax.set_title(f"Elo by {param} (mean +/- std, {num_seeds} seeds)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        elo_fname = f"elo_{pname}.png"
        fig.savefig(os.path.join(REPORT_DIR, elo_fname), dpi=150)
        plt.close(fig)

        # Entropy curves
        fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
        for i, val in enumerate(values):
            runs = results.get(str(val), [])
            if not runs:
                continue
            arrays = []
            iters = None
            for r in runs:
                m = r["metrics"]
                if "entropy" in m:
                    arrays.append(smooth(np.array(m["entropy"], dtype=float), 20))
                    if iters is None:
                        iters = np.array(m["iteration"])
            if not arrays:
                continue
            min_len = min(len(a) for a in arrays)
            arrays = [a[:min_len] for a in arrays]
            iters = iters[:min_len]
            stacked = np.stack(arrays)
            mean = stacked.mean(axis=0)
            c = colors[i % len(colors)]
            ax.plot(iters, mean, "-", color=c, linewidth=1.5, label=f"{param}={val}")
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Entropy", fontsize=12)
        ax.set_title(f"Entropy by {param}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        ent_fname = f"entropy_{pname}.png"
        fig.savefig(os.path.join(REPORT_DIR, ent_fname), dpi=150)
        plt.close(fig)

        # Bar chart
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        final_means, final_stds, peak_means, peak_stds = [], [], [], []
        labels = []
        for val in values:
            runs = results.get(str(val), [])
            elos = [r["metrics"]["elo"][-1] for r in runs if "elo" in r["metrics"]]
            peaks = [max(r["metrics"]["elo"]) for r in runs if "elo" in r["metrics"]]
            final_means.append(np.mean(elos) if elos else 0)
            final_stds.append(np.std(elos) if elos else 0)
            peak_means.append(np.mean(peaks) if peaks else 0)
            peak_stds.append(np.std(peaks) if peaks else 0)
            labels.append(str(val))
        x = np.arange(len(labels))
        w = 0.35
        bars1 = ax.bar(x - w/2, final_means, w, yerr=final_stds, capsize=4,
                        label="Final Elo", color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.8)
        bars2 = ax.bar(x + w/2, peak_means, w, yerr=peak_stds, capsize=4,
                        label="Peak Elo", color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.4)
        ax.bar_label(bars1, [f"{m:.0f}" for m in final_means], fontsize=8, padding=3)
        ax.bar_label(bars2, [f"{m:.0f}" for m in peak_means], fontsize=8, padding=3)
        ax.axhline(y=1584, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel("Elo", fontsize=12)
        ax.set_title(f"Final & Peak Elo by {param}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        bar_fname = f"bar_{pname}.png"
        fig.savefig(os.path.join(REPORT_DIR, bar_fname), dpi=150)
        plt.close(fig)

        best_val = final_config[param]
        all_sections.append({
            "heading": f"Phase: {param}",
            "text": f"Sweeping <b>{param}</b> over {values}. Best: <b>{param}={best_val}</b>.",
            "plots": [
                (elo_fname, f"Elo curves by {param} (mean +/- std)."),
                (bar_fname, f"Final and peak Elo by {param}."),
                (ent_fname, f"Entropy over training by {param}."),
            ],
            "page_break": True,
        })

    report_path = os.path.join(REPORT_DIR, "hparam_sweep_v2_report.pdf")
    generate_pdf_report(
        report_path=report_path,
        title="Connect 4 PPO v2: Hparam Sweep (Residual Architecture)",
        sections=all_sections,
        plot_dir=REPORT_DIR,
    )
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        generate_report_from_results()
    else:
        main()
        generate_report_from_results()
