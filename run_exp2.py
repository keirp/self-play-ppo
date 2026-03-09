"""Experiment 2: Draw reward + best entropy from exp1 (parallel)."""
import subprocess
import sys
import time
import json
import numpy as np

# Will use best ent_coef from exp1 — for now test both with ent_coef=0.01
# which is likely the winner based on TicTacToe experience
configs = [
    ("exp2_dr00", {"draw_reward": 0.0}),           # baseline
    ("exp2_dr02", {"draw_reward": 0.2}),
    ("exp2_dr05", {"draw_reward": 0.5}),
    ("exp2_dr10", {"draw_reward": 1.0}),
]

NUM_ITERS = 2000
MAX_PARALLEL = 4

procs = []
t0 = time.time()

for name, overrides in configs:
    cmd = [
        sys.executable, "experiment_runner.py",
        name, str(NUM_ITERS), json.dumps(overrides),
        "--eval-interval", "100",
    ]
    print(f"Starting {name}...")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    procs.append((name, p))

for name, p in procs:
    stdout, _ = p.communicate()
    output = stdout.decode()
    lines = output.strip().split("\n")
    for line in lines[-8:]:
        print(line)
    print()

elapsed = time.time() - t0
print(f"All experiments done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

print("\n" + "="*70)
print(f"{'Config':<25} {'Final Elo':>10} {'Avg Elo (last 10)':>20} {'Peak Elo':>10}")
print("="*70)
for name, _ in configs:
    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        last_10 = np.mean(elos[-10:])
        print(f"{name:<25} {elos[-1]:>10.0f} {last_10:>20.0f} {max(elos):>10.0f}")
    except Exception as e:
        print(f"{name:<25} ERROR: {e}")
