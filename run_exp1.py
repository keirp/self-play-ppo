"""Experiment 1: Entropy coefficient sweep (parallel)."""
import subprocess
import sys
import time
import json
import numpy as np

configs = [
    ("exp1_ent_001", {"ent_coef": 0.001}),  # baseline
    ("exp1_ent_01", {"ent_coef": 0.01}),
    ("exp1_ent_03", {"ent_coef": 0.03}),
    ("exp1_ent_05", {"ent_coef": 0.05}),
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

# Wait for all
for name, p in procs:
    stdout, _ = p.communicate()
    output = stdout.decode()
    # Print last 5 lines (summary)
    lines = output.strip().split("\n")
    for line in lines[-8:]:
        print(line)
    print()

elapsed = time.time() - t0
print(f"All experiments done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# Load and compare results
print("\n" + "="*70)
print(f"{'Config':<20} {'Final Elo':>10} {'Avg Elo (last 10)':>20} {'Peak Elo':>10}")
print("="*70)
for name, _ in configs:
    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        last_10 = np.mean(elos[-10:])
        print(f"{name:<20} {elos[-1]:>10.0f} {last_10:>20.0f} {max(elos):>10.0f}")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")
