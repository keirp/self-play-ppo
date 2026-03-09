"""Experiment 4: Mixed opponents - adding heuristic to training pool."""
import subprocess
import sys
import time
import json
import numpy as np

configs = [
    ("exp4_baseline", {}),
    ("exp4_pinned20", {
        "pinned_opponent_paths": ["data/heuristic_net_params.npy"],
        "pinned_frac": 0.2,
    }),
    ("exp4_pinned50", {
        "pinned_opponent_paths": ["data/heuristic_net_params.npy"],
        "pinned_frac": 0.5,
    }),
]

NUM_ITERS = 3000

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

print("\n" + "="*80)
print(f"{'Config':<20} {'Final Elo':>10} {'Avg(last10)':>12} {'Peak Elo':>10} {'vsHeur W%':>10}")
print("="*80)
for name, _ in configs:
    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        hw = data["metrics"]["vs_heuristic_win_rate"]
        last_10 = np.mean(elos[-10:])
        last_hw = np.mean(hw[-5:])
        print(f"{name:<20} {elos[-1]:>10.0f} {last_10:>12.0f} {max(elos):>10.0f} {last_hw:>10.1%}")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")
