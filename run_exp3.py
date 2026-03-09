"""Experiment 3: Games per iteration + model size (parallel)."""
import subprocess
import sys
import time
import json
import numpy as np

configs = [
    # More games per iteration
    ("exp3_gpi1024", {"games_per_iter": 1024}),
    ("exp3_gpi2048", {"games_per_iter": 2048}),
    # Larger model (need subprocess for different c4_init)
    ("exp3_h512_l6", {"hidden_size": 512, "num_layers": 6}),
    ("exp3_h256_l8", {"hidden_size": 256, "num_layers": 8}),
]

NUM_ITERS = 2000

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
print(f"{'Config':<20} {'Final Elo':>10} {'Avg(last10)':>12} {'Peak Elo':>10}")
print("="*70)
for name, _ in configs:
    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        last_10 = np.mean(elos[-10:])
        print(f"{name:<20} {elos[-1]:>10.0f} {last_10:>12.0f} {max(elos):>10.0f}")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")
