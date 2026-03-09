"""Experiment 5: Scale data + longer training."""
import subprocess
import sys
import time
import json
import numpy as np

configs = [
    # Scale games per iter
    ("exp5_gpi2048_5k", {"games_per_iter": 2048}, 5000),
    ("exp5_gpi4096_5k", {"games_per_iter": 4096}, 5000),
    # Also try scaling batch size to match
    ("exp5_gpi2048_bs512_5k", {"games_per_iter": 2048, "batch_size": 512}, 5000),
]

procs = []
t0 = time.time()

for name, overrides, num_iters in configs:
    cmd = [
        sys.executable, "experiment_runner.py",
        name, str(num_iters), json.dumps(overrides),
        "--eval-interval", "200",
    ]
    print(f"Starting {name} ({num_iters} iters)...")
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
print(f"{'Config':<30} {'Final Elo':>10} {'Avg(last5)':>12} {'Peak Elo':>10}")
print("="*80)
for name, _, _ in configs:
    try:
        with open(f"experiments/{name}/results.json") as f:
            data = json.load(f)
        elos = data["metrics"]["elo"]
        last_5 = np.mean(elos[-5:])
        print(f"{name:<30} {elos[-1]:>10.0f} {last_5:>12.0f} {max(elos):>10.0f}")
    except Exception as e:
        print(f"{name:<30} ERROR: {e}")
