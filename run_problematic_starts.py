"""Run problematic starting positions experiment for Connect 4.

Sweeps problematic_start_frac across [0, 0.1, 0.2, 0.3, 0.5].
Each config runs for 5000 iters with eval every 50 iters.
Primary metric: avg Elo of last 100 eval points.
"""
import sys
import os
import time
import json
import subprocess
import tempfile
import numpy as np

NUM_ITERS = 5000
EVAL_INTERVAL = 50
NUM_SEEDS = 2
MAX_PARALLEL = 4
OUT_DIR = "experiments/problematic_starts"

SWEEP_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]

BASE_CONFIG = {
    "hidden_size": 256,
    "num_layers": 6,
    "lr": 3e-4,
    "ent_coef": 0.001,
    "batch_size": 256,
    "clip_eps": 0.2,
    "games_per_iter": 2048,
    "opp_temperature": 1.5,
    "elo_games_per_opp": 100,
    "opponent_pool_max": 50,
    "snapshot_interval": 25,
    "ppo_epochs": 4,
    "problematic_buffer_size": 10000,
}


def make_script(config, seed, out_path):
    return f'''
import sys, os, json, random, numpy as np
sys.path.insert(0, "{os.getcwd()}")
random.seed({seed})
np.random.seed({seed})

from src.connect4_c import Connect4TrainerC, _lib
import ctypes
_lib.c4_seed(ctypes.c_ulong({seed}))

config = {json.dumps(config)}
trainer = Connect4TrainerC(config)
metrics = trainer.train(num_iterations={NUM_ITERS}, eval_interval={EVAL_INTERVAL}, verbose=True)

os.makedirs("{out_path}", exist_ok=True)
with open(os.path.join("{out_path}", "metrics.json"), "w") as f:
    json.dump(metrics, f)
np.save(os.path.join("{out_path}", "final_params.npy"), trainer.params)
if trainer.best_params is not None:
    np.save(os.path.join("{out_path}", "best_params.npy"), trainer.best_params)
print("DONE")
'''


def run_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = []

    for frac in SWEEP_VALUES:
        name = f"prob_{int(frac*100):02d}pct"
        config = {**BASE_CONFIG, "problematic_start_frac": frac}
        for seed in range(NUM_SEEDS):
            out_path = os.path.join(OUT_DIR, f"{name}_s{seed}")
            jobs.append({
                "name": f"{name}_s{seed}",
                "config": config,
                "seed": seed + 42,
                "out_path": out_path,
            })

    print(f"Running {len(jobs)} jobs, {MAX_PARALLEL} at a time")
    running = []
    results = []
    job_idx = 0

    while job_idx < len(jobs) or running:
        # Launch new jobs
        while len(running) < MAX_PARALLEL and job_idx < len(jobs):
            job = jobs[job_idx]
            script = make_script(job["config"], job["seed"], job["out_path"])
            tf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
            tf.write(script)
            tf.close()
            log_path = os.path.join(job["out_path"], "train.log")
            os.makedirs(job["out_path"], exist_ok=True)
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, tf.name],
                stdout=log_file, stderr=subprocess.STDOUT,
            )
            running.append({"proc": proc, "job": job, "script": tf.name, "start": time.time(), "log_file": log_file})
            print(f"  Started {job['name']}")
            job_idx += 1

        # Check for completion
        still_running = []
        for r in running:
            ret = r["proc"].poll()
            if ret is not None:
                elapsed = time.time() - r["start"]
                r["log_file"].close()
                os.unlink(r["script"])
                log_path = os.path.join(r["job"]["out_path"], "train.log")
                try:
                    with open(log_path) as f:
                        stdout = f.read()
                except Exception:
                    stdout = ""
                success = "DONE" in stdout
                # Extract last Elo from stdout
                elo_lines = [l for l in stdout.split("\n") if "Elo:" in l]
                last_elo = "?"
                if elo_lines:
                    try:
                        last_elo = elo_lines[-1].split("Elo:")[1].split("|")[0].strip()
                    except Exception:
                        pass
                status = "OK" if success else "FAIL"
                print(f"  {status} {r['job']['name']} ({elapsed/60:.1f}m) last_elo={last_elo}")
                if not success:
                    for line in stdout.split("\n")[-10:]:
                        if line.strip():
                            print(f"    {line}")
                results.append({"job": r["job"], "success": success, "elapsed": elapsed})
            else:
                still_running.append(r)
        running = still_running

        if running:
            time.sleep(2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for frac in SWEEP_VALUES:
        name = f"prob_{int(frac*100):02d}pct"
        elos = []
        for seed in range(NUM_SEEDS):
            mpath = os.path.join(OUT_DIR, f"{name}_s{seed}", "metrics.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    m = json.load(f)
                avg100 = np.mean(m["elo"][-100:])
                elos.append(avg100)
        if elos:
            print(f"  {name}: avg_last_100 = {np.mean(elos):.0f} ± {np.std(elos):.0f} (seeds: {[f'{e:.0f}' for e in elos]})")


if __name__ == "__main__":
    t0 = time.time()
    run_all()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f}m")
