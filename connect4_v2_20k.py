"""Train Connect 4 agent for 20k iterations with best v2 config."""

import json
import time
import os
from src.connect4_c import Connect4TrainerC

config = {
    "hidden_size": 256,
    "num_layers": 6,
    "games_per_iter": 512,
    "lr": 3e-4,
    "ent_coef": 0.001,
    "batch_size": 256,
    "ppo_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "draw_reward": 0.0,
    "opponent_pool_max": 50,
    "snapshot_interval": 25,
    "opponent_sampling": "uniform",
}

OUT_DIR = "reports/connect4_v2_20k"
os.makedirs(OUT_DIR, exist_ok=True)

trainer = Connect4TrainerC(config)

t0 = time.time()
metrics = trainer.train(20000, eval_interval=100, verbose=True)
wall_time = time.time() - t0

results = {
    "metrics": metrics,
    "wall_time": wall_time,
    "total_params": trainer.total_params,
    "config": config,
}

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f)

print(f"\nDone. Wall time: {wall_time:.1f}s ({wall_time/60:.1f} min)")
