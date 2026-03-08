"""Benchmark training speed with detailed per-phase timing."""

import time
import torch
import numpy as np
import random
import copy
import sys
import ctypes
from src.ppo import SelfPlayTrainer, collect_self_play_games, ppo_update
from src.ppo_fast import SelfPlayTrainerFast, collect_self_play_games_fast, ppo_update_fast
from src.ppo_c import SelfPlayTrainerC, _lib, _fp, _llp, _ip
from src.environment import TicTacToe

TUNED_CONFIG = {
    "device": "cpu",
    "lr": 3e-3,
    "ent_coef": 0.05,
    "hidden_size": 256,
    "num_layers": 4,
    "games_per_iter": 512,
    "clip_eps": 0.1,
    "snapshot_interval": 25,
    "opponent_sampling": "uniform",
    "draw_reward": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ppo_epochs": 4,
    "batch_size": 64,
    "max_grad_norm": 0.5,
    "opponent_pool_max": 20,
    "num_iterations": 50,
    "eval_interval": 25,
}


def benchmark(config=None, label=""):
    if config is None:
        config = TUNED_CONFIG.copy()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    num_iters = config["num_iterations"]
    games_per_iter = config["games_per_iter"]

    trainer = SelfPlayTrainer(config)

    t_collect = 0
    t_update = 0
    t_opponent = 0
    t_eval = 0
    t_snapshot = 0

    total_start = time.perf_counter()

    for i in range(num_iters):
        trainer.iteration += 1
        trainer.agent.train()

        t0 = time.perf_counter()
        opponent = trainer.select_opponent()
        t_opponent += time.perf_counter() - t0

        t0 = time.perf_counter()
        buffer, game_results = collect_self_play_games(
            trainer.agent, opponent, games_per_iter, trainer.device, config=config
        )
        t_collect += time.perf_counter() - t0

        t0 = time.perf_counter()
        update_stats = ppo_update(trainer.agent, trainer.optimizer, buffer, config, trainer.device)
        t_update += time.perf_counter() - t0

        if trainer.iteration % trainer.snapshot_interval == 0:
            t0 = time.perf_counter()
            snapshot = trainer.agent.save_snapshot()
            trainer.opponent_pool.append(snapshot)
            if len(trainer.opponent_pool) > trainer.opponent_pool_max:
                keep_indices = [0] + list(range(
                    len(trainer.opponent_pool) - trainer.opponent_pool_max + 1,
                    len(trainer.opponent_pool)
                ))
                trainer.opponent_pool = [trainer.opponent_pool[i] for i in keep_indices]
            t_snapshot += time.perf_counter() - t0

        if trainer.iteration % config["eval_interval"] == 0:
            t0 = time.perf_counter()
            trainer.evaluate()
            t_eval += time.perf_counter() - t0

    total_time = time.perf_counter() - total_start

    print(f"\n{'='*50}", flush=True)
    print(f"BENCHMARK: {label}", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Total time:       {total_time:.3f}s", flush=True)
    print(f"  Collect games:  {t_collect:.3f}s ({t_collect/total_time*100:.1f}%)", flush=True)
    print(f"  PPO update:     {t_update:.3f}s ({t_update/total_time*100:.1f}%)", flush=True)
    print(f"  Select opponent:{t_opponent:.3f}s ({t_opponent/total_time*100:.1f}%)", flush=True)
    print(f"  Evaluation:     {t_eval:.3f}s ({t_eval/total_time*100:.1f}%)", flush=True)
    print(f"  Snapshots:      {t_snapshot:.3f}s ({t_snapshot/total_time*100:.1f}%)", flush=True)
    overhead = total_time - t_collect - t_update - t_opponent - t_eval - t_snapshot
    print(f"  Other overhead: {overhead:.3f}s ({overhead/total_time*100:.1f}%)", flush=True)
    print(f"Per-iteration:    {total_time/num_iters*1000:.1f}ms", flush=True)
    print(f"{'='*50}\n", flush=True)

    return {
        "total": total_time,
        "collect": t_collect,
        "update": t_update,
        "opponent": t_opponent,
        "eval": t_eval,
        "snapshot": t_snapshot,
        "per_iter_ms": total_time / num_iters * 1000,
    }


def benchmark_fast(config=None, label=""):
    if config is None:
        config = TUNED_CONFIG.copy()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    num_iters = config["num_iterations"]
    games_per_iter = config["games_per_iter"]

    trainer = SelfPlayTrainerFast(config)

    t_collect = 0
    t_update = 0
    t_opponent = 0
    t_eval = 0
    t_snapshot = 0

    total_start = time.perf_counter()

    for i in range(num_iters):
        trainer.iteration += 1
        trainer.agent.train()

        t0 = time.perf_counter()
        opponent = trainer.select_opponent()
        t_opponent += time.perf_counter() - t0

        t0 = time.perf_counter()
        buffer_data, game_results = collect_self_play_games_fast(
            trainer.agent, opponent, games_per_iter, trainer.device, config=config
        )
        t_collect += time.perf_counter() - t0

        t0 = time.perf_counter()
        update_stats = ppo_update_fast(trainer.agent, trainer.optimizer, buffer_data, config, trainer.device)
        t_update += time.perf_counter() - t0

        if trainer.iteration % trainer.snapshot_interval == 0:
            t0 = time.perf_counter()
            snapshot = copy.deepcopy(trainer.agent.state_dict())
            trainer.opponent_pool.append(snapshot)
            if len(trainer.opponent_pool) > trainer.opponent_pool_max:
                keep_indices = [0] + list(range(
                    len(trainer.opponent_pool) - trainer.opponent_pool_max + 1,
                    len(trainer.opponent_pool)
                ))
                trainer.opponent_pool = [trainer.opponent_pool[i] for i in keep_indices]
            t_snapshot += time.perf_counter() - t0

        if trainer.iteration % config["eval_interval"] == 0:
            t0 = time.perf_counter()
            trainer.evaluate()
            t_eval += time.perf_counter() - t0

    total_time = time.perf_counter() - total_start

    print(f"\n{'='*50}", flush=True)
    print(f"BENCHMARK: {label}", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Total time:       {total_time:.3f}s", flush=True)
    print(f"  Collect games:  {t_collect:.3f}s ({t_collect/total_time*100:.1f}%)", flush=True)
    print(f"  PPO update:     {t_update:.3f}s ({t_update/total_time*100:.1f}%)", flush=True)
    print(f"  Select opponent:{t_opponent:.3f}s ({t_opponent/total_time*100:.1f}%)", flush=True)
    print(f"  Evaluation:     {t_eval:.3f}s ({t_eval/total_time*100:.1f}%)", flush=True)
    print(f"  Snapshots:      {t_snapshot:.3f}s ({t_snapshot/total_time*100:.1f}%)", flush=True)
    overhead = total_time - t_collect - t_update - t_opponent - t_eval - t_snapshot
    print(f"  Other overhead: {overhead:.3f}s ({overhead/total_time*100:.1f}%)", flush=True)
    print(f"Per-iteration:    {total_time/num_iters*1000:.1f}ms", flush=True)
    print(f"{'='*50}\n", flush=True)

    return {
        "total": total_time,
        "collect": t_collect,
        "update": t_update,
        "opponent": t_opponent,
        "eval": t_eval,
        "snapshot": t_snapshot,
        "per_iter_ms": total_time / num_iters * 1000,
    }


def benchmark_c(config=None, label=""):
    if config is None:
        config = TUNED_CONFIG.copy()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    num_iters = config["num_iterations"]
    games_per_iter = config["games_per_iter"]

    trainer = SelfPlayTrainerC(config)

    t_collect = 0
    t_update = 0
    t_opponent = 0
    t_eval = 0
    t_snapshot = 0

    total_start = time.perf_counter()

    for i in range(num_iters):
        trainer.iteration += 1

        t0 = time.perf_counter()
        opp_params = trainer.select_opponent()
        t_opponent += time.perf_counter() - t0

        _lib.ppo_seed(ctypes.c_ulong(random.getrandbits(64)))

        t0 = time.perf_counter()
        n_trans = _lib.collect_games(
            _fp(trainer.params), _fp(opp_params), games_per_iter, config.get("draw_reward", 0.5),
            _fp(trainer.buf_obs), _llp(trainer.buf_actions), _fp(trainer.buf_log_probs),
            _fp(trainer.buf_values), _fp(trainer.buf_valid_masks),
            _fp(trainer.buf_rewards), _fp(trainer.buf_dones), _ip(trainer.game_results),
        )
        t_collect += time.perf_counter() - t0

        t0 = time.perf_counter()
        _lib.ppo_update(
            _fp(trainer.params), _fp(trainer.adam_m), _fp(trainer.adam_v), _ip(trainer.adam_t),
            _fp(trainer.buf_obs), _llp(trainer.buf_actions), _fp(trainer.buf_log_probs),
            _fp(trainer.buf_values), _fp(trainer.buf_valid_masks),
            _fp(trainer.buf_rewards), _fp(trainer.buf_dones), n_trans,
            config.get("gamma", 0.99), config.get("gae_lambda", 0.95),
            config.get("clip_eps", 0.1), config.get("vf_coef", 0.5),
            config.get("ent_coef", 0.05), config.get("lr", 3e-3),
            config.get("max_grad_norm", 0.5), config.get("ppo_epochs", 4),
            config.get("batch_size", 64), _fp(trainer.stats_out),
        )
        t_update += time.perf_counter() - t0

        if trainer.iteration % trainer.snapshot_interval == 0:
            t0 = time.perf_counter()
            trainer.opponent_pool.append(trainer.params.copy())
            if len(trainer.opponent_pool) > trainer.opponent_pool_max:
                keep = [0] + list(range(len(trainer.opponent_pool) - trainer.opponent_pool_max + 1,
                                        len(trainer.opponent_pool)))
                trainer.opponent_pool = [trainer.opponent_pool[i] for i in keep]
            t_snapshot += time.perf_counter() - t0

        if trainer.iteration % config["eval_interval"] == 0:
            t0 = time.perf_counter()
            trainer.evaluate()
            t_eval += time.perf_counter() - t0

    total_time = time.perf_counter() - total_start

    print(f"\n{'='*50}", flush=True)
    print(f"BENCHMARK: {label}", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"Total time:       {total_time:.3f}s", flush=True)
    print(f"  Collect games:  {t_collect:.3f}s ({t_collect/total_time*100:.1f}%)", flush=True)
    print(f"  PPO update:     {t_update:.3f}s ({t_update/total_time*100:.1f}%)", flush=True)
    print(f"  Select opponent:{t_opponent:.3f}s ({t_opponent/total_time*100:.1f}%)", flush=True)
    print(f"  Evaluation:     {t_eval:.3f}s ({t_eval/total_time*100:.1f}%)", flush=True)
    print(f"  Snapshots:      {t_snapshot:.3f}s ({t_snapshot/total_time*100:.1f}%)", flush=True)
    overhead = total_time - t_collect - t_update - t_opponent - t_eval - t_snapshot
    print(f"  Other overhead: {overhead:.3f}s ({overhead/total_time*100:.1f}%)", flush=True)
    print(f"Per-iteration:    {total_time/num_iters*1000:.1f}ms", flush=True)
    print(f"{'='*50}\n", flush=True)

    return {
        "total": total_time,
        "collect": t_collect,
        "update": t_update,
        "opponent": t_opponent,
        "eval": t_eval,
        "snapshot": t_snapshot,
        "per_iter_ms": total_time / num_iters * 1000,
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("baseline", "all"):
        benchmark(label="BASELINE (Python/PyTorch)")
    if mode in ("fast", "all"):
        benchmark_fast(label="OPTIMIZED (Python/PyTorch batched)")
    if mode in ("c", "all"):
        benchmark_c(label="C BACKEND (Accelerate BLAS)")
