"""Python wrapper for C PPO implementation via ctypes."""

import ctypes
import numpy as np
import os
import time
import copy
import random
from collections import defaultdict

from src.model import TicTacToeNet
from src.environment import play_vs_optimal, play_vs_random

# Load shared library
_LIB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csrc", "ppo_core.dylib")
_lib = ctypes.CDLL(_LIB_PATH)

# Function signatures
_lib.ppo_init.restype = None
_lib.ppo_init.argtypes = []

_lib.ppo_seed.restype = None
_lib.ppo_seed.argtypes = [ctypes.c_ulong]

_f_ptr = ctypes.POINTER(ctypes.c_float)
_ll_ptr = ctypes.POINTER(ctypes.c_longlong)
_i_ptr = ctypes.POINTER(ctypes.c_int)

_lib.ppo_update.restype = None
_lib.ppo_update.argtypes = [
    _f_ptr, _f_ptr, _f_ptr, _i_ptr,          # params, adam_m, adam_v, adam_t
    _f_ptr, _ll_ptr, _f_ptr, _f_ptr,         # obs, actions, log_probs, values
    _f_ptr, _f_ptr, _f_ptr, ctypes.c_int,    # valid_masks, rewards, dones, N
    ctypes.c_float, ctypes.c_float,           # gamma, gae_lambda
    ctypes.c_float, ctypes.c_float,           # clip_eps, vf_coef
    ctypes.c_float, ctypes.c_float,           # ent_coef, lr
    ctypes.c_float, ctypes.c_int,             # max_grad_norm, num_epochs
    ctypes.c_int, _f_ptr,                     # batch_size, out_stats
]

_lib.collect_games.restype = ctypes.c_int
_lib.collect_games.argtypes = [
    _f_ptr, _f_ptr,                           # agent_params, opp_params
    ctypes.c_int, ctypes.c_float,             # num_games, draw_reward
    _f_ptr, _ll_ptr, _f_ptr,                  # out_obs, out_actions, out_log_probs
    _f_ptr, _f_ptr, _f_ptr, _f_ptr,          # out_values, out_valid_masks, out_rewards, out_dones
    _i_ptr,                                    # out_game_results
]

# Initialize C library
_lib.ppo_init()

TOTAL_PARAMS = 207114


def _fp(arr):
    return arr.ctypes.data_as(_f_ptr)

def _llp(arr):
    return arr.ctypes.data_as(_ll_ptr)

def _ip(arr):
    return arr.ctypes.data_as(_i_ptr)


def extract_params(model):
    """Extract model parameters as flat numpy array matching C layout."""
    parts = []
    for name, p in model.named_parameters():
        parts.append(p.data.cpu().numpy().ravel())
    return np.concatenate(parts).astype(np.float32).copy()


def load_params(model, params):
    """Load flat numpy array back into PyTorch model."""
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        p.data.copy_(
            __import__('torch').from_numpy(params[offset:offset+n].reshape(p.shape))
        )
        offset += n


def log(msg):
    print(msg, flush=True)


class SelfPlayTrainerC:
    """Self-play PPO trainer using pure C backend."""

    def __init__(self, config):
        self.config = config
        self.device = "cpu"

        model_kwargs = {
            "hidden_size": config.get("hidden_size", 256),
            "num_layers": config.get("num_layers", 4),
        }
        self.model_kwargs = model_kwargs
        # PyTorch model only used for evaluation
        self.agent = TicTacToeNet(**model_kwargs)

        # C-side state: flat parameter arrays
        self.params = extract_params(self.agent)
        self.adam_m = np.zeros(TOTAL_PARAMS, dtype=np.float32)
        self.adam_v = np.zeros(TOTAL_PARAMS, dtype=np.float32)
        self.adam_t = np.array([0], dtype=np.int32)

        # Opponent pool as list of flat numpy arrays
        self.opponent_pool = [self.params.copy()]
        self.opponent_pool_max = config.get("opponent_pool_max", 20)
        self.snapshot_interval = config.get("snapshot_interval", 10)

        self.iteration = 0
        self.metrics_history = defaultdict(list)

        # Pre-allocate collection buffers
        max_games = config.get("games_per_iter", 512)
        max_trans = max_games * 5
        self.buf_obs = np.zeros((max_trans, 27), dtype=np.float32)
        self.buf_actions = np.zeros(max_trans, dtype=np.int64)
        self.buf_log_probs = np.zeros(max_trans, dtype=np.float32)
        self.buf_values = np.zeros(max_trans, dtype=np.float32)
        self.buf_valid_masks = np.zeros((max_trans, 9), dtype=np.float32)
        self.buf_rewards = np.zeros(max_trans, dtype=np.float32)
        self.buf_dones = np.zeros(max_trans, dtype=np.float32)
        self.game_results = np.zeros(3, dtype=np.int32)
        self.stats_out = np.zeros(4, dtype=np.float32)

    def select_opponent(self):
        strategy = self.config.get("opponent_sampling", "uniform")
        if strategy == "uniform":
            idx = random.randint(0, len(self.opponent_pool) - 1)
        elif strategy == "latest":
            idx = len(self.opponent_pool) - 1
        elif strategy == "weighted_recent":
            n = len(self.opponent_pool)
            weights = np.arange(1, n + 1, dtype=np.float64)
            weights /= weights.sum()
            idx = np.random.choice(n, p=weights)
        else:
            idx = random.randint(0, len(self.opponent_pool) - 1)
        return self.opponent_pool[idx]

    def evaluate(self):
        load_params(self.agent, self.params)
        self.agent.eval()
        policy_fn = self.agent.get_policy_fn("cpu", deterministic=True)

        results = {}
        w1, d1, l1 = play_vs_random(policy_fn, policy_plays_as=1, num_games=50)
        w2, d2, l2 = play_vs_random(policy_fn, policy_plays_as=-1, num_games=50)
        results["vs_random_win_rate"] = (w1 + w2) / 100
        results["vs_random_draw_rate"] = (d1 + d2) / 100
        results["vs_random_loss_rate"] = (l1 + l2) / 100

        w1, d1, l1 = play_vs_optimal(policy_fn, policy_plays_as=1, num_games=1)
        w2, d2, l2 = play_vs_optimal(policy_fn, policy_plays_as=-1, num_games=1)
        results["vs_optimal_win_rate"] = (w1 + w2) / 2
        results["vs_optimal_draw_rate"] = (d1 + d2) / 2
        results["vs_optimal_loss_rate"] = (l1 + l2) / 2
        results["exploitability"] = results["vs_optimal_loss_rate"]

        return results

    def train(self, num_iterations, eval_interval=10, verbose=True):
        games_per_iter = self.config.get("games_per_iter", 512)
        config = self.config

        if verbose:
            log(f"Starting C-backend training: {num_iterations} iterations, {games_per_iter} games/iter")

        for i in range(num_iterations):
            self.iteration += 1
            iter_start = time.time()

            # Select opponent
            opp_params = self.select_opponent()

            # Seed C RNG (mix iteration with random for variety)
            _lib.ppo_seed(ctypes.c_ulong(random.getrandbits(64)))

            # Collect games in C
            n_trans = _lib.collect_games(
                _fp(self.params), _fp(opp_params),
                games_per_iter, config.get("draw_reward", 0.0),
                _fp(self.buf_obs), _llp(self.buf_actions), _fp(self.buf_log_probs),
                _fp(self.buf_values), _fp(self.buf_valid_masks),
                _fp(self.buf_rewards), _fp(self.buf_dones),
                _ip(self.game_results),
            )

            # PPO update in C
            _lib.ppo_update(
                _fp(self.params), _fp(self.adam_m), _fp(self.adam_v), _ip(self.adam_t),
                _fp(self.buf_obs), _llp(self.buf_actions), _fp(self.buf_log_probs),
                _fp(self.buf_values), _fp(self.buf_valid_masks),
                _fp(self.buf_rewards), _fp(self.buf_dones), n_trans,
                config.get("gamma", 0.99), config.get("gae_lambda", 0.95),
                config.get("clip_eps", 0.2), config.get("vf_coef", 0.5),
                config.get("ent_coef", 0.01), config.get("lr", 3e-4),
                config.get("max_grad_norm", 0.5), config.get("ppo_epochs", 4),
                config.get("batch_size", 64), _fp(self.stats_out),
            )

            iter_time = time.time() - iter_start
            gr = self.game_results

            self.metrics_history["iteration"].append(self.iteration)
            self.metrics_history["sp_win_rate"].append(gr[0] / games_per_iter)
            self.metrics_history["sp_draw_rate"].append(gr[1] / games_per_iter)
            self.metrics_history["sp_loss_rate"].append(gr[2] / games_per_iter)
            self.metrics_history["policy_loss"].append(float(self.stats_out[0]))
            self.metrics_history["value_loss"].append(float(self.stats_out[1]))
            self.metrics_history["entropy"].append(float(self.stats_out[2]))
            self.metrics_history["approx_kl"].append(float(self.stats_out[3]))
            self.metrics_history["pool_size"].append(len(self.opponent_pool))

            if verbose:
                log(
                    f"[iter {self.iteration:4d}/{num_iterations}] "
                    f"SP W/D/L: {gr[0]:3d}/{gr[1]:3d}/{gr[2]:3d} | "
                    f"ploss: {self.stats_out[0]:.4f} vloss: {self.stats_out[1]:.4f} "
                    f"ent: {self.stats_out[2]:.3f} kl: {self.stats_out[3]:.4f} | "
                    f"{iter_time*1000:.1f}ms"
                )

            if self.iteration % self.snapshot_interval == 0:
                self.opponent_pool.append(self.params.copy())
                if len(self.opponent_pool) > self.opponent_pool_max:
                    keep = [0] + list(range(
                        len(self.opponent_pool) - self.opponent_pool_max + 1,
                        len(self.opponent_pool)
                    ))
                    self.opponent_pool = [self.opponent_pool[i] for i in keep]

            if self.iteration % eval_interval == 0:
                eval_start = time.time()
                eval_results = self.evaluate()
                eval_time = time.time() - eval_start
                for key, val in eval_results.items():
                    self.metrics_history[key].append(val)
                self.metrics_history["eval_iteration"].append(self.iteration)

                if verbose:
                    log(
                        f"  EVAL @ {self.iteration}: "
                        f"vs Random W: {eval_results['vs_random_win_rate']:.2f} | "
                        f"vs Optimal D: {eval_results['vs_optimal_draw_rate']:.2f} "
                        f"L: {eval_results['vs_optimal_loss_rate']:.2f} | "
                        f"Pool: {len(self.opponent_pool)} | eval {eval_time:.1f}s"
                    )

        # Sync params back to PyTorch model
        load_params(self.agent, self.params)
        return dict(self.metrics_history)
