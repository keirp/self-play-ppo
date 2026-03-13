"""Python wrapper for Connect 4 C PPO implementation via ctypes."""

import ctypes
import numpy as np
import os
import time
import random
import torch
from collections import defaultdict

from src.connect4 import Connect4, play_vs_random, play_vs_heuristic

# Load shared library
_LIB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csrc", "connect4_ppo.dylib")
_lib = ctypes.CDLL(_LIB_PATH)

_f_ptr = ctypes.POINTER(ctypes.c_float)
_ll_ptr = ctypes.POINTER(ctypes.c_longlong)
_i_ptr = ctypes.POINTER(ctypes.c_int)

# Function signatures
_lib.c4_init.restype = None
_lib.c4_init.argtypes = [ctypes.c_int, ctypes.c_int]

_lib.c4_total_params.restype = ctypes.c_int
_lib.c4_total_params.argtypes = []

_lib.c4_seed.restype = None
_lib.c4_seed.argtypes = [ctypes.c_ulong]

_lib.c4_ppo_update.restype = None
_lib.c4_ppo_update.argtypes = [
    _f_ptr, _f_ptr, _f_ptr, _i_ptr,
    _f_ptr, _ll_ptr, _f_ptr, _f_ptr,
    _f_ptr, _f_ptr, _f_ptr, ctypes.c_int,
    ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_int,
    ctypes.c_int, _f_ptr,
]

_lib.c4_collect_games.restype = ctypes.c_int
_lib.c4_collect_games.argtypes = [
    _f_ptr, _f_ptr,
    ctypes.c_int, ctypes.c_float, ctypes.c_float,
    _f_ptr, _ll_ptr, _f_ptr,
    _f_ptr, _f_ptr, _f_ptr, _f_ptr,
    _i_ptr,
]

_lib.c4_collect_games_from_pos.restype = ctypes.c_int
_lib.c4_collect_games_from_pos.argtypes = [
    _f_ptr, _f_ptr,
    ctypes.c_int, ctypes.c_float, ctypes.c_float,
    _f_ptr, _f_ptr,  # start_boards, start_current_players
    _f_ptr, _ll_ptr, _f_ptr,
    _f_ptr, _f_ptr, _f_ptr, _f_ptr,
    _i_ptr,
    _f_ptr,  # out_agent_players
    _i_ptr,  # out_trans_game_idx
]

INPUT_DIM = 126  # 6*7*3
POLICY_DIM = 7


def _fp(arr):
    return arr.ctypes.data_as(_f_ptr)

def _llp(arr):
    return arr.ctypes.data_as(_ll_ptr)

def _ip(arr):
    return arr.ctypes.data_as(_i_ptr)


class ResidualBlock(torch.nn.Module):
    """Pre-norm residual block: x = x + GELU(Linear(LayerNorm(x)))"""

    def __init__(self, hidden_size):
        super().__init__()
        self.ln = torch.nn.LayerNorm(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return x + torch.nn.functional.gelu(self.linear(self.ln(x)))


class Connect4Net(torch.nn.Module):
    """Connect 4 actor-critic network with residual blocks, LayerNorm, GELU.

    Architecture (matches C backend layout):
        Input projection: Linear(126, H)
        Residual blocks:  x = x + GELU(Linear(LayerNorm(x)))  [num_layers-1 times]
        Final LayerNorm:  LayerNorm(x)
        Output heads:     Linear(H, 7), Linear(H, 1)

    Parameter order (must match C for extract_params/load_params):
        input_proj.weight, input_proj.bias,
        blocks[i].ln.weight, blocks[i].ln.bias, blocks[i].linear.weight, blocks[i].linear.bias,
        final_ln.weight, final_ln.bias,
        policy_head.weight, policy_head.bias,
        value_head.weight, value_head.bias
    """

    def __init__(self, hidden_size=512, num_layers=4):
        super().__init__()
        self.input_proj = torch.nn.Linear(INPUT_DIM, hidden_size)
        num_blocks = num_layers - 1
        self.blocks = torch.nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        self.final_ln = torch.nn.LayerNorm(hidden_size)
        self.policy_head = torch.nn.Linear(hidden_size, POLICY_DIM)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, obs, valid_moves_mask=None):
        if obs.dim() == 3:
            obs = obs.reshape(obs.shape[0], -1)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.input_proj(obs)
        for block in self.blocks:
            h = block(h)
        h = self.final_ln(h)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        if valid_moves_mask is not None:
            if valid_moves_mask.dim() == 1:
                valid_moves_mask = valid_moves_mask.unsqueeze(0)
            logits = logits - 1e8 * (1 - valid_moves_mask)
        return logits, value

    def get_policy_fn(self, device, deterministic=False):
        self.eval()
        @torch.no_grad()
        def policy_fn(obs, valid_mask):
            obs_t = torch.from_numpy(obs.flatten()).float().unsqueeze(0).to(device)
            mask_t = torch.from_numpy(valid_mask).float().unsqueeze(0).to(device)
            logits, _ = self(obs_t, mask_t)
            if deterministic:
                return logits.argmax(dim=-1).item()
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
        return policy_fn


def extract_params(model):
    parts = []
    for name, p in model.named_parameters():
        parts.append(p.data.cpu().numpy().ravel())
    return np.concatenate(parts).astype(np.float32).copy()


def load_params(model, params):
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        p.data.copy_(torch.from_numpy(params[offset:offset+n].reshape(p.shape)))
        offset += n


def log(msg):
    print(msg, flush=True)


class ProblematicStatesBuffer:
    """Circular buffer of (board, current_player) states from lost games."""

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.boards = np.zeros((max_size, 42), dtype=np.float32)
        self.current_players = np.zeros(max_size, dtype=np.float32)
        self.count = 0
        self.size = 0

    def add_states(self, boards, current_players):
        for i in range(len(boards)):
            if self.size < self.max_size:
                self.boards[self.size] = boards[i]
                self.current_players[self.size] = current_players[i]
                self.size += 1
            else:
                j = np.random.randint(0, self.count + 1)
                if j < self.max_size:
                    self.boards[j] = boards[i]
                    self.current_players[j] = current_players[i]
            self.count += 1

    def sample(self, n):
        if self.size == 0:
            return None, None
        indices = np.random.randint(0, self.size, size=n)
        return self.boards[indices].copy(), self.current_players[indices].copy()


class Connect4TrainerC:
    """Self-play PPO trainer for Connect 4 using pure C backend."""

    def __init__(self, config):
        self.config = config
        self.device = "cpu"

        hidden_size = config.get("hidden_size", 512)
        num_layers = config.get("num_layers", 4)

        # Initialize C library with architecture
        _lib.c4_init(hidden_size, num_layers)
        total_params = _lib.c4_total_params()

        # PyTorch model for evaluation only
        self.agent = Connect4Net(hidden_size=hidden_size, num_layers=num_layers)

        # C-side state
        self.params = extract_params(self.agent)
        assert len(self.params) == total_params, f"Param count mismatch: {len(self.params)} vs {total_params}"
        self.total_params = total_params
        self.adam_m = np.zeros(total_params, dtype=np.float32)
        self.adam_v = np.zeros(total_params, dtype=np.float32)
        self.adam_t = np.array([0], dtype=np.int32)

        self.opponent_pool = [self.params.copy()]
        self.opponent_pool_max = config.get("opponent_pool_max", 20)
        self.snapshot_interval = config.get("snapshot_interval", 25)
        self.snapshot_count = 1  # total snapshots seen (for reservoir sampling)

        # Pinned opponents (never evicted by reservoir sampling)
        self.pinned_opponents = []
        pinned_paths = config.get("pinned_opponent_paths", [])
        for path in pinned_paths:
            p = np.load(path).astype(np.float32)
            self.pinned_opponents.append(p)
        # Fraction of games played against pinned opponents
        self.pinned_frac = config.get("pinned_frac", 0.0)

        self.iteration = 0
        self.metrics_history = defaultdict(list)
        self.best_elo = -float("inf")
        self.best_params = None

        # Pre-allocate buffers (2x for mirror augmentation)
        max_games = config.get("games_per_iter", 1024)
        max_trans = max_games * 22  # ~21 agent moves max per game
        self.mirror_augment = config.get("mirror_augment", False)
        buf_mult = 2 if self.mirror_augment else 1
        self.buf_obs = np.zeros((max_trans * buf_mult, INPUT_DIM), dtype=np.float32)
        self.buf_actions = np.zeros(max_trans * buf_mult, dtype=np.int64)
        self.buf_log_probs = np.zeros(max_trans * buf_mult, dtype=np.float32)
        self.buf_values = np.zeros(max_trans * buf_mult, dtype=np.float32)
        self.buf_valid_masks = np.zeros((max_trans * buf_mult, POLICY_DIM), dtype=np.float32)
        self.buf_rewards = np.zeros(max_trans * buf_mult, dtype=np.float32)
        self.buf_dones = np.zeros(max_trans * buf_mult, dtype=np.float32)
        self.game_results = np.zeros(3, dtype=np.int32)
        self.stats_out = np.zeros(4, dtype=np.float32)

        # Problematic starting positions
        self.problematic_start_frac = config.get("problematic_start_frac", 0.0)
        self.problematic_buffer = ProblematicStatesBuffer(
            max_size=config.get("problematic_buffer_size", 10000)
        )
        self.buf_agent_players = np.zeros(max_games, dtype=np.float32)
        self.buf_trans_game_idx = np.zeros(max_trans, dtype=np.int32)
        self.buf_start_boards = np.zeros((max_games, 42), dtype=np.float32)
        self.buf_start_cplayers = np.ones(max_games, dtype=np.float32)

        # Precompute mirror index for observations: swap columns in 6x7x3 layout
        # obs layout: obs[row * 7 * 3 + col * 3 + ch] for row in 0..5, col in 0..6, ch in 0..2
        self._mirror_obs_idx = np.zeros(INPUT_DIM, dtype=np.int32)
        for r in range(6):
            for c in range(7):
                for ch in range(3):
                    src = r * 7 * 3 + c * 3 + ch
                    dst = r * 7 * 3 + (6 - c) * 3 + ch
                    self._mirror_obs_idx[src] = dst

    def select_opponent(self):
        # With probability pinned_frac, play against a pinned opponent
        if self.pinned_opponents and random.random() < self.pinned_frac:
            return random.choice(self.pinned_opponents)

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

    def _extract_problematic_states(self, buf_offset, n_trans, num_games):
        """Extract board states from lost games where agent was overconfident.

        Only stores states where value > 0.3 in games the agent lost — positions
        where the agent thought it was winning but actually lost. These represent
        fundamental misunderstandings in the agent's evaluation.
        """
        obs = self.buf_obs[buf_offset:buf_offset + n_trans]
        values = self.buf_values[buf_offset:buf_offset + n_trans]
        rewards = self.buf_rewards[buf_offset:buf_offset + n_trans]
        dones = self.buf_dones[buf_offset:buf_offset + n_trans]
        tgi = self.buf_trans_game_idx[:n_trans]

        # Build per-game outcome map
        game_outcome = {}
        for i in range(n_trans):
            if dones[i] > 0.5:
                game_outcome[tgi[i]] = rewards[i]

        outcomes = np.array([game_outcome.get(tgi[i], 0) for i in range(n_trans)])
        lost = outcomes < -0.5

        # Select states where agent was overconfident (V > 0.3) but lost
        mask = lost & (values > 0.3)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return

        sel_obs = obs[indices]
        sel_game_ids = tgi[indices]

        my_pieces = sel_obs[:, 0::3]
        opp_pieces = sel_obs[:, 1::3]
        ap = self.buf_agent_players[sel_game_ids]
        ap_col = ap.reshape(-1, 1)

        boards = my_pieces * ap_col + opp_pieces * (-ap_col)
        self.problematic_buffer.add_states(boards, ap.copy())

    def evaluate(self, num_games=100, compute_elo=True, elo_games_per_opp=20):
        load_params(self.agent, self.params)
        self.agent.eval()
        # Use stochastic policy for evaluation to get smoother signal
        policy_fn_stoch = self.agent.get_policy_fn("cpu", deterministic=False)

        results = {}
        ng = num_games // 2
        w1, d1, l1 = play_vs_random(policy_fn_stoch, policy_plays_as=1, num_games=ng)
        w2, d2, l2 = play_vs_random(policy_fn_stoch, policy_plays_as=-1, num_games=ng)
        total = ng * 2
        results["vs_random_win_rate"] = (w1 + w2) / total
        results["vs_random_draw_rate"] = (d1 + d2) / total
        results["vs_random_loss_rate"] = (l1 + l2) / total

        w1, d1, l1 = play_vs_heuristic(policy_fn_stoch, policy_plays_as=1, num_games=ng)
        w2, d2, l2 = play_vs_heuristic(policy_fn_stoch, policy_plays_as=-1, num_games=ng)
        results["vs_heuristic_win_rate"] = (w1 + w2) / total
        results["vs_heuristic_draw_rate"] = (d1 + d2) / total
        results["vs_heuristic_loss_rate"] = (l1 + l2) / total

        if compute_elo:
            from src.elo import compute_elo as _compute_elo
            elo_result = _compute_elo(
                self.params,
                hidden_size=self.config.get("hidden_size", 512),
                num_layers=self.config.get("num_layers", 4),
                games_per_opponent=elo_games_per_opp,
            )
            results["elo"] = elo_result["elo"]

        return results

    def train(self, num_iterations, eval_interval=25, verbose=True):
        import math
        games_per_iter = self.config.get("games_per_iter", 1024)
        config = self.config
        base_lr = config.get("lr", 3e-4)
        lr_schedule = config.get("lr_schedule", "constant")  # "constant" or "cosine"
        lr_min_frac = config.get("lr_min_frac", 0.1)  # minimum LR as fraction of base
        base_ent = config.get("ent_coef", 0.01)
        ent_schedule = config.get("ent_schedule", "constant")  # "constant" or "cosine"
        ent_min_frac = config.get("ent_min_frac", 0.1)  # minimum ent_coef as fraction of base

        if verbose:
            log(f"Starting C4 training: {num_iterations} iters, {games_per_iter} games/iter, "
                f"H={config.get('hidden_size',512)}, L={config.get('num_layers',4)}, "
                f"params={self.total_params}, lr_schedule={lr_schedule}")

        for i in range(num_iterations):
            self.iteration += 1
            iter_start = time.time()

            progress = i / max(num_iterations - 1, 1)

            # Compute learning rate
            if lr_schedule == "cosine":
                lr = base_lr * (lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))
            else:
                lr = base_lr

            # Compute entropy coefficient
            if ent_schedule == "cosine":
                ent_coef = base_ent * (ent_min_frac + (1 - ent_min_frac) * 0.5 * (1 + math.cos(math.pi * progress)))
            else:
                ent_coef = base_ent

            # Compute opponent temperature (for curriculum: start high, decay to 1.0)
            base_opp_temp = config.get("opp_temperature", 1.0)
            opp_temp_schedule = config.get("opp_temp_schedule", "constant")
            if opp_temp_schedule == "cosine":
                # Decay from base_opp_temp to 1.0
                opp_temp = 1.0 + (base_opp_temp - 1.0) * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                opp_temp = base_opp_temp

            num_opps_per_iter = config.get("num_opps_per_iter", 1)
            games_per_opp = games_per_iter // num_opps_per_iter
            n_trans = 0
            self.game_results[:] = 0
            _tmp_gr = np.zeros(3, dtype=np.int32)

            for opp_idx in range(num_opps_per_iter):
                opp_params = self.select_opponent()
                _lib.c4_seed(ctypes.c_ulong(random.getrandbits(64)))

                # Prepare starting positions
                n_prob = 0
                if self.problematic_start_frac > 0 and self.problematic_buffer.size > 0:
                    n_prob = int(games_per_opp * self.problematic_start_frac)

                self.buf_start_boards[:games_per_opp] = 0.0
                self.buf_start_cplayers[:games_per_opp] = 1.0
                if n_prob > 0:
                    prob_boards, prob_cplayers = self.problematic_buffer.sample(n_prob)
                    self.buf_start_boards[:n_prob] = prob_boards
                    self.buf_start_cplayers[:n_prob] = prob_cplayers

                nt = _lib.c4_collect_games_from_pos(
                    _fp(self.params), _fp(opp_params),
                    games_per_opp, config.get("draw_reward", 0.0),
                    opp_temp,
                    _fp(self.buf_start_boards), _fp(self.buf_start_cplayers),
                    _fp(self.buf_obs[n_trans:]), _llp(self.buf_actions[n_trans:]),
                    _fp(self.buf_log_probs[n_trans:]),
                    _fp(self.buf_values[n_trans:]), _fp(self.buf_valid_masks[n_trans:]),
                    _fp(self.buf_rewards[n_trans:]), _fp(self.buf_dones[n_trans:]),
                    _ip(_tmp_gr),
                    _fp(self.buf_agent_players), _ip(self.buf_trans_game_idx),
                )
                self.game_results += _tmp_gr

                # Extract problematic states from lost games
                if self.problematic_start_frac > 0 and nt > 0:
                    self._extract_problematic_states(n_trans, nt, games_per_opp)

                n_trans += nt

            # Mirror augmentation: append left-right mirrored copies of all transitions
            if self.mirror_augment and n_trans > 0:
                n = n_trans
                # Mirror observations: swap columns
                self.buf_obs[n:2*n] = self.buf_obs[:n][:, self._mirror_obs_idx]
                # Mirror actions: column c -> 6-c
                self.buf_actions[n:2*n] = 6 - self.buf_actions[:n]
                # Mirror valid masks: swap columns
                self.buf_valid_masks[n:2*n] = self.buf_valid_masks[:n][:, ::-1]
                # Compute CORRECT log_probs and values for mirrored obs/actions
                # (can't reuse originals because policy isn't symmetric)
                load_params(self.agent, self.params)
                self.agent.eval()
                with torch.no_grad():
                    m_obs_t = torch.from_numpy(self.buf_obs[n:2*n].copy())
                    m_mask_t = torch.from_numpy(self.buf_valid_masks[n:2*n].copy())
                    m_act_t = torch.from_numpy(self.buf_actions[n:2*n].copy()).long()
                    logits, values = self.agent(m_obs_t, m_mask_t)
                    log_probs_all = torch.log_softmax(logits, dim=-1)
                    m_log_probs = log_probs_all.gather(1, m_act_t.unsqueeze(1)).squeeze(1)
                    self.buf_log_probs[n:2*n] = m_log_probs.numpy()
                    self.buf_values[n:2*n] = values.numpy()
                # Copy unchanged buffers (rewards and dones are the same)
                self.buf_rewards[n:2*n] = self.buf_rewards[:n]
                self.buf_dones[n:2*n] = self.buf_dones[:n]
                n_trans = 2 * n

            _lib.c4_ppo_update(
                _fp(self.params), _fp(self.adam_m), _fp(self.adam_v), _ip(self.adam_t),
                _fp(self.buf_obs), _llp(self.buf_actions), _fp(self.buf_log_probs),
                _fp(self.buf_values), _fp(self.buf_valid_masks),
                _fp(self.buf_rewards), _fp(self.buf_dones), n_trans,
                config.get("gamma", 0.99), config.get("gae_lambda", 0.95),
                config.get("clip_eps", 0.2), config.get("vf_coef", 0.5),
                ent_coef, lr,
                config.get("max_grad_norm", 0.5), config.get("ppo_epochs", 4),
                config.get("batch_size", 256), _fp(self.stats_out),
            )

            iter_time = time.time() - iter_start
            gr = self.game_results

            self.metrics_history["iteration"].append(self.iteration)
            self.metrics_history["lr"].append(lr)
            self.metrics_history["sp_win_rate"].append(gr[0] / games_per_iter)
            self.metrics_history["sp_draw_rate"].append(gr[1] / games_per_iter)
            self.metrics_history["sp_loss_rate"].append(gr[2] / games_per_iter)
            self.metrics_history["policy_loss"].append(float(self.stats_out[0]))
            self.metrics_history["value_loss"].append(float(self.stats_out[1]))
            self.metrics_history["entropy"].append(float(self.stats_out[2]))
            self.metrics_history["approx_kl"].append(float(self.stats_out[3]))
            self.metrics_history["pool_size"].append(len(self.opponent_pool))
            self.metrics_history["n_transitions"].append(n_trans)

            if verbose:
                log(
                    f"[iter {self.iteration:4d}/{num_iterations}] "
                    f"SP W/D/L: {gr[0]:3d}/{gr[1]:3d}/{gr[2]:3d} | "
                    f"ploss: {self.stats_out[0]:.4f} vloss: {self.stats_out[1]:.4f} "
                    f"ent: {self.stats_out[2]:.3f} kl: {self.stats_out[3]:.4f} | "
                    f"trans: {n_trans} | {iter_time*1000:.1f}ms"
                )

            if self.iteration % self.snapshot_interval == 0:
                self.snapshot_count += 1
                if len(self.opponent_pool) < self.opponent_pool_max:
                    self.opponent_pool.append(self.params.copy())
                else:
                    # Reservoir sampling: replace a random element with probability
                    # pool_max / snapshot_count
                    j = random.randint(0, self.snapshot_count - 1)
                    if j < self.opponent_pool_max:
                        self.opponent_pool[j] = self.params.copy()

            if self.iteration % eval_interval == 0:
                eval_start = time.time()
                elo_gpo = config.get("elo_games_per_opp", 20)
                eval_results = self.evaluate(elo_games_per_opp=elo_gpo)
                eval_time = time.time() - eval_start
                for key, val in eval_results.items():
                    self.metrics_history[key].append(val)
                self.metrics_history["eval_iteration"].append(self.iteration)

                # Track best checkpoint
                if "elo" in eval_results and eval_results["elo"] > self.best_elo:
                    self.best_elo = eval_results["elo"]
                    self.best_params = self.params.copy()
                    self.metrics_history["best_elo_iter"] = self.iteration

                if verbose:
                    elo_str = f" Elo: {eval_results['elo']:.0f}" if "elo" in eval_results else ""
                    log(
                        f"  EVAL @ {self.iteration}: "
                        f"vs Random W: {eval_results['vs_random_win_rate']:.2f} | "
                        f"vs Heuristic W: {eval_results['vs_heuristic_win_rate']:.2f} "
                        f"D: {eval_results['vs_heuristic_draw_rate']:.2f} "
                        f"L: {eval_results['vs_heuristic_loss_rate']:.2f} |"
                        f"{elo_str} | "
                        f"Pool: {len(self.opponent_pool)} | eval {eval_time:.1f}s"
                    )

        load_params(self.agent, self.params)
        return dict(self.metrics_history)
