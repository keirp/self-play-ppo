"""Fast Elo computation against a fixed reference pool.

The reference pool consists of 16 training snapshots (every 200 iters from
the v1 baseline run) plus random and heuristic opponents. Their Elo ratings
are pre-computed from a full round-robin tournament.

Usage:
    from src.elo import compute_elo
    elo = compute_elo(params, hidden_size=256, num_layers=6)
"""

import numpy as np
import os
import json
import torch

from src.connect4_c import Connect4Net, load_params, INPUT_DIM, POLICY_DIM
from src.connect4 import Connect4, _heuristic_move


class LegacyConnect4Net(torch.nn.Module):
    """Old plain-MLP architecture for loading reference pool snapshots."""

    def __init__(self, hidden_size=512, num_layers=4):
        super().__init__()
        layers = []
        in_dim = INPUT_DIM
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(in_dim, hidden_size))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_size
        self.trunk = torch.nn.Sequential(*layers)
        self.policy_head = torch.nn.Linear(hidden_size, POLICY_DIM)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, obs, valid_moves_mask=None):
        if obs.dim() == 3:
            obs = obs.reshape(obs.shape[0], -1)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.trunk(obs)
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

# Path to reference pool data
_POOL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "elo_reference_pool")


def _build_reference_pool():
    """Load reference pool from disk. Called once on first use."""
    pool_path = os.path.join(_POOL_DIR, "pool.npz")
    meta_path = os.path.join(_POOL_DIR, "pool_meta.json")

    if not os.path.exists(pool_path):
        raise FileNotFoundError(
            f"Reference pool not found at {pool_path}. "
            "Run `python -m src.elo --build` to create it."
        )

    data = np.load(pool_path)
    with open(meta_path) as f:
        meta = json.load(f)

    pool = []
    for entry in meta["players"]:
        name = entry["name"]
        elo = entry["elo"]
        if name == "random":
            pool.append({"name": name, "elo": elo, "type": "random"})
        elif name == "heuristic":
            pool.append({"name": name, "elo": elo, "type": "heuristic"})
        else:
            params = data[name]
            pool.append({"name": name, "elo": elo, "type": "snapshot", "params": params})

    return pool, meta


# Lazy-loaded singleton
_pool_cache = None


def _get_pool():
    global _pool_cache
    if _pool_cache is None:
        _pool_cache = _build_reference_pool()
    return _pool_cache


def _play_games(policy_a, policy_b_or_type, env_cls, num_games):
    """Play games between policy_a and policy_b. Returns (wins_a, draws, losses_a)."""
    wa, da, la = 0, 0, 0
    for g in range(num_games):
        env = env_cls()
        obs = env.reset()
        a_plays_as = 1 if g < num_games // 2 else -1

        while not env.done:
            valid = env.get_valid_moves()
            if env.current_player == a_plays_as:
                action = policy_a(obs, valid)
            else:
                if policy_b_or_type == "random":
                    action = np.random.choice(np.where(valid > 0)[0])
                elif policy_b_or_type == "heuristic":
                    action = _heuristic_move(env)
                else:
                    action = policy_b_or_type(obs, valid)
            obs, reward, done, info = env.step(action)

        if env.winner == a_plays_as:
            wa += 1
        elif env.winner == 0:
            da += 1
        else:
            la += 1

    return wa, da, la


def _mle_elo(results, pool_elos, anchor_elo=None):
    """Compute MLE Elo for a new player given game results vs pool members.

    Uses Newton's method to maximize the log-likelihood:
        L(R) = sum_i [ w_i * log(p_i) + l_i * log(1-p_i) + d_i * log(2*p_i*(1-p_i)) ]
    where p_i = 1 / (1 + 10^((R_i - R) / 400))

    For simplicity, we treat draws as half-wins (Bradley-Terry model):
        L(R) = sum_i [ s_i * log(p_i) + (n_i - s_i) * log(1-p_i) ]
    where s_i = wins + 0.5*draws, n_i = total games
    """
    ln10_400 = np.log(10) / 400

    # Initial guess
    if not any(w + d + l > 0 for w, d, l in results):
        return 1500.0

    R = 1400.0  # start near pool center

    # Newton's method
    for _ in range(50):
        grad = 0.0
        hess = 0.0
        for i, (wi, di, li) in enumerate(results):
            ni = wi + di + li
            if ni == 0:
                continue
            si = wi + 0.5 * di
            p = 1.0 / (1.0 + 10 ** ((pool_elos[i] - R) / 400))
            grad += ln10_400 * (si - ni * p)
            hess -= ln10_400 ** 2 * ni * p * (1 - p)

        if abs(hess) < 1e-12:
            break
        step = grad / hess  # hess is negative, so this moves R toward the maximum
        step = max(-200, min(200, step))
        R -= step
        if abs(grad) < 1e-6:
            break

    return R


def compute_elo(params, hidden_size=256, num_layers=6, games_per_opponent=20,
                deterministic=False):
    """Compute Elo rating for a checkpoint against the fixed reference pool.

    Args:
        params: numpy array of model parameters (flat float32)
        hidden_size: model hidden size
        num_layers: model number of layers
        games_per_opponent: games to play against each pool member (half per side)
        deterministic: use deterministic (argmax) policy

    Returns:
        dict with 'elo', 'results' (per-opponent), and 'games_played'
    """
    pool, meta = _get_pool()

    # Build policy for the test agent
    model = Connect4Net(hidden_size=hidden_size, num_layers=num_layers)
    load_params(model, params)
    model.eval()
    policy_fn = model.get_policy_fn("cpu", deterministic=deterministic)

    # Build pool model (reused for all snapshot opponents) — uses legacy architecture
    pool_model = LegacyConnect4Net(hidden_size=meta["hidden_size"], num_layers=meta["num_layers"])

    results = []
    pool_elos = []
    per_opponent = []

    for member in pool:
        pool_elos.append(member["elo"])

        if member["type"] == "random":
            opponent = "random"
        elif member["type"] == "heuristic":
            opponent = "heuristic"
        else:
            load_params(pool_model, member["params"])
            pool_model.eval()
            opponent = pool_model.get_policy_fn("cpu", deterministic=False)

        w, d, l = _play_games(policy_fn, opponent, Connect4, games_per_opponent)
        results.append((w, d, l))
        per_opponent.append({
            "name": member["name"],
            "ref_elo": member["elo"],
            "wins": w, "draws": d, "losses": l,
        })

    elo = _mle_elo(results, pool_elos)

    return {
        "elo": elo,
        "games_played": sum(w + d + l for w, d, l in results),
        "per_opponent": per_opponent,
    }


def compute_elo_batch(params_list, hidden_size=256, num_layers=6,
                      games_per_opponent=20, deterministic=False, labels=None):
    """Compute Elo for multiple checkpoints efficiently.

    Reuses the pool model across evaluations.

    Args:
        params_list: list of numpy arrays
        labels: optional list of labels for each checkpoint
        (other args same as compute_elo)

    Returns:
        list of dicts with 'elo', 'label', etc.
    """
    pool, meta = _get_pool()

    pool_model = LegacyConnect4Net(hidden_size=meta["hidden_size"], num_layers=meta["num_layers"])
    test_model = Connect4Net(hidden_size=hidden_size, num_layers=num_layers)

    # Pre-build pool policies
    pool_policies = []
    pool_elos = []
    for member in pool:
        pool_elos.append(member["elo"])
        if member["type"] == "random":
            pool_policies.append("random")
        elif member["type"] == "heuristic":
            pool_policies.append("heuristic")
        else:
            # We'll reload params each time since get_policy_fn captures model state
            pool_policies.append(member)

    out = []
    for idx, params in enumerate(params_list):
        load_params(test_model, params)
        test_model.eval()
        policy_fn = test_model.get_policy_fn("cpu", deterministic=deterministic)

        results = []
        for i, member in enumerate(pool):
            if member["type"] in ("random", "heuristic"):
                opponent = member["type"]
            else:
                load_params(pool_model, member["params"])
                pool_model.eval()
                opponent = pool_model.get_policy_fn("cpu", deterministic=False)

            w, d, l = _play_games(policy_fn, opponent, Connect4, games_per_opponent)
            results.append((w, d, l))

        elo = _mle_elo(results, pool_elos)
        label = labels[idx] if labels else f"checkpoint_{idx}"
        out.append({"elo": elo, "label": label,
                     "games_played": sum(w + d + l for w, d, l in results)})

    return out


def build_reference_pool():
    """Build the reference pool from the v1 training snapshots."""
    snapshots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "reports", "connect4_elo", "snapshots")
    elo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "reports", "connect4_elo", "elo_data.json")

    with open(elo_path) as f:
        elo_data = json.load(f)

    ratings = elo_data["elo_ratings"]

    # Select every 200 iterations for a compact pool
    selected_iters = list(range(0, 3001, 200))
    arrays = {}
    players = []

    for it in selected_iters:
        snap_path = os.path.join(snapshots_dir, f"snapshot_{it}.npy")
        if not os.path.exists(snap_path):
            print(f"  Skipping iter {it} (no snapshot)")
            continue
        name = f"iter_{it}"
        elo = ratings.get(name, 1500.0)
        arrays[name] = np.load(snap_path)
        players.append({"name": name, "elo": elo})

    # Add random and heuristic
    players.append({"name": "random", "elo": ratings["random"]})
    players.append({"name": "heuristic", "elo": ratings["heuristic"]})

    meta = {
        "players": players,
        "hidden_size": 256,
        "num_layers": 6,
        "source": "v1 baseline training (connect4_elo.py)",
        "anchor": "random = 1000",
        "games_per_pair_in_tournament": 20,
    }

    os.makedirs(_POOL_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(_POOL_DIR, "pool.npz"), **arrays)
    with open(os.path.join(_POOL_DIR, "pool_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Reference pool saved to {_POOL_DIR}")
    print(f"  {len(players)} players ({len(arrays)} snapshots + random + heuristic)")
    print(f"  Pool file: {os.path.getsize(os.path.join(_POOL_DIR, 'pool.npz')) / 1024 / 1024:.1f}MB")
    for p in players:
        print(f"    {p['name']:>15s}: Elo {p['elo']:.0f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_reference_pool()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick test: compute Elo for a known snapshot
        import time
        pool, meta = _get_pool()
        test_params = np.load(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "reports", "connect4_elo", "snapshots", "snapshot_3000.npy"
        ))
        t0 = time.time()
        result = compute_elo(test_params, games_per_opponent=20)
        elapsed = time.time() - t0
        print(f"Elo: {result['elo']:.0f} ({result['games_played']} games in {elapsed:.1f}s)")
        print(f"Expected: ~1585 (from tournament)")
    else:
        print("Usage: python -m src.elo --build   # Build reference pool")
        print("       python -m src.elo --test    # Test with final checkpoint")
