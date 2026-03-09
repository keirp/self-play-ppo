"""Fine-tune a checkpoint by playing against reference pool opponents.

The main bottleneck is that self-play never creates the specific board patterns
that reference pool opponents produce. This script directly addresses that by:
1. Loading a strong checkpoint from self-play training
2. Playing games against each reference pool opponent (in Python)
3. Collecting transitions and running PPO update (in C)
4. Repeating for many iterations

This is slow (~2s per game in Python) but creates exactly the training data
needed to eliminate blind spots against the reference pool.
"""
import numpy as np
import torch
import time
import os
import json
import sys

from src.connect4_c import (
    Connect4Net, Connect4TrainerC, load_params, extract_params,
    _lib, _fp, _llp, _ip, INPUT_DIM, POLICY_DIM
)
from src.connect4 import Connect4, _heuristic_move
from src.elo import LegacyConnect4Net, compute_elo
import ctypes


def collect_games_vs_opponent(agent_net, opp_policy_fn, num_games, device="cpu"):
    """Play games in Python and collect transitions for PPO.

    Returns buffers compatible with c4_ppo_update.
    """
    all_obs = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_valid_masks = []
    all_rewards = []
    all_dones = []

    wins, draws, losses = 0, 0, 0

    for g in range(num_games):
        env = Connect4()
        obs = env.reset()
        agent_plays_as = 1 if g < num_games // 2 else -1

        game_obs, game_acts, game_lps, game_vals, game_vms = [], [], [], [], []

        while not env.done:
            valid = env.get_valid_moves()
            valid_f = valid.astype(np.float32)

            if env.current_player == agent_plays_as:
                # Agent move — record transition
                obs_flat = obs.flatten().astype(np.float32)
                obs_t = torch.from_numpy(obs_flat).unsqueeze(0)
                mask_t = torch.from_numpy(valid_f).unsqueeze(0)

                with torch.no_grad():
                    logits, value = agent_net(obs_t, mask_t)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                    log_prob = torch.log_softmax(logits, dim=-1)[0, action].item()
                    val = value.item()

                game_obs.append(obs_flat.copy())
                game_acts.append(action)
                game_lps.append(log_prob)
                game_vals.append(val)
                game_vms.append(valid_f.copy())

                obs, reward, done, info = env.step(action)
            else:
                # Opponent move
                if opp_policy_fn == "heuristic":
                    action = _heuristic_move(env)
                else:
                    action = opp_policy_fn(obs, valid)
                obs, reward, done, info = env.step(action)

        # Compute rewards
        if env.winner == agent_plays_as:
            final_reward = 1.0
            wins += 1
        elif env.winner == 0:
            final_reward = 0.0
            draws += 1
        else:
            final_reward = -1.0
            losses += 1

        n = len(game_obs)
        if n > 0:
            rewards = np.zeros(n, dtype=np.float32)
            dones = np.zeros(n, dtype=np.float32)
            rewards[-1] = final_reward
            dones[-1] = 1.0

            all_obs.extend(game_obs)
            all_actions.extend(game_acts)
            all_log_probs.extend(game_lps)
            all_values.extend(game_vals)
            all_valid_masks.extend(game_vms)
            all_rewards.extend(rewards)
            all_dones.extend(dones)

    n_trans = len(all_obs)
    return {
        "obs": np.array(all_obs, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.int64),
        "log_probs": np.array(all_log_probs, dtype=np.float32),
        "values": np.array(all_values, dtype=np.float32),
        "valid_masks": np.array(all_valid_masks, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
        "n_trans": n_trans,
        "wins": wins, "draws": draws, "losses": losses,
    }


def main():
    # Load checkpoint
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/exp18_temp15_best/best_params.npy"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "experiments/finetune_pool"
    num_finetune_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    games_per_opp = 50  # games per reference pool opponent per iter

    os.makedirs(out_dir, exist_ok=True)

    # Config matching the training setup
    config = {
        "hidden_size": 256,
        "num_layers": 6,
        "lr": 1e-4,  # Lower LR for fine-tuning
        "ent_coef": 0.001,
        "batch_size": 256,
        "ppo_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.1,  # Conservative clipping for fine-tuning
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    # Initialize C backend
    _lib.c4_init(config["hidden_size"], config["num_layers"])
    total_params = _lib.c4_total_params()

    # Load checkpoint
    params = np.load(checkpoint_path).astype(np.float32)
    assert len(params) == total_params, f"Param mismatch: {len(params)} vs {total_params}"

    adam_m = np.zeros(total_params, dtype=np.float32)
    adam_v = np.zeros(total_params, dtype=np.float32)
    adam_t = np.array([0], dtype=np.int32)
    stats_out = np.zeros(4, dtype=np.float32)

    # Create agent model
    agent_net = Connect4Net(hidden_size=config["hidden_size"], num_layers=config["num_layers"])

    # Load reference pool
    pool_path = "data/elo_reference_pool/pool.npz"
    meta_path = "data/elo_reference_pool/pool_meta.json"
    pool_data = np.load(pool_path)
    with open(meta_path) as f:
        meta = json.load(f)

    # Build pool opponent policies
    pool_model = LegacyConnect4Net(hidden_size=meta["hidden_size"], num_layers=meta["num_layers"])
    pool_opponents = []
    for entry in meta["players"]:
        name = entry["name"]
        elo = entry["elo"]
        if name == "random":
            pool_opponents.append({"name": name, "elo": elo, "type": "random"})
        elif name == "heuristic":
            pool_opponents.append({"name": name, "elo": elo, "type": "heuristic"})
        else:
            pool_opponents.append({"name": name, "elo": elo, "type": "snapshot",
                                    "params": pool_data[name]})

    # Focus on harder opponents (Elo > 1300)
    hard_opponents = [o for o in pool_opponents if o.get("elo", 0) > 1300 or o["type"] in ("random", "heuristic")]
    print(f"Fine-tuning with {len(hard_opponents)} opponents, {games_per_opp} games each")
    print(f"Checkpoint: {checkpoint_path}")

    # Initial eval
    load_params(agent_net, params)
    result = compute_elo(params, games_per_opponent=100, deterministic=False)
    print(f"Initial Elo (stochastic, 100g): {result['elo']:.0f}")
    result_d = compute_elo(params, games_per_opponent=100, deterministic=True)
    print(f"Initial Elo (deterministic, 100g): {result_d['elo']:.0f}")

    best_elo = result["elo"]
    best_params = params.copy()

    for it in range(num_finetune_iters):
        t0 = time.time()
        load_params(agent_net, params)
        agent_net.eval()

        # Collect games against all hard opponents
        all_trans = {
            "obs": [], "actions": [], "log_probs": [], "values": [],
            "valid_masks": [], "rewards": [], "dones": [],
        }
        total_w, total_d, total_l = 0, 0, 0

        for opp in hard_opponents:
            if opp["type"] == "random":
                def opp_fn(obs, vm):
                    return np.random.choice(np.where(vm > 0)[0])
            elif opp["type"] == "heuristic":
                opp_fn = "heuristic"  # handled in collect_games_vs_opponent
            else:
                load_params(pool_model, opp["params"])
                pool_model.eval()
                opp_fn = pool_model.get_policy_fn("cpu", deterministic=False)

            data = collect_games_vs_opponent(agent_net, opp_fn, games_per_opp)
            total_w += data["wins"]
            total_d += data["draws"]
            total_l += data["losses"]

            if data["n_trans"] > 0:
                for k in ["obs", "actions", "log_probs", "values", "valid_masks", "rewards", "dones"]:
                    all_trans[k].append(data[k])

        # Concatenate all transitions
        obs_buf = np.concatenate(all_trans["obs"])
        act_buf = np.concatenate(all_trans["actions"])
        lp_buf = np.concatenate(all_trans["log_probs"])
        val_buf = np.concatenate(all_trans["values"])
        vm_buf = np.concatenate(all_trans["valid_masks"])
        rew_buf = np.concatenate(all_trans["rewards"])
        done_buf = np.concatenate(all_trans["dones"])
        n_trans = len(obs_buf)

        # PPO update using C backend
        _lib.c4_ppo_update(
            _fp(params), _fp(adam_m), _fp(adam_v), _ip(adam_t),
            _fp(obs_buf), _llp(act_buf), _fp(lp_buf),
            _fp(val_buf), _fp(vm_buf),
            _fp(rew_buf), _fp(done_buf), n_trans,
            config["gamma"], config["gae_lambda"],
            config["clip_eps"], config["vf_coef"],
            config["ent_coef"], config["lr"],
            config["max_grad_norm"], config["ppo_epochs"],
            config["batch_size"], _fp(stats_out),
        )

        elapsed = time.time() - t0
        total_games = total_w + total_d + total_l
        print(f"[ft {it+1:3d}/{num_finetune_iters}] "
              f"W/D/L: {total_w}/{total_d}/{total_l} ({total_w/total_games:.1%}) | "
              f"ploss: {stats_out[0]:.4f} vloss: {stats_out[1]:.4f} "
              f"ent: {stats_out[2]:.3f} kl: {stats_out[3]:.4f} | "
              f"trans: {n_trans} | {elapsed:.1f}s", flush=True)

        # Eval every 10 iters
        if (it + 1) % 10 == 0:
            result = compute_elo(params, games_per_opponent=100, deterministic=False)
            print(f"  EVAL @ ft_{it+1}: Elo {result['elo']:.0f} (best: {best_elo:.0f})", flush=True)
            if result["elo"] > best_elo:
                best_elo = result["elo"]
                best_params = params.copy()

    # Save results
    np.save(os.path.join(out_dir, "finetuned_params.npy"), params)
    np.save(os.path.join(out_dir, "best_finetuned_params.npy"), best_params)

    # Final high-precision eval
    print("\n--- Final Evaluation ---")
    result_s = compute_elo(best_params, games_per_opponent=200, deterministic=False)
    print(f"Best stochastic (200g): {result_s['elo']:.0f}")
    result_d = compute_elo(best_params, games_per_opponent=200, deterministic=True)
    print(f"Best deterministic (200g): {result_d['elo']:.0f}")

    for opp in result_d["per_opponent"]:
        total = opp["wins"] + opp["draws"] + opp["losses"]
        wr = (opp["wins"] + 0.5 * opp["draws"]) / total
        print(f"  {opp['name']:>12s} (Elo {opp['ref_elo']:4.0f}): {wr:.1%}")


if __name__ == "__main__":
    main()
