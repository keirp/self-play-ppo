"""Optimized PPO training with batched self-play for tic-tac-toe."""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import defaultdict

from src.environment import play_vs_optimal, play_vs_random
from src.environment_fast import VectorizedTicTacToe
from src.model import TicTacToeNet


def log(msg):
    print(msg, flush=True)


def collect_self_play_games_fast(agent, opponent, num_games, device="cpu", config=None):
    """Batched game collection — all games run simultaneously with batched inference."""
    draw_reward = config.get("draw_reward", 0.0) if config else 0.0

    env = VectorizedTicTacToe(num_games)

    # Randomly assign agent side per game
    agent_player = np.where(np.random.randint(0, 2, num_games), 1.0, -1.0).astype(np.float32)

    # Pre-allocate storage — max ~5 moves per player per game
    max_transitions_per_game = 5
    max_total = num_games * max_transitions_per_game

    all_obs = np.empty((max_total, 27), dtype=np.float32)
    all_actions = np.empty(max_total, dtype=np.int64)
    all_log_probs = np.empty(max_total, dtype=np.float32)
    all_values = np.empty(max_total, dtype=np.float32)
    all_valid_masks = np.empty((max_total, 9), dtype=np.float32)
    # Track which game each transition belongs to, and whether it's the last
    all_game_idx = np.empty(max_total, dtype=np.int64)
    trans_count = 0

    # Track last transition index per game (for assigning terminal reward)
    last_trans_idx = np.full(num_games, -1, dtype=np.int64)

    while not np.all(env.done):
        active = ~env.done
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            break

        # Get observations and valid moves for active games
        obs_batch = env.get_obs_batch()  # (N, 27)
        valid_batch = env.get_valid_moves_batch()  # (N, 9)

        # Determine which active games have agent playing vs opponent
        is_agent_turn = (env.current_player[active_idx] == agent_player[active_idx])
        agent_idx = active_idx[is_agent_turn]
        opp_idx = active_idx[~is_agent_turn]

        actions = np.zeros(num_games, dtype=np.int64)

        # Agent batch inference
        if len(agent_idx) > 0:
            obs_a = torch.from_numpy(obs_batch[agent_idx])
            mask_a = torch.from_numpy(valid_batch[agent_idx])

            with torch.inference_mode():
                logits_a, values_a = agent(obs_a, mask_a)
                probs_a = F.softmax(logits_a, dim=-1)
                sampled_a = torch.multinomial(probs_a, 1).squeeze(-1)
                log_probs_a = torch.log(probs_a.gather(1, sampled_a.unsqueeze(1)).squeeze(1) + 1e-8)

            actions_a = sampled_a.numpy()
            actions[agent_idx] = actions_a

            # Store transitions
            n_a = len(agent_idx)
            end = trans_count + n_a
            all_obs[trans_count:end] = obs_batch[agent_idx]
            all_actions[trans_count:end] = actions_a
            all_log_probs[trans_count:end] = log_probs_a.numpy()
            all_values[trans_count:end] = values_a.numpy()
            all_valid_masks[trans_count:end] = valid_batch[agent_idx]
            all_game_idx[trans_count:end] = agent_idx
            last_trans_idx[agent_idx] = np.arange(trans_count, end)
            trans_count = end

        # Opponent batch inference
        if len(opp_idx) > 0:
            obs_o = torch.from_numpy(obs_batch[opp_idx])
            mask_o = torch.from_numpy(valid_batch[opp_idx])

            with torch.inference_mode():
                logits_o, _ = opponent(obs_o, mask_o)
                probs_o = F.softmax(logits_o, dim=-1)
                sampled_o = torch.multinomial(probs_o, 1).squeeze(-1)

            actions[opp_idx] = sampled_o.numpy()

        # Step all active games
        env.step_batch(actions)

    # Build buffer
    total = trans_count
    obs_out = all_obs[:total]
    actions_out = all_actions[:total]
    log_probs_out = all_log_probs[:total]
    values_out = all_values[:total]
    valid_masks_out = all_valid_masks[:total]
    game_idx_out = all_game_idx[:total]

    # Compute rewards: +1 win, draw_reward draw, -1 loss from agent perspective
    rewards = np.zeros(total, dtype=np.float32)
    dones = np.zeros(total, dtype=np.float32)

    # Vectorized reward assignment
    agent_wins = (env.winner == agent_player)
    draws = (env.winner == 0)
    agent_losses = ~agent_wins & ~draws

    game_rewards = np.where(agent_wins, 1.0, np.where(draws, draw_reward, -1.0)).astype(np.float32)
    game_results = {
        "wins": int(agent_wins.sum()),
        "draws": int(draws.sum()),
        "losses": int(agent_losses.sum()),
    }

    valid_last = last_trans_idx >= 0
    valid_idx = last_trans_idx[valid_last]
    rewards[valid_idx] = game_rewards[valid_last]
    dones[valid_idx] = 1.0

    # Return as pre-built numpy arrays (skip RolloutBuffer overhead)
    buffer_data = {
        "obs": obs_out,
        "actions": actions_out,
        "log_probs": log_probs_out,
        "values": values_out,
        "valid_masks": valid_masks_out,
        "rewards": rewards,
        "dones": dones,
    }

    return buffer_data, game_results


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute GAE advantages and returns from numpy arrays."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1 or dones[t]:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * not_done * last_gae

    returns = advantages + values
    return returns, advantages


def ppo_update_fast(agent, optimizer, buffer_data, config, device="cpu"):
    """Optimized PPO update."""
    returns, advantages = compute_gae(
        buffer_data["rewards"], buffer_data["values"], buffer_data["dones"],
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
    )

    # Convert to tensors once (from_numpy is zero-copy on CPU)
    obs_t = torch.from_numpy(buffer_data["obs"])
    actions_t = torch.from_numpy(buffer_data["actions"])
    old_log_probs_t = torch.from_numpy(buffer_data["log_probs"])
    returns_t = torch.from_numpy(returns)
    advantages_t = torch.from_numpy(advantages)
    valid_masks_t = torch.from_numpy(buffer_data["valid_masks"])

    # Normalize advantages
    n = len(obs_t)
    if n > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    clip_eps = config.get("clip_eps", 0.2)
    vf_coef = config.get("vf_coef", 0.5)
    ent_coef = config.get("ent_coef", 0.01)
    num_epochs = config.get("ppo_epochs", 4)
    batch_size = config.get("batch_size", 64)
    max_grad_norm = config.get("max_grad_norm", 0.5)

    # Pre-extract model components to avoid repeated attribute lookups
    shared = agent.shared
    policy_head = agent.policy_head
    value_head = agent.value_head

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    num_updates = 0

    neg_inf_mask = -1e8 * (1 - valid_masks_t)  # precompute once

    for _ in range(num_epochs):
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch_obs = obs_t[idx]
            batch_actions = actions_t[idx]
            batch_old_log_probs = old_log_probs_t[idx]
            batch_returns = returns_t[idx]
            batch_advantages = advantages_t[idx]
            batch_neg_inf = neg_inf_mask[idx]
            batch_masks = valid_masks_t[idx]

            # Forward — inline, skip model.forward overhead
            x = shared(batch_obs)
            logits = policy_head(x) + batch_neg_inf
            values = value_head(x).squeeze(-1)

            # Use log_softmax (numerically stable, single fused op)
            log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            # Policy loss
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, batch_returns)

            # Entropy from log_probs (avoid recomputing probs then log again)
            probs = torch.exp(log_probs)
            masked_probs = probs * batch_masks
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(masked_probs * torch.log(masked_probs + 1e-8)).sum(dim=-1).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            with torch.inference_mode():
                approx_kl = (batch_old_log_probs - new_log_probs).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_approx_kl += approx_kl
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
        "approx_kl": total_approx_kl / max(num_updates, 1),
    }


class SelfPlayTrainerFast:
    """Optimized self-play PPO trainer."""

    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cpu")

        model_kwargs = {
            "hidden_size": config.get("hidden_size", 128),
            "num_layers": config.get("num_layers", 3),
        }
        self.model_kwargs = model_kwargs
        self.agent = TicTacToeNet(**model_kwargs).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=config.get("lr", 3e-4),
            foreach=True,
        )

        self.opponent_pool = []
        self.opponent_pool_max = config.get("opponent_pool_max", 20)
        self.snapshot_interval = config.get("snapshot_interval", 10)

        random_model = TicTacToeNet(**model_kwargs)
        self.opponent_pool.append(random_model.state_dict())

        # Keep a pre-allocated opponent model to avoid re-creation
        self._opponent_model = TicTacToeNet(**model_kwargs).to(self.device)
        self._opponent_model.eval()

        self.iteration = 0
        self.metrics_history = defaultdict(list)

    def select_opponent(self):
        """Sample opponent — reuse pre-allocated model."""
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

        self._opponent_model.load_state_dict(self.opponent_pool[idx])
        self._opponent_model.eval()
        return self._opponent_model

    def evaluate(self):
        """Evaluate current agent against various benchmarks."""
        self.agent.eval()
        policy_fn = self.agent.get_policy_fn(self.device, deterministic=True)

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

        self.agent.train()
        return results

    def train(self, num_iterations, eval_interval=10, verbose=True):
        """Main training loop."""
        games_per_iter = self.config.get("games_per_iter", 128)
        if verbose:
            log(f"Starting training: {num_iterations} iterations, {games_per_iter} games/iter, eval every {eval_interval}")

        for i in range(num_iterations):
            self.iteration += 1
            self.agent.train()
            iter_start = time.time()

            opponent = self.select_opponent()

            buffer_data, game_results = collect_self_play_games_fast(
                self.agent, opponent, games_per_iter, self.device, config=self.config
            )

            update_stats = ppo_update_fast(self.agent, self.optimizer, buffer_data, self.config, self.device)

            iter_time = time.time() - iter_start

            self.metrics_history["iteration"].append(self.iteration)
            self.metrics_history["sp_win_rate"].append(game_results["wins"] / games_per_iter)
            self.metrics_history["sp_draw_rate"].append(game_results["draws"] / games_per_iter)
            self.metrics_history["sp_loss_rate"].append(game_results["losses"] / games_per_iter)
            self.metrics_history["policy_loss"].append(update_stats["policy_loss"])
            self.metrics_history["value_loss"].append(update_stats["value_loss"])
            self.metrics_history["entropy"].append(update_stats["entropy"])
            self.metrics_history["approx_kl"].append(update_stats["approx_kl"])
            self.metrics_history["pool_size"].append(len(self.opponent_pool))

            if verbose:
                log(
                    f"[iter {self.iteration:4d}/{num_iterations}] "
                    f"SP W/D/L: {game_results['wins']:3d}/{game_results['draws']:3d}/{game_results['losses']:3d} | "
                    f"ploss: {update_stats['policy_loss']:.4f} vloss: {update_stats['value_loss']:.4f} "
                    f"ent: {update_stats['entropy']:.3f} kl: {update_stats['approx_kl']:.4f} | "
                    f"{iter_time:.2f}s"
                )

            if self.iteration % self.snapshot_interval == 0:
                snapshot = copy.deepcopy(self.agent.state_dict())
                self.opponent_pool.append(snapshot)
                if len(self.opponent_pool) > self.opponent_pool_max:
                    keep_indices = [0] + list(range(
                        len(self.opponent_pool) - self.opponent_pool_max + 1,
                        len(self.opponent_pool)
                    ))
                    self.opponent_pool = [self.opponent_pool[i] for i in keep_indices]

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
                        f"vs Optimal D: {eval_results['vs_optimal_draw_rate']:.2f} L: {eval_results['vs_optimal_loss_rate']:.2f} | "
                        f"Pool: {len(self.opponent_pool)} | eval took {eval_time:.1f}s"
                    )

        return dict(self.metrics_history)
