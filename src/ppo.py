"""PPO training with self-play for tic-tac-toe."""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import defaultdict


def log(msg):
    print(msg, flush=True)

from src.environment import TicTacToe, play_vs_optimal, play_vs_random
from src.model import TicTacToeNet


class RolloutBuffer:
    """Stores trajectories for PPO updates."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.valid_masks = []

    def add(self, obs, action, log_prob, reward, value, done, valid_mask):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.valid_masks.append(valid_mask)

    def clear(self):
        self.__init__()

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1 or self.dones[t]:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            returns[t] = advantages[t] + self.values[t]

        return returns, advantages


def collect_self_play_games(agent, opponent, num_games, device="cpu", config=None):
    """Collect games of agent vs opponent. Agent always sees itself as player 1 perspective.

    We randomly assign who goes first each game. The agent collects transitions
    from its own perspective.
    """
    buffer = RolloutBuffer()
    game_results = {"wins": 0, "draws": 0, "losses": 0}

    for _ in range(num_games):
        env = TicTacToe()
        obs = env.reset()

        # Randomly assign sides
        agent_player = random.choice([1, -1])

        # Temporary storage for this game's agent transitions
        game_transitions = []

        while not env.done:
            current_is_agent = (env.current_player == agent_player)

            if current_is_agent:
                obs_t = torch.FloatTensor(obs).to(device)
                mask = env.get_valid_moves()
                mask_t = torch.FloatTensor(mask).to(device)

                action, log_prob, value = agent.get_action(obs_t, mask_t)
                game_transitions.append({
                    "obs": obs.copy(),
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                    "valid_mask": mask.copy(),
                })
            else:
                # Opponent's turn
                obs_t = torch.FloatTensor(obs).to(device)
                mask = env.get_valid_moves()
                mask_t = torch.FloatTensor(mask).to(device)

                with torch.no_grad():
                    action, _, _ = opponent.get_action(obs_t, mask_t)

            obs, reward, done, info = env.step(action)

        # Assign rewards from agent's perspective
        draw_reward = config.get("draw_reward", 0.0) if config else 0.0
        if env.winner == agent_player:
            agent_reward = 1.0
            game_results["wins"] += 1
        elif env.winner == 0:
            agent_reward = draw_reward
            game_results["draws"] += 1
        else:
            agent_reward = -1.0
            game_results["losses"] += 1

        # Add transitions to buffer — only the last move gets the terminal reward
        for i, t in enumerate(game_transitions):
            is_last = (i == len(game_transitions) - 1)
            buffer.add(
                obs=t["obs"],
                action=t["action"],
                log_prob=t["log_prob"],
                reward=agent_reward if is_last else 0.0,
                value=t["value"],
                done=is_last,
                valid_mask=t["valid_mask"],
            )

    return buffer, game_results


def ppo_update(agent, optimizer, buffer, config, device="cpu"):
    """Perform PPO update on collected data."""
    returns, advantages = buffer.compute_returns_and_advantages(
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
    )

    # Convert to tensors
    obs_t = torch.FloatTensor(np.array(buffer.obs)).to(device)
    actions_t = torch.LongTensor(buffer.actions).to(device)
    old_log_probs_t = torch.FloatTensor(buffer.log_probs).to(device)
    returns_t = torch.FloatTensor(returns).to(device)
    advantages_t = torch.FloatTensor(advantages).to(device)
    valid_masks_t = torch.FloatTensor(np.array(buffer.valid_masks)).to(device)

    # Normalize advantages
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    clip_eps = config.get("clip_eps", 0.2)
    vf_coef = config.get("vf_coef", 0.5)
    ent_coef = config.get("ent_coef", 0.01)
    num_epochs = config.get("ppo_epochs", 4)
    batch_size = config.get("batch_size", 64)

    n = len(buffer.obs)
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_approx_kl = 0
    num_updates = 0

    for _ in range(num_epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch_obs = obs_t[idx]
            batch_actions = actions_t[idx]
            batch_old_log_probs = old_log_probs_t[idx]
            batch_returns = returns_t[idx]
            batch_advantages = advantages_t[idx]
            batch_masks = valid_masks_t[idx]

            logits, values = agent(batch_obs, batch_masks)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-8)

            # New log probs for taken actions
            new_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            # Policy loss (clipped)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, batch_returns)

            # Entropy bonus (only over valid moves)
            masked_probs = probs * batch_masks
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(masked_probs * torch.log(masked_probs + 1e-8)).sum(dim=-1).mean()

            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.get("max_grad_norm", 0.5))
            optimizer.step()

            # Tracking
            with torch.no_grad():
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


class SelfPlayTrainer:
    """Orchestrates self-play PPO training with historical opponent sampling."""

    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cpu")

        # Create agent
        model_kwargs = {
            "hidden_size": config.get("hidden_size", 128),
            "num_layers": config.get("num_layers", 3),
        }
        self.model_kwargs = model_kwargs
        self.agent = TicTacToeNet(**model_kwargs).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=config.get("lr", 3e-4),
        )

        # Opponent pool
        self.opponent_pool = []
        self.opponent_pool_max = config.get("opponent_pool_max", 20)
        self.snapshot_interval = config.get("snapshot_interval", 10)

        # Add initial random policy to pool
        random_model = TicTacToeNet(**model_kwargs)
        self.opponent_pool.append(random_model.state_dict())

        # Training state
        self.iteration = 0
        self.metrics_history = defaultdict(list)

    def select_opponent(self):
        """Sample an opponent from the pool."""
        strategy = self.config.get("opponent_sampling", "uniform")

        if strategy == "uniform":
            idx = random.randint(0, len(self.opponent_pool) - 1)
        elif strategy == "latest":
            idx = len(self.opponent_pool) - 1
        elif strategy == "weighted_recent":
            # Weight more towards recent opponents
            n = len(self.opponent_pool)
            weights = np.arange(1, n + 1, dtype=np.float64)
            weights = weights / weights.sum()
            idx = np.random.choice(n, p=weights)
        else:
            idx = random.randint(0, len(self.opponent_pool) - 1)

        state_dict = self.opponent_pool[idx]
        opponent = TicTacToeNet(**self.model_kwargs).to(self.device)
        opponent.load_state_dict(state_dict)
        opponent.eval()
        return opponent

    def evaluate(self):
        """Evaluate current agent against various benchmarks."""
        self.agent.eval()
        policy_fn = self.agent.get_policy_fn(self.device, deterministic=True)

        results = {}

        # vs random (deterministic policy, stochastic opponent)
        w1, d1, l1 = play_vs_random(policy_fn, policy_plays_as=1, num_games=50)
        w2, d2, l2 = play_vs_random(policy_fn, policy_plays_as=-1, num_games=50)
        results["vs_random_win_rate"] = (w1 + w2) / 100
        results["vs_random_draw_rate"] = (d1 + d2) / 100
        results["vs_random_loss_rate"] = (l1 + l2) / 100

        # vs optimal (both deterministic, so 1 game per side suffices)
        w1, d1, l1 = play_vs_optimal(policy_fn, policy_plays_as=1, num_games=1)
        w2, d2, l2 = play_vs_optimal(policy_fn, policy_plays_as=-1, num_games=1)
        results["vs_optimal_win_rate"] = (w1 + w2) / 2
        results["vs_optimal_draw_rate"] = (d1 + d2) / 2
        results["vs_optimal_loss_rate"] = (l1 + l2) / 2

        # Exploit rate: how often optimal beats our policy
        # Perfect policy should never lose to optimal => loss_rate = 0
        results["exploitability"] = results["vs_optimal_loss_rate"]

        self.agent.train()
        return results

    def train(self, num_iterations, eval_interval=10, verbose=True):
        """Main training loop."""
        games_per_iter = self.config.get("games_per_iter", 128)
        log(f"Starting training: {num_iterations} iterations, {games_per_iter} games/iter, eval every {eval_interval}")

        for i in range(num_iterations):
            self.iteration += 1
            self.agent.train()
            iter_start = time.time()

            # Select opponent
            opponent = self.select_opponent()

            # Collect games
            buffer, game_results = collect_self_play_games(
                self.agent, opponent, games_per_iter, self.device, config=self.config
            )

            # PPO update
            update_stats = ppo_update(self.agent, self.optimizer, buffer, self.config, self.device)

            iter_time = time.time() - iter_start

            # Record metrics
            self.metrics_history["iteration"].append(self.iteration)
            self.metrics_history["sp_win_rate"].append(game_results["wins"] / games_per_iter)
            self.metrics_history["sp_draw_rate"].append(game_results["draws"] / games_per_iter)
            self.metrics_history["sp_loss_rate"].append(game_results["losses"] / games_per_iter)
            self.metrics_history["policy_loss"].append(update_stats["policy_loss"])
            self.metrics_history["value_loss"].append(update_stats["value_loss"])
            self.metrics_history["entropy"].append(update_stats["entropy"])
            self.metrics_history["approx_kl"].append(update_stats["approx_kl"])
            self.metrics_history["pool_size"].append(len(self.opponent_pool))

            # Log every iteration
            if verbose:
                log(
                    f"[iter {self.iteration:4d}/{num_iterations}] "
                    f"SP W/D/L: {game_results['wins']:3d}/{game_results['draws']:3d}/{game_results['losses']:3d} | "
                    f"ploss: {update_stats['policy_loss']:.4f} vloss: {update_stats['value_loss']:.4f} "
                    f"ent: {update_stats['entropy']:.3f} kl: {update_stats['approx_kl']:.4f} | "
                    f"{iter_time:.2f}s"
                )

            # Snapshot to opponent pool
            if self.iteration % self.snapshot_interval == 0:
                snapshot = self.agent.save_snapshot()
                self.opponent_pool.append(snapshot)
                if len(self.opponent_pool) > self.opponent_pool_max:
                    # Keep first (random) and most recent
                    keep_indices = [0] + list(range(
                        len(self.opponent_pool) - self.opponent_pool_max + 1,
                        len(self.opponent_pool)
                    ))
                    self.opponent_pool = [self.opponent_pool[i] for i in keep_indices]

            # Evaluate
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
