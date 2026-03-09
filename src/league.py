"""AlphaStar-style league training for Connect 4.

Implements three agent types:
  - Main Agent (MA): Trains via PFSP against all league members
  - Main Exploiter (ME): Finds weaknesses in the main agent, resets periodically
  - League Exploiter (LE): Finds weaknesses across all league members, resets periodically

Key concepts:
  - Prioritized Fictitious Self-Play (PFSP): Weight opponents by how much we lose to them
  - Payoff matrix: Track win rates between all pairs of players
  - Exploiter resets: Reset exploiters to initial weights to discover new weaknesses
"""

import numpy as np
import random
import time
import os
import json
import ctypes
from collections import defaultdict

from src.connect4_c import (
    _lib, _fp, _llp, _ip, Connect4Net, extract_params, load_params,
    INPUT_DIM, POLICY_DIM, log,
)
from src.connect4 import play_vs_random, play_vs_heuristic


class LeaguePlayer:
    """A frozen player in the league (snapshot of an agent at some point in training)."""

    def __init__(self, params, name, agent_type, iteration, parent=None):
        self.params = params.copy()
        self.name = name
        self.agent_type = agent_type  # "main", "main_exploiter", "league_exploiter"
        self.iteration = iteration
        self.parent = parent  # which active agent produced this


class ActiveAgent:
    """An actively training agent in the league."""

    def __init__(self, name, agent_type, params, total_params, config):
        self.name = name
        self.agent_type = agent_type
        self.params = params.copy()
        self.total_params = total_params
        self.adam_m = np.zeros(total_params, dtype=np.float32)
        self.adam_v = np.zeros(total_params, dtype=np.float32)
        self.adam_t = np.array([0], dtype=np.int32)
        self.config = config
        self.iteration = 0
        self.initial_params = params.copy()  # for reset
        self.stats_out = np.zeros(4, dtype=np.float32)

        # Pre-allocate buffers
        max_games = config.get("games_per_iter", 2048)
        max_trans = max_games * 22
        self.buf_obs = np.zeros((max_trans, INPUT_DIM), dtype=np.float32)
        self.buf_actions = np.zeros(max_trans, dtype=np.int64)
        self.buf_log_probs = np.zeros(max_trans, dtype=np.float32)
        self.buf_values = np.zeros(max_trans, dtype=np.float32)
        self.buf_valid_masks = np.zeros((max_trans, POLICY_DIM), dtype=np.float32)
        self.buf_rewards = np.zeros(max_trans, dtype=np.float32)
        self.buf_dones = np.zeros(max_trans, dtype=np.float32)
        self.game_results = np.zeros(3, dtype=np.int32)

    def reset(self):
        """Reset agent to initial parameters (like SL reset in AlphaStar)."""
        self.params = self.initial_params.copy()
        self.adam_m[:] = 0
        self.adam_v[:] = 0
        self.adam_t[0] = 0


class PayoffMatrix:
    """Track win rates between all league members + active agents."""

    def __init__(self):
        # player_name -> {opponent_name -> [wins, games]}
        self.records = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    def update(self, player_name, opponent_name, wins, draws, losses):
        """Update records. Draws count as 0.5 wins."""
        total = wins + draws + losses
        effective_wins = wins + 0.5 * draws
        self.records[player_name][opponent_name][0] += effective_wins
        self.records[player_name][opponent_name][1] += total

    def win_rate(self, player_name, opponent_name):
        """Get win rate of player against opponent. Returns 0.5 if no games played."""
        rec = self.records[player_name][opponent_name]
        if rec[1] == 0:
            return 0.5
        return rec[0] / rec[1]

    def win_rates_against(self, player_name, opponent_names):
        """Get win rates of player against a list of opponents."""
        return [self.win_rate(player_name, opp) for opp in opponent_names]


def pfsp_sample(win_rates, weighting="hard", p=2.0):
    """Prioritized Fictitious Self-Play opponent selection.

    Args:
        win_rates: list of win rates against each candidate opponent
        weighting: "hard" uses f(x) = (1-x)^p, "var" uses f(x) = x*(1-x)
        p: exponent for hard weighting (higher = more focused on losses)

    Returns:
        index of selected opponent
    """
    win_rates = np.array(win_rates, dtype=np.float64)

    if weighting == "hard":
        # f_hard: focus on opponents we lose to
        weights = np.power(np.maximum(1.0 - win_rates, 1e-6), p)
    elif weighting == "var":
        # f_var: focus on competitive matchups (near 50% win rate)
        weights = win_rates * (1.0 - win_rates) + 1e-6
    else:
        weights = np.ones_like(win_rates)

    total = weights.sum()
    if total < 1e-12:
        return random.randint(0, len(win_rates) - 1)

    probs = weights / total
    return np.random.choice(len(win_rates), p=probs)


class LeagueTrainer:
    """AlphaStar-style league training system for Connect 4."""

    def __init__(self, config):
        self.config = config
        hidden_size = config.get("hidden_size", 256)
        num_layers = config.get("num_layers", 6)

        # Initialize C backend
        _lib.c4_init(hidden_size, num_layers)
        total_params = _lib.c4_total_params()
        self.total_params = total_params

        # Create initial random parameters
        model = Connect4Net(hidden_size=hidden_size, num_layers=num_layers)
        init_params = extract_params(model)

        # League: frozen players
        self.league = []

        # Active agents
        self.main_agent = ActiveAgent(
            "main", "main", init_params, total_params, config
        )

        # Main exploiter: starts from same init, targets main agent weaknesses
        self.main_exploiter = ActiveAgent(
            "main_exploiter", "main_exploiter", init_params, total_params, config
        )

        # League exploiter: targets weaknesses across all league members
        self.league_exploiter = ActiveAgent(
            "league_exploiter", "league_exploiter", init_params, total_params, config
        )

        self.active_agents = [self.main_agent, self.main_exploiter, self.league_exploiter]

        # Payoff tracking
        self.payoff = PayoffMatrix()

        # Add initial snapshots to league
        self._add_to_league(self.main_agent, iteration=0)

        # PyTorch model for evaluation
        self.eval_model = model

        # Metrics
        self.metrics = defaultdict(list)

        # Config
        self.snapshot_interval = config.get("snapshot_interval", 50)
        self.exploiter_reset_threshold = config.get("exploiter_reset_threshold", 0.7)
        self.exploiter_reset_timeout = config.get("exploiter_reset_timeout", 500)
        self.pfsp_p = config.get("pfsp_p", 2.0)
        self.self_play_frac = config.get("self_play_frac", 0.35)

    def _add_to_league(self, agent, iteration=None):
        """Add a frozen snapshot of an agent to the league."""
        it = iteration if iteration is not None else agent.iteration
        name = f"{agent.name}_iter{it}"
        player = LeaguePlayer(
            agent.params, name, agent.agent_type, it, parent=agent.name
        )
        self.league.append(player)
        log(f"  [LEAGUE] Added {name} (total: {len(self.league)} players)")
        return player

    def _select_opponent_main(self):
        """Main Agent opponent selection:
        - 35% self-play
        - 15% vs live exploiters (adversarial pressure)
        - 50% PFSP over all league members
        """
        r = random.random()
        if r < self.self_play_frac:
            # Self-play: use own current params
            return self.main_agent.params, "self"
        elif r < self.self_play_frac + 0.15:
            # Play against a live exploiter
            exploiter = random.choice([self.main_exploiter, self.league_exploiter])
            return exploiter.params, f"live_{exploiter.name}"

        # PFSP over all league members
        if not self.league:
            return self.main_agent.params, "self"

        names = [p.name for p in self.league]
        wr = self.payoff.win_rates_against(self.main_agent.name, names)
        idx = pfsp_sample(wr, weighting="hard", p=self.pfsp_p)
        return self.league[idx].params, self.league[idx].name

    def _select_opponent_main_exploiter(self):
        """Main Exploiter: play against current main agent (or PFSP curriculum if too weak)."""
        # Check win rate against main agent
        wr_vs_main = self.payoff.win_rate(
            self.main_exploiter.name, self.main_agent.name
        )

        if wr_vs_main > 0.2:
            # Strong enough: play directly against current main
            return self.main_agent.params, self.main_agent.name
        else:
            # Too weak: use PFSP with f_var over main agent's historical snapshots
            main_snapshots = [p for p in self.league if p.parent == "main"]
            if not main_snapshots:
                return self.main_agent.params, self.main_agent.name

            names = [p.name for p in main_snapshots]
            wr = self.payoff.win_rates_against(self.main_exploiter.name, names)
            idx = pfsp_sample(wr, weighting="var")
            return main_snapshots[idx].params, main_snapshots[idx].name

    def _select_opponent_league_exploiter(self):
        """League Exploiter: PFSP over ALL league members."""
        if not self.league:
            return self.main_agent.params, "self"

        names = [p.name for p in self.league]
        wr = self.payoff.win_rates_against(self.league_exploiter.name, names)
        idx = pfsp_sample(wr, weighting="hard", p=self.pfsp_p)
        return self.league[idx].params, self.league[idx].name

    def _select_opponent(self, agent):
        """Select opponent based on agent type."""
        if agent.agent_type == "main":
            return self._select_opponent_main()
        elif agent.agent_type == "main_exploiter":
            return self._select_opponent_main_exploiter()
        elif agent.agent_type == "league_exploiter":
            return self._select_opponent_league_exploiter()
        else:
            raise ValueError(f"Unknown agent type: {agent.agent_type}")

    def _train_step(self, agent):
        """Run one training iteration for an agent."""
        import math
        config = self.config
        games_per_iter = config.get("games_per_iter", 2048)
        opp_temp = config.get("opp_temperature", 1.0)

        # Select opponent
        opp_params, opp_name = self._select_opponent(agent)

        # Collect games
        _lib.c4_seed(ctypes.c_ulong(random.getrandbits(64)))
        agent.game_results[:] = 0

        n_trans = _lib.c4_collect_games(
            _fp(agent.params), _fp(opp_params),
            games_per_iter, config.get("draw_reward", 0.0),
            opp_temp,
            _fp(agent.buf_obs), _llp(agent.buf_actions),
            _fp(agent.buf_log_probs),
            _fp(agent.buf_values), _fp(agent.buf_valid_masks),
            _fp(agent.buf_rewards), _fp(agent.buf_dones),
            _ip(agent.game_results),
        )

        # Update payoff matrix
        gr = agent.game_results
        self.payoff.update(agent.name, opp_name, int(gr[0]), int(gr[1]), int(gr[2]))

        # PPO update
        lr = config.get("lr", 3e-4)
        _lib.c4_ppo_update(
            _fp(agent.params), _fp(agent.adam_m), _fp(agent.adam_v), _ip(agent.adam_t),
            _fp(agent.buf_obs), _llp(agent.buf_actions), _fp(agent.buf_log_probs),
            _fp(agent.buf_values), _fp(agent.buf_valid_masks),
            _fp(agent.buf_rewards), _fp(agent.buf_dones), n_trans,
            config.get("gamma", 0.99), config.get("gae_lambda", 0.95),
            config.get("clip_eps", 0.2), config.get("vf_coef", 0.5),
            config.get("ent_coef", 0.001), lr,
            config.get("max_grad_norm", 0.5), config.get("ppo_epochs", 4),
            config.get("batch_size", 256), _fp(agent.stats_out),
        )

        agent.iteration += 1
        return {
            "n_trans": n_trans,
            "wins": int(gr[0]),
            "draws": int(gr[1]),
            "losses": int(gr[2]),
            "opp_name": opp_name,
            "ploss": float(agent.stats_out[0]),
            "vloss": float(agent.stats_out[1]),
            "ent": float(agent.stats_out[2]),
            "kl": float(agent.stats_out[3]),
        }

    def _check_exploiter_snapshot_and_reset(self, agent):
        """Check if exploiter should snapshot to league and reset."""
        if agent.agent_type == "main":
            return  # Main agent never resets

        # Check win rate thresholds
        if agent.agent_type == "main_exploiter":
            wr = self.payoff.win_rate(agent.name, self.main_agent.name)
            should_snapshot = wr > self.exploiter_reset_threshold
        elif agent.agent_type == "league_exploiter":
            # Check average win rate against all league members
            if self.league:
                names = [p.name for p in self.league]
                wrs = self.payoff.win_rates_against(agent.name, names)
                avg_wr = np.mean(wrs)
                should_snapshot = avg_wr > self.exploiter_reset_threshold
            else:
                should_snapshot = False
        else:
            should_snapshot = False

        # Timeout: force snapshot + reset after N iterations without one
        timeout = agent.iteration > 0 and agent.iteration % self.exploiter_reset_timeout == 0

        if should_snapshot or timeout:
            self._add_to_league(agent)

            if agent.agent_type == "main_exploiter":
                # Always reset
                agent.reset()
                log(f"  [RESET] {agent.name} reset to initial params")
            elif agent.agent_type == "league_exploiter":
                # 25% chance of reset
                if random.random() < 0.25:
                    agent.reset()
                    log(f"  [RESET] {agent.name} reset to initial params")

    def evaluate_agent(self, agent, elo_games_per_opp=100):
        """Evaluate an agent against the reference pool."""
        from src.elo import compute_elo as _compute_elo
        result = _compute_elo(
            agent.params,
            hidden_size=self.config.get("hidden_size", 256),
            num_layers=self.config.get("num_layers", 6),
            games_per_opponent=elo_games_per_opp,
        )
        return result

    def train(self, num_iterations, eval_interval=250, out_dir="experiments/league"):
        """Run league training."""
        os.makedirs(out_dir, exist_ok=True)
        config = self.config
        elo_gpo = config.get("elo_games_per_opp", 100)

        best_elo = -float("inf")
        best_params = None

        log(f"League training: {num_iterations} iters, "
            f"{config.get('games_per_iter', 2048)} games/iter, "
            f"3 agents (MA + ME + LE)")
        log(f"PFSP p={self.pfsp_p}, self_play_frac={self.self_play_frac}, "
            f"snapshot_interval={self.snapshot_interval}, "
            f"exploiter_reset_timeout={self.exploiter_reset_timeout}")

        for i in range(num_iterations):
            t0 = time.time()

            # Train all three agents (one iteration each)
            results = {}
            for agent in self.active_agents:
                r = self._train_step(agent)
                results[agent.name] = r

            elapsed = time.time() - t0

            # Snapshot main agent periodically
            if (i + 1) % self.snapshot_interval == 0:
                self._add_to_league(self.main_agent)

            # Check exploiter snapshot + reset
            for agent in [self.main_exploiter, self.league_exploiter]:
                self._check_exploiter_snapshot_and_reset(agent)

            # Logging
            ma_r = results["main"]
            me_r = results["main_exploiter"]
            le_r = results["league_exploiter"]

            self.metrics["iteration"].append(i + 1)
            self.metrics["ma_wr"].append(
                ma_r["wins"] / max(ma_r["wins"] + ma_r["draws"] + ma_r["losses"], 1))
            self.metrics["me_wr"].append(
                me_r["wins"] / max(me_r["wins"] + me_r["draws"] + me_r["losses"], 1))
            self.metrics["le_wr"].append(
                le_r["wins"] / max(le_r["wins"] + le_r["draws"] + le_r["losses"], 1))
            self.metrics["league_size"].append(len(self.league))

            if (i + 1) % 50 == 0:
                log(f"[iter {i+1:5d}/{num_iterations}] "
                    f"MA: W/D/L {ma_r['wins']}/{ma_r['draws']}/{ma_r['losses']} "
                    f"vs {ma_r['opp_name'][:20]:>20s} | "
                    f"ME: {me_r['wins']}/{me_r['draws']}/{me_r['losses']} | "
                    f"LE: {le_r['wins']}/{le_r['draws']}/{le_r['losses']} | "
                    f"league: {len(self.league)} | {elapsed:.1f}s")

            # Evaluate main agent
            if (i + 1) % eval_interval == 0:
                eval_t0 = time.time()
                result = self.evaluate_agent(self.main_agent, elo_games_per_opp=elo_gpo)
                eval_time = time.time() - eval_t0

                elo = result["elo"]
                self.metrics["eval_iteration"].append(i + 1)
                self.metrics["elo"].append(elo)

                if elo > best_elo:
                    best_elo = elo
                    best_params = self.main_agent.params.copy()
                    np.save(os.path.join(out_dir, "best_params.npy"), best_params)

                # Also eval exploiters for comparison
                me_result = self.evaluate_agent(self.main_exploiter, elo_games_per_opp=elo_gpo)
                le_result = self.evaluate_agent(self.league_exploiter, elo_games_per_opp=elo_gpo)
                self.metrics["me_elo"].append(me_result["elo"])
                self.metrics["le_elo"].append(le_result["elo"])

                log(f"  EVAL @ {i+1}: MA Elo={elo:.0f} | ME Elo={me_result['elo']:.0f} | "
                    f"LE Elo={le_result['elo']:.0f} | best={best_elo:.0f} | "
                    f"league={len(self.league)} | {eval_time:.1f}s")

        # Save final results
        np.save(os.path.join(out_dir, "main_final_params.npy"), self.main_agent.params)
        if best_params is not None:
            np.save(os.path.join(out_dir, "best_params.npy"), best_params)

        # Save metrics
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(dict(self.metrics), f)

        # Save payoff info
        payoff_summary = {}
        for player_name in self.payoff.records:
            payoff_summary[player_name] = {}
            for opp_name in self.payoff.records[player_name]:
                rec = self.payoff.records[player_name][opp_name]
                payoff_summary[player_name][opp_name] = {
                    "wins": rec[0], "games": rec[1],
                    "wr": rec[0] / rec[1] if rec[1] > 0 else 0.5,
                }
        with open(os.path.join(out_dir, "payoff.json"), "w") as f:
            json.dump(payoff_summary, f, indent=2)

        log(f"\nTraining complete. Best Elo: {best_elo:.0f}")
        log(f"League size: {len(self.league)} players")
        log(f"Results saved to {out_dir}")

        return {
            "best_elo": best_elo,
            "best_params": best_params,
            "metrics": dict(self.metrics),
        }
