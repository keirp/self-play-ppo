"""Tic-tac-toe environment for self-play PPO."""

import numpy as np


class TicTacToe:
    """Tic-tac-toe environment supporting two-player self-play.

    Board representation: 3x3 grid
      0 = empty, 1 = player 1 (X), -1 = player 2 (O)

    Observation: 3x3x3 tensor
      Channel 0: current player's pieces
      Channel 1: opponent's pieces
      Channel 2: all ones (bias plane, indicates it's a valid position)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.current_player = 1  # 1 or -1
        self.done = False
        self.winner = 0  # 0=ongoing/draw, 1=player1 wins, -1=player2 wins
        return self._get_obs()

    def _get_obs(self):
        """Get observation from perspective of current player."""
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obs[:, :, 0] = (self.board == self.current_player).astype(np.float32)
        obs[:, :, 1] = (self.board == -self.current_player).astype(np.float32)
        obs[:, :, 2] = 1.0  # bias plane
        return obs

    def get_valid_moves(self):
        """Return mask of valid moves (1=valid, 0=invalid)."""
        return (self.board.flatten() == 0).astype(np.float32)

    def step(self, action):
        """Play action (0-8) for current player. Returns (obs, reward, done, info)."""
        row, col = action // 3, action % 3

        if self.done:
            raise ValueError("Game is already over")
        if self.board[row, col] != 0:
            raise ValueError(f"Invalid move: position {action} is occupied")

        self.board[row, col] = self.current_player

        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0

        acting_player = self.current_player
        self.current_player *= -1

        obs = self._get_obs()
        info = {"acting_player": acting_player, "winner": self.winner}

        return obs, reward, self.done, info

    def _check_win(self, player):
        b = self.board
        for i in range(3):
            if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                return True
        if b[0, 0] == b[1, 1] == b[2, 2] == player:
            return True
        if b[0, 2] == b[1, 1] == b[2, 0] == player:
            return True
        return False

    def clone(self):
        env = TicTacToe()
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.done = self.done
        env.winner = self.winner
        return env

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        for row in self.board:
            lines.append(" ".join(symbols[int(v)] for v in row))
        return "\n".join(lines)


_minimax_cache = {}


def _check_winner_fast(board_tuple):
    """Check winner on a tuple board. Returns 1, -1, or 0."""
    lines = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # cols
        (0,4,8), (2,4,6),            # diags
    ]
    for a, b, c in lines:
        if board_tuple[a] == board_tuple[b] == board_tuple[c] != 0:
            return board_tuple[a]
    return 0


def _minimax(board_tuple, player, maximizing_player):
    """Cached minimax on tuple board. Returns (best_action, value_for_maximizing_player)."""
    key = (board_tuple, player)
    if key in _minimax_cache:
        return _minimax_cache[key]

    winner = _check_winner_fast(board_tuple)
    if winner != 0:
        val = 1.0 if winner == maximizing_player else -1.0
        _minimax_cache[key] = (None, val)
        return None, val

    empty = [i for i in range(9) if board_tuple[i] == 0]
    if not empty:
        _minimax_cache[key] = (None, 0.0)
        return None, 0.0

    is_max = (player == maximizing_player)
    best_action = empty[0]
    best_val = -2.0 if is_max else 2.0

    board_list = list(board_tuple)
    for i in empty:
        board_list[i] = player
        _, val = _minimax(tuple(board_list), -player, maximizing_player)
        board_list[i] = 0

        if is_max:
            if val > best_val:
                best_val = val
                best_action = i
        else:
            if val < best_val:
                best_val = val
                best_action = i

    _minimax_cache[key] = (best_action, best_val)
    return best_action, best_val


def get_optimal_move(board, player):
    """Minimax to find optimal move. Returns (best_action, value)."""
    board_tuple = tuple(int(x) for x in board.flatten())
    action, value = _minimax(board_tuple, player, player)
    return action, value


def play_vs_optimal(policy_fn, policy_plays_as=1, num_games=100):
    """Play policy against optimal opponent. Returns win/draw/loss counts."""
    wins, draws, losses = 0, 0, 0

    for _ in range(num_games):
        env = TicTacToe()
        obs = env.reset()

        while not env.done:
            if env.current_player == policy_plays_as:
                action = policy_fn(obs, env.get_valid_moves())
            else:
                action, _ = get_optimal_move(env.board, env.current_player)

            obs, _, done, info = env.step(action)

        if env.winner == policy_plays_as:
            wins += 1
        elif env.winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


def play_vs_random(policy_fn, policy_plays_as=1, num_games=100):
    """Play policy against random opponent. Returns win/draw/loss counts."""
    wins, draws, losses = 0, 0, 0

    for _ in range(num_games):
        env = TicTacToe()
        obs = env.reset()

        while not env.done:
            if env.current_player == policy_plays_as:
                action = policy_fn(obs, env.get_valid_moves())
            else:
                valid = env.get_valid_moves()
                valid_actions = np.where(valid > 0)[0]
                action = np.random.choice(valid_actions)

            obs, _, done, info = env.step(action)

        if env.winner == policy_plays_as:
            wins += 1
        elif env.winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses
