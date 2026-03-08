"""Connect 4 environment for self-play RL."""

import numpy as np
from collections import defaultdict


ROWS = 6
COLS = 7
WIN_LEN = 4


class Connect4:
    """Connect 4 game environment.

    Board: 6 rows x 7 columns. Row 0 is the bottom.
    Players: 1 (first/X) and -1 (second/O).
    Actions: 0-6 (column index). Pieces fall to lowest empty row.
    Observation: (6, 7, 3) tensor per player perspective:
        Channel 0: Current player's pieces
        Channel 1: Opponent's pieces
        Channel 2: Bias plane (all ones)
    """

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = 0
        # Track height of each column for O(1) drop
        self.col_height = np.zeros(COLS, dtype=np.int8)

    def reset(self):
        self.board[:] = 0
        self.current_player = 1
        self.done = False
        self.winner = 0
        self.col_height[:] = 0
        return self._get_obs()

    def _get_obs(self):
        """Get observation from current player's perspective."""
        obs = np.zeros((ROWS, COLS, 3), dtype=np.float32)
        obs[:, :, 0] = (self.board == self.current_player).astype(np.float32)
        obs[:, :, 1] = (self.board == -self.current_player).astype(np.float32)
        obs[:, :, 2] = 1.0
        return obs

    def get_valid_moves(self):
        """Return binary mask of valid columns (not full)."""
        return (self.col_height < ROWS).astype(np.float32)

    def step(self, action):
        """Drop piece in column `action`. Returns (obs, reward, done, info)."""
        assert not self.done, "Game is already over"
        col = action
        assert 0 <= col < COLS and self.col_height[col] < ROWS, f"Invalid move: col {col}"

        row = self.col_height[col]
        self.board[row, col] = self.current_player
        self.col_height[col] += 1

        if self._check_win(row, col, self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif np.all(self.col_height >= ROWS):
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0

        self.current_player *= -1
        obs = self._get_obs()
        return obs, reward, self.done, {"winner": self.winner}

    def _check_win(self, row, col, player):
        """Check if placing at (row, col) creates 4-in-a-row for player."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag, anti-diag
        for dr, dc in directions:
            count = 1
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= WIN_LEN:
                return True
        return False

    def clone(self):
        env = Connect4()
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.done = self.done
        env.winner = self.winner
        env.col_height = self.col_height.copy()
        return env

    def render(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        for r in range(ROWS - 1, -1, -1):
            lines.append(' '.join(symbols[self.board[r, c]] for c in range(COLS)))
        lines.append(' '.join(str(c) for c in range(COLS)))
        return '\n'.join(lines)


def play_vs_random(policy_fn, policy_plays_as=1, num_games=100):
    """Play policy against random opponent. Returns (wins, draws, losses)."""
    wins, draws, losses = 0, 0, 0
    for _ in range(num_games):
        env = Connect4()
        obs = env.reset()
        while not env.done:
            valid = env.get_valid_moves()
            if env.current_player == policy_plays_as:
                action = policy_fn(obs, valid)
            else:
                valid_cols = np.where(valid > 0)[0]
                action = np.random.choice(valid_cols)
            obs, reward, done, info = env.step(action)
        if env.winner == policy_plays_as:
            wins += 1
        elif env.winner == 0:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses


def play_vs_heuristic(policy_fn, policy_plays_as=1, num_games=100):
    """Play policy against a simple heuristic opponent that blocks/completes threats."""
    wins, draws, losses = 0, 0, 0
    for _ in range(num_games):
        env = Connect4()
        obs = env.reset()
        while not env.done:
            valid = env.get_valid_moves()
            if env.current_player == policy_plays_as:
                action = policy_fn(obs, valid)
            else:
                action = _heuristic_move(env)
            obs, reward, done, info = env.step(action)
        if env.winner == policy_plays_as:
            wins += 1
        elif env.winner == 0:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses


def _heuristic_move(env):
    """Simple heuristic: win if possible, block opponent win, else center-biased random."""
    player = env.current_player
    valid_cols = np.where(env.get_valid_moves() > 0)[0]

    # Check for winning move
    for col in valid_cols:
        e = env.clone()
        e.step(col)
        if e.winner == player:
            return col

    # Check for blocking move
    for col in valid_cols:
        e = env.clone()
        e.current_player *= -1  # Pretend opponent plays
        e.step(col)
        if e.winner == -player:
            return col

    # Prefer center columns
    center_order = [3, 2, 4, 1, 5, 0, 6]
    for col in center_order:
        if col in valid_cols:
            return col

    return np.random.choice(valid_cols)
