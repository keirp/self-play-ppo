"""Fast vectorized tic-tac-toe environment for batched self-play."""

import numpy as np

# Win condition indices (row, col, diag)
_WIN_LINES = np.array([
    [0,1,2], [3,4,5], [6,7,8],  # rows
    [0,3,6], [1,4,7], [2,5,8],  # cols
    [0,4,8], [2,4,6],            # diags
], dtype=np.int32)


class VectorizedTicTacToe:
    """Batch of N tic-tac-toe games running in parallel.

    Boards: (N, 9) array with 0=empty, 1=player1, -1=player2
    """

    def __init__(self, n):
        self.n = n
        self.boards = np.zeros((n, 9), dtype=np.float32)
        self.current_player = np.ones(n, dtype=np.float32)  # all start as player 1
        self.done = np.zeros(n, dtype=bool)
        self.winner = np.zeros(n, dtype=np.float32)

    def get_obs_batch(self):
        """Get observations for all games from current player's perspective.
        Returns (N, 27) flattened observation.
        """
        # Channel 0: current player's pieces (N, 9)
        mine = (self.boards == self.current_player[:, None]).astype(np.float32)
        # Channel 1: opponent's pieces (N, 9)
        opp = (self.boards == -self.current_player[:, None]).astype(np.float32)
        # Channel 2: bias (all ones) (N, 9)
        bias = np.ones((self.n, 9), dtype=np.float32)
        # Stack: (N, 27)
        return np.concatenate([mine, opp, bias], axis=1)

    def get_valid_moves_batch(self):
        """Return (N, 9) mask of valid moves."""
        return (self.boards == 0).astype(np.float32)

    def step_batch(self, actions):
        """Apply actions for all non-done games.
        actions: (N,) int array of chosen positions (0-8).
        Returns updated state. Only modifies non-done games.
        """
        active = ~self.done
        if not np.any(active):
            return

        # Place pieces
        idx = np.arange(self.n)[active]
        self.boards[idx, actions[idx]] = self.current_player[idx]

        # Check wins for active games
        for line in _WIN_LINES:
            vals = self.boards[np.ix_(idx, line)]  # (active_n, 3)
            # Player wins if all 3 positions equal to player
            for player in [1.0, -1.0]:
                wins = np.all(vals == player, axis=1)
                if np.any(wins):
                    win_idx = idx[wins]
                    newly_done = win_idx[~self.done[win_idx]]
                    self.done[newly_done] = True
                    self.winner[newly_done] = player

        # Check draws (board full, no winner)
        still_active = idx[~self.done[idx]]
        if len(still_active) > 0:
            board_full = np.all(self.boards[still_active] != 0, axis=1)
            draw_idx = still_active[board_full]
            self.done[draw_idx] = True
            # winner stays 0

        # Switch player for non-done games
        still_going = ~self.done
        self.current_player[still_going] *= -1

    def check_wins_batch(self, idx):
        """Check wins for subset of games. Returns winner array."""
        winners = np.zeros(len(idx), dtype=np.float32)
        for line in _WIN_LINES:
            vals = self.boards[np.ix_(idx, line)]
            for player in [1.0, -1.0]:
                wins = np.all(vals == player, axis=1)
                winners[wins] = player
        return winners
