"""Neural network for tic-tac-toe PPO agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class TicTacToeNet(nn.Module):
    """Actor-critic network for tic-tac-toe.

    Input: 3x3x3 observation tensor (my pieces, opp pieces, bias)
    Output: policy logits (9,) and value scalar
    """

    def __init__(self, hidden_size=128, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size

        # Shared trunk
        layers = [nn.Linear(27, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.shared = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Linear(hidden_size, 9)

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs, valid_moves_mask=None):
        """
        obs: (batch, 3, 3, 3) or (3, 3, 3)
        valid_moves_mask: (batch, 9) or (9,) — 1 for valid, 0 for invalid
        Returns: policy_logits (batch, 9), value (batch,)
        """
        single = obs.dim() == 3
        if single:
            obs = obs.unsqueeze(0)
            if valid_moves_mask is not None:
                valid_moves_mask = valid_moves_mask.unsqueeze(0)

        x = obs.reshape(obs.shape[0], -1)  # flatten to (batch, 27)
        x = self.shared(x)

        logits = self.policy_head(x)

        # Mask invalid moves
        if valid_moves_mask is not None:
            logits = logits - 1e8 * (1 - valid_moves_mask)

        value = self.value_head(x).squeeze(-1)

        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value

    def get_action(self, obs, valid_moves_mask, deterministic=False):
        """Sample an action from the policy. Returns action, log_prob, value."""
        with torch.no_grad():
            logits, value = self.forward(obs, valid_moves_mask)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs)
            else:
                action = torch.multinomial(probs.unsqueeze(0) if probs.dim() == 1 else probs, 1).squeeze(-1)
                if action.dim() > 0:
                    action = action.squeeze(0)

            log_prob = torch.log(probs[action] + 1e-8)

        return action.item(), log_prob.item(), value.item()

    def get_policy_fn(self, device="cpu", deterministic=True):
        """Return a function compatible with play_vs_optimal/play_vs_random."""
        self.eval()

        def policy_fn(obs, valid_mask):
            obs_t = torch.FloatTensor(obs).to(device)
            mask_t = torch.FloatTensor(valid_mask).to(device)
            action, _, _ = self.get_action(obs_t, mask_t, deterministic=deterministic)
            return action

        return policy_fn

    def save_snapshot(self):
        """Return a deep copy of the model state for opponent pool."""
        return copy.deepcopy(self.state_dict())

    @classmethod
    def from_snapshot(cls, state_dict, **kwargs):
        """Create a model from a saved snapshot."""
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        model.eval()
        return model
