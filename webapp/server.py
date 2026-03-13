"""Flask server for tic-tac-toe and connect 4 vs trained agents with neural network visualization."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import json
from flask import Flask, jsonify, request, send_from_directory

from src.model import TicTacToeNet
from src.connect4_c import Connect4Net

app = Flask(__name__, static_folder="static")

# Load tic-tac-toe model
TTT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pt")
ttt_model = TicTacToeNet(hidden_size=256, num_layers=4)
ttt_model.load_state_dict(torch.load(TTT_MODEL_PATH, map_location="cpu", weights_only=True))
ttt_model.eval()

# Load connect 4 model
C4_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "connect4_policy.pt")
c4_model = Connect4Net(hidden_size=256, num_layers=6)
c4_model.load_state_dict(torch.load(C4_MODEL_PATH, map_location="cpu", weights_only=True))
c4_model.eval()


# ---- Tic-Tac-Toe helpers ----

def ttt_get_obs(board, current_player):
    b = np.array(board, dtype=np.float32).reshape(3, 3)
    obs = np.zeros((3, 3, 3), dtype=np.float32)
    obs[:, :, 0] = (b == current_player).astype(np.float32)
    obs[:, :, 1] = (b == -current_player).astype(np.float32)
    obs[:, :, 2] = 1.0
    return obs


def ttt_get_activations(obs_tensor, valid_mask_tensor):
    activations = []
    x = obs_tensor.reshape(1, -1)
    activations.append(x.squeeze(0).tolist())
    for i, layer in enumerate(ttt_model.shared):
        x = layer(x)
        activations.append(x.squeeze(0).tolist())
    trunk_output = x
    policy_logits = ttt_model.policy_head(trunk_output)
    masked_logits = policy_logits - 1e8 * (1 - valid_mask_tensor.reshape(1, -1))
    probs = F.softmax(masked_logits, dim=-1)
    value = ttt_model.value_head(trunk_output)
    return {
        "activations": activations,
        "policy_logits": policy_logits.squeeze(0).tolist(),
        "masked_logits": masked_logits.squeeze(0).tolist(),
        "probabilities": probs.squeeze(0).tolist(),
        "value": value.item(),
    }


def ttt_check_winner(board):
    lines = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6),
    ]
    for a, b, c in lines:
        vals = [board[a], board[b], board[c]]
        if vals[0] == vals[1] == vals[2] != 0:
            return vals[0]
    return 0


# ---- Connect 4 helpers ----

C4_ROWS = 6
C4_COLS = 7


def c4_get_obs(board, current_player):
    """board is flat list of 42 values (row-major, row 0 = bottom)."""
    b = np.array(board, dtype=np.float32).reshape(C4_ROWS, C4_COLS)
    obs = np.zeros((C4_ROWS, C4_COLS, 3), dtype=np.float32)
    obs[:, :, 0] = (b == current_player).astype(np.float32)
    obs[:, :, 1] = (b == -current_player).astype(np.float32)
    obs[:, :, 2] = 1.0
    return obs


def c4_get_valid(board):
    """Return valid moves mask (7 columns). A column is valid if top row is empty."""
    b = np.array(board).reshape(C4_ROWS, C4_COLS)
    return np.array([1.0 if b[C4_ROWS - 1, c] == 0 else 0.0 for c in range(C4_COLS)], dtype=np.float32)


def c4_drop_piece(board, col, player):
    """Drop piece into column. Returns row where it landed, or -1 if full."""
    b = board  # flat list
    for r in range(C4_ROWS):
        idx = r * C4_COLS + col
        if b[idx] == 0:
            b[idx] = player
            return r
    return -1


def c4_check_winner(board):
    """Check for 4-in-a-row. Returns winning player or 0."""
    b = np.array(board).reshape(C4_ROWS, C4_COLS)
    # Horizontal
    for r in range(C4_ROWS):
        for c in range(C4_COLS - 3):
            s = b[r, c] + b[r, c+1] + b[r, c+2] + b[r, c+3]
            if s == 4: return 1
            if s == -4: return -1
    # Vertical
    for r in range(C4_ROWS - 3):
        for c in range(C4_COLS):
            s = b[r, c] + b[r+1, c] + b[r+2, c] + b[r+3, c]
            if s == 4: return 1
            if s == -4: return -1
    # Diagonal up-right
    for r in range(C4_ROWS - 3):
        for c in range(C4_COLS - 3):
            s = b[r, c] + b[r+1, c+1] + b[r+2, c+2] + b[r+3, c+3]
            if s == 4: return 1
            if s == -4: return -1
    # Diagonal up-left
    for r in range(C4_ROWS - 3):
        for c in range(3, C4_COLS):
            s = b[r, c] + b[r+1, c-1] + b[r+2, c-2] + b[r+3, c-3]
            if s == 4: return 1
            if s == -4: return -1
    return 0


def c4_get_activations(obs_tensor, valid_mask_tensor):
    activations = []
    x = obs_tensor.reshape(1, -1)
    activations.append(x.squeeze(0).tolist())
    for i, layer in enumerate(c4_model.trunk):
        x = layer(x)
        activations.append(x.squeeze(0).tolist())
    trunk_output = x
    policy_logits = c4_model.policy_head(trunk_output)
    masked_logits = policy_logits - 1e8 * (1 - valid_mask_tensor.reshape(1, -1))
    probs = F.softmax(masked_logits, dim=-1)
    value = c4_model.value_head(trunk_output)
    return {
        "activations": activations,
        "policy_logits": policy_logits.squeeze(0).tolist(),
        "masked_logits": masked_logits.squeeze(0).tolist(),
        "probabilities": probs.squeeze(0).tolist(),
        "value": value.item(),
    }


# ---- Routes ----

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/move", methods=["POST"])
def make_move():
    """Handle move for tic-tac-toe."""
    data = request.json
    board = data["board"]
    human_action = data["action"]
    human_player = data.get("human_player", 1)
    ai_player = -human_player

    if human_action == -1:
        obs = ttt_get_obs(board, ai_player)
        obs_t = torch.FloatTensor(obs)
        valid = np.array([1.0 if v == 0 else 0.0 for v in board], dtype=np.float32)
        mask_t = torch.FloatTensor(valid)
        with torch.no_grad():
            act_data = ttt_get_activations(obs_t, mask_t)
            ai_action = int(np.argmax(act_data["probabilities"]))
        board[ai_action] = ai_player
        return jsonify({
            "board": board, "winner": 0, "done": False,
            "ai_action": ai_action, "activations": act_data,
        })

    if board[human_action] != 0:
        return jsonify({"error": "Invalid move"}), 400
    board[human_action] = human_player

    winner = ttt_check_winner(board)
    if winner != 0 or all(v != 0 for v in board):
        obs = ttt_get_obs(board, ai_player)
        obs_t = torch.FloatTensor(obs)
        valid = np.array([1.0 if v == 0 else 0.0 for v in board], dtype=np.float32)
        mask_t = torch.FloatTensor(valid)
        with torch.no_grad():
            act_data = ttt_get_activations(obs_t, mask_t)
        return jsonify({
            "board": board, "winner": int(winner), "done": True,
            "ai_action": None, "activations": act_data,
        })

    obs = ttt_get_obs(board, ai_player)
    obs_t = torch.FloatTensor(obs)
    valid = np.array([1.0 if v == 0 else 0.0 for v in board], dtype=np.float32)
    mask_t = torch.FloatTensor(valid)
    with torch.no_grad():
        act_data = ttt_get_activations(obs_t, mask_t)
        ai_action = int(np.argmax(act_data["probabilities"]))
    board[ai_action] = ai_player
    winner = ttt_check_winner(board)
    done = winner != 0 or all(v != 0 for v in board)
    return jsonify({
        "board": board, "winner": int(winner), "done": done,
        "ai_action": ai_action, "activations": act_data,
    })


@app.route("/api/activations", methods=["POST"])
def get_board_activations():
    data = request.json
    board = data["board"]
    perspective_player = data.get("perspective_player", -1)
    obs = ttt_get_obs(board, perspective_player)
    obs_t = torch.FloatTensor(obs)
    valid = np.array([1.0 if v == 0 else 0.0 for v in board], dtype=np.float32)
    mask_t = torch.FloatTensor(valid)
    with torch.no_grad():
        act_data = ttt_get_activations(obs_t, mask_t)
    return jsonify({"activations": act_data})


@app.route("/api/c4/move", methods=["POST"])
def c4_make_move():
    """Handle move for connect 4."""
    data = request.json
    board = data["board"]  # flat list of 42 values (row-major, row 0 = bottom)
    human_action = data["action"]  # column 0-6, or -1 for AI-first
    human_player = data.get("human_player", 1)
    ai_player = -human_player

    if human_action == -1:
        obs = c4_get_obs(board, ai_player)
        obs_t = torch.FloatTensor(obs)
        valid = c4_get_valid(board)
        mask_t = torch.FloatTensor(valid)
        with torch.no_grad():
            act_data = c4_get_activations(obs_t, mask_t)
            ai_action = int(np.argmax(act_data["probabilities"]))
        c4_drop_piece(board, ai_action, ai_player)
        return jsonify({
            "board": board, "winner": 0, "done": False,
            "ai_action": ai_action, "activations": act_data,
        })

    # Apply human move
    valid = c4_get_valid(board)
    if valid[human_action] < 0.5:
        return jsonify({"error": "Column full"}), 400
    c4_drop_piece(board, human_action, human_player)

    winner = c4_check_winner(board)
    board_full = all(v != 0 for v in board)
    if winner != 0 or board_full:
        obs = c4_get_obs(board, ai_player)
        obs_t = torch.FloatTensor(obs)
        valid = c4_get_valid(board)
        mask_t = torch.FloatTensor(valid)
        with torch.no_grad():
            act_data = c4_get_activations(obs_t, mask_t)
        return jsonify({
            "board": board, "winner": int(winner), "done": True,
            "ai_action": None, "activations": act_data,
        })

    # AI's turn
    obs = c4_get_obs(board, ai_player)
    obs_t = torch.FloatTensor(obs)
    valid = c4_get_valid(board)
    mask_t = torch.FloatTensor(valid)
    with torch.no_grad():
        act_data = c4_get_activations(obs_t, mask_t)
        ai_action = int(np.argmax(act_data["probabilities"]))
    c4_drop_piece(board, ai_action, ai_player)
    winner = c4_check_winner(board)
    done = winner != 0 or all(v != 0 for v in board)
    return jsonify({
        "board": board, "winner": int(winner), "done": done,
        "ai_action": ai_action, "activations": act_data,
    })


@app.route("/api/c4/activations", methods=["POST"])
def c4_get_board_activations():
    data = request.json
    board = data["board"]
    perspective_player = data.get("perspective_player", -1)
    obs = c4_get_obs(board, perspective_player)
    obs_t = torch.FloatTensor(obs)
    valid = c4_get_valid(board)
    mask_t = torch.FloatTensor(valid)
    with torch.no_grad():
        act_data = c4_get_activations(obs_t, mask_t)
    return jsonify({"activations": act_data})


if __name__ == "__main__":
    print("Starting game server on http://localhost:5001", flush=True)
    app.run(host="0.0.0.0", port=5001, debug=False)
