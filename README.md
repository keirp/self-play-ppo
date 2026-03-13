# Self-Play PPO for Board Games

Training board game agents (Tic-Tac-Toe, Connect 4) from scratch using PPO self-play, with a **pure C backend** for fast training on CPU.

**Best result**: ~1870 sustained Elo (1970 peak) in Connect 4, trained in minutes on an M2 Mac. The agent beats strong heuristic opponents 100% of the time.

## Highlights

- **Pure C training backend** — forward pass, backprop, Adam optimizer, and game simulation all in C with Apple Accelerate BLAS. 13x faster than PyTorch on CPU.
- **Self-play PPO** — the agent plays against past versions of itself, with an opponent pool and temperature-based exploration.
- **Elo evaluation** — agents are rated against a fixed reference pool of 18 opponents using MLE Elo estimation.
- **Problematic starting positions** — a curriculum learning technique that replays board states where the agent was overconfident but lost.
- **Extensive hyperparameter experiments** — 19 experiments documented in `experiments.md`.

## Project Structure

### Core (start here)

| File | Description |
|------|-------------|
| `csrc/connect4_ppo.c` | **Pure C backend** — neural net, PPO, game engine, all in one file (~1800 lines) |
| `csrc/ppo_core.c` | C backend for Tic-Tac-Toe (simpler, good starting point) |
| `src/connect4_c.py` | Python wrapper for C backend — `Connect4TrainerC` class, training loop, Elo eval |
| `src/elo.py` | Elo rating computation against fixed reference pool |
| `src/connect4.py` | Pure Python Connect 4 environment |
| `src/report.py` | PDF report generation (reportlab) |

### Tic-Tac-Toe (simpler game, good for understanding)

| File | Description |
|------|-------------|
| `src/environment.py` | Tic-Tac-Toe env with cached minimax for optimal opponent |
| `src/model.py` | PyTorch actor-critic network |
| `src/ppo.py` | PPO training with self-play |
| `src/ppo_fast.py` | Optimized PyTorch version (vectorized env) |
| `src/ppo_c.py` | ctypes wrapper for C backend |

### Experiments & Reports

| File | Description |
|------|-------------|
| `experiments.md` | **Log of all 19 experiments** with results and insights |
| `run_problematic_starts.py` | Sweep runner for problematic starting positions experiment |
| `generate_problematic_report.py` | PDF report generator for problematic starts |
| `generate_c4_report.py` | PDF report for Connect 4 hyperparameter experiments |
| `benchmark.py` | Three-way speed benchmark: Python vs PyTorch vs C |

### Web Interface

| File | Description |
|------|-------------|
| `webapp/server.py` | Flask server for playing against the trained agent |
| `webapp/static/` | HTML/JS frontend |

## Quick Start

### Build the C backend

```bash
# macOS with Apple Accelerate
cc -O3 -ffast-math -march=native -shared -fPIC \
   -DACCELERATE_NEW_LAPACK \
   -o csrc/connect4_ppo.dylib csrc/connect4_ppo.c \
   -framework Accelerate

# Linux with OpenBLAS (untested)
cc -O3 -ffast-math -march=native -shared -fPIC \
   -o csrc/connect4_ppo.so csrc/connect4_ppo.c \
   -lopenblas
```

### Train a Connect 4 agent

```python
from src.connect4_c import Connect4TrainerC

config = {
    "hidden_size": 256,
    "num_layers": 6,
    "lr": 3e-4,
    "ent_coef": 0.001,
    "batch_size": 256,
    "clip_eps": 0.2,
    "games_per_iter": 2048,
    "opp_temperature": 1.5,
    "ppo_epochs": 4,
}

trainer = Connect4TrainerC(config)
metrics = trainer.train(num_iterations=5000, eval_interval=50, verbose=True)
```

### Run the benchmark

```bash
python benchmark.py
```

## Architecture

The neural network is a residual MLP with LayerNorm and GELU activations:

```
Input (126) → [Linear → LayerNorm → GELU → + residual] × 6 → Policy head (7) + Value head (1)
```

- **Input**: 126 dims = 42 cells × 3 planes (my pieces, opponent pieces, bias)
- **Hidden**: 256 units per layer, ~366K parameters
- **Output**: 7-dim policy (column probabilities) + scalar value estimate

The entire forward pass, backprop, and Adam optimizer are implemented in C using Apple Accelerate for matrix operations.

## Key Findings

### What works (ranked by impact)
1. **More games per iteration** (gpi=2048) — +180 Elo over gpi=512
2. **Opponent temperature** (1.5) — +80 Elo peak, creates diverse training positions
3. **Small batch size** (256) — more gradient steps per iteration
4. **Moderate clipping** (0.2) — 0.1 is too conservative, agent stops learning
5. **Problematic starting positions** — +60-90 Elo at gpi=512 by replaying lost positions

### What doesn't work
- Bigger models (diminishing returns past 256×6)
- Mirror augmentation (breaks PPO importance sampling → NaN)
- High entropy coefficient (too much randomness)
- Pinned opponents (self-play diversity is better)
- 8 PPO epochs (policy diverges from too-aggressive updates)

See `experiments.md` for the full experiment log.

## Performance

On Apple M2 Max (CPU only):

| Backend | Speed (5000 iters, gpi=512) |
|---------|---------------------------|
| Python/PyTorch | ~47 min |
| Optimized PyTorch | ~17 min |
| **Pure C + Accelerate** | **~3.5 min** |

## How Problematic Starting Positions Work

Instead of always starting games from blank boards, a fraction of training games start from board states where the agent previously lost and was overconfident (value estimate > 0.3 despite losing). This forces the agent to practice its weakest positions.

The states are stored in a circular buffer with reservoir sampling. Analysis showed 85% of states from lost games have V > 0.5 — the agent is almost always overconfident when it loses.

## License

MIT
