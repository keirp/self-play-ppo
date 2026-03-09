# Connect 4 Self-Play PPO: Hyperparameter Optimization Report

## Goal
Reach **2100 stable Elo** against a fixed reference pool using self-play PPO with a pure C backend.

**Result: Peak 1970 Elo achieved (stochastic, 100 games/opp). Did not reach 2100.**

---

## System Overview

### Architecture
- **Model**: Residual MLP with LayerNorm, GELU activations, and residual connections
- **Size**: 256 hidden units, 6 layers (366K parameters)
- **Backend**: Pure C with Apple Accelerate BLAS (~1s/iter for 2048 games)
- **Training**: PPO with self-play against historical snapshots (pool of 50)

### Evaluation
- **Reference pool**: 18 fixed opponents — 16 training snapshots (iter 0–3000, every 200) + random + heuristic
- **Strongest opponents**: iter_3000 (1585 Elo), heuristic (1584 Elo)
- **Measurement**: 100 games/opponent (1800 total), stochastic play
- **Note**: Reference pool uses a different architecture (plain MLP with ReLU) than the training agent (residual MLP with LayerNorm/GELU)

---

## Experiments Summary

### Baseline Configuration
```
lr=3e-4, ent_coef=0.001, batch_size=256, hidden=256, layers=6,
pool_size=50, games_per_iter=512, clip_eps=0.2, ppo_epochs=4,
gamma=0.99, gae_lambda=0.95
```

### Experiment 1: Entropy Coefficient (2k iters)
| ent_coef | Peak Elo | Avg10 | Result |
|----------|----------|-------|--------|
| **0.001** | **1722** | **1663** | **Winner** |
| 0.01 | 1704 | 1605 | |
| 0.03 | 1600 | 1538 | |
| 0.05 | 1558 | 1529 | |

**Finding**: Lower entropy is better. Higher entropy adds too much randomness.

### Experiment 2: Draw Reward (2k iters)
| draw_reward | Peak Elo | Avg10 |
|-------------|----------|-------|
| 0.0 | 1648 | 1570 |
| 0.2 | 1726 | 1631 |
| 0.5 | 1653 | 1608 |
| 1.0 | 1602 | 1557 |

**Finding**: No significant effect. Kept default (0.0).

### Experiment 3: Games Per Iter + Model Size (2k iters)
| Config | Peak Elo | Params | Result |
|--------|----------|--------|--------|
| gpi=512 (baseline) | 1722 | 366K | |
| gpi=1024 | 1677 | 366K | |
| **gpi=2048** | **1903** | **366K** | **+181 Elo** |
| hidden=512, layers=6 | 1650 | 1.39M | |
| hidden=256, layers=8 | 1656 | 499K | |

**Finding**: `gpi=2048` was the single biggest improvement. More data per iteration = better gradient estimates. Larger models didn't help.

### Experiment 4: Mixed Opponents (3k iters, gpi=2048)
| Config | Avg Elo |
|--------|---------|
| Baseline (self-play only) | 1735 |
| 20% pinned heuristic | 1608 |
| 50% pinned heuristic | 1525 |

**Finding**: Pinning strong opponents into the training pool hurts. Self-play diversity is better.

### Experiment 5: Long Run (5k iters, gpi=2048)
- Peak: 1879, Final: 1817, Avg5: 1821
- High-precision eval (100 games/opp): **1837 true Elo**
- 100% deterministic win rate vs heuristic

### Experiments 6–8: Long Runs + Batch Size
| Config | Peak Elo | Notes |
|--------|----------|-------|
| gpi=2048, bs=512, 15k | ~1534 | bs=512 much worse (fewer gradient steps) |
| gpi=2048, bs=256, 15k | **1901** | Plateaus ~1830 after iter 2500 |

**Finding**: `bs=256` is crucial — it gives 8 gradient steps per data point (2048/256) which is the sweet spot.

### Experiment 9: Breaking the 1830 Plateau
| Config | Peak Elo | Notes |
|--------|----------|-------|
| gpi=4096, bs=1024 | 1806 | Fewer gradient steps with large batch |
| Cosine LR 3e-4→3e-5 | 1867 | Stoch 1890, Det **1908** (100g/opp) |
| 8 PPO epochs | 1477 | KL divergence too high (0.10–0.13) |

**Finding**: Cosine LR gave a small improvement. 8 PPO epochs destabilizes training.

### Key Discovery: Blind Spots (from Exp 9 high-precision eval)
- Agent achieves 95%+ vs strongest opponents but only 78–80% vs mid-range snapshots (iter 1400, 2600)
- The agent overfits to its own play style and struggles against the different MLP architecture used by pool opponents
- This architectural mismatch is a fundamental ceiling on Elo

### Experiment 10: Stability Approaches
| Config | Avg Elo | Notes |
|--------|---------|-------|
| clip=0.1, pool=200, snap_interval=5 | 1710 | Stable but stopped improving |

**Finding**: `clip_eps=0.1` is too conservative — agent stops learning. The oscillation at clip=0.2 is partly eval noise.

### Experiment 12: Mirror Augmentation
| Config | Result |
|--------|--------|
| Mirror, bs=256 | 1805 peak, oscillating |
| Mirror, bs=512 | **NaN** at iter 231 |
| Mirror, clip=0.1 | **NaN** at iter ~190 |

**Finding**: Mirror augmentation breaks PPO's importance sampling. Mirrored transitions have different log-probs than the collecting policy expects, causing ratio explosion.

### Experiment 13: gpi=4096, bs=256
- Peak: 1644 (100 games/opp)
- ~1875 gradient steps per iter vs 936 baseline — policy overshoots

**Finding**: Too many gradient steps per iteration hurts.

### Experiment 14: Long Baseline + PPO Epochs
| Config | Peak Elo | Notes |
|--------|----------|-------|
| Baseline, 20k iters | **1886** | Confirms ~1890 ceiling |
| ppo_epochs=2 | 1605 | Too slow convergence |

### Experiment 15: Training Diversity
| Config | Peak/Avg Elo | Notes |
|--------|-------------|-------|
| Ent schedule 0.01→0.001 | 1586 | High entropy hurts |
| 4 opponents per iter | 1723 stable | Lower ceiling than baseline |

### Experiment 16: Opponent Temperature (**BREAKTHROUGH**)
| Config | Peak Elo | Avg Elo | Notes |
|--------|----------|---------|-------|
| **opp_temp=1.5** | **1970** | **~1900** | **+80 Elo over baseline** |
| opp_temp=2.0 | 1760 | ~1730 | Too random |

**Finding**: Scaling opponent logits by 1/1.5 before softmax makes opponents play more randomly, creating diverse board states during self-play. This was the single biggest improvement after gpi=2048. temp=2.0 makes opponents too weak to teach useful play.

### Experiment 17: Combined Approaches
| Config | Peak Elo | Notes |
|--------|----------|-------|
| Temp anneal 1.5→1.0 | 1632 | Killed at iter 2000. Worse than constant |
| Temp=1.5 + 2 opponents | — | Did not produce results |
| Temp=1.5 + cosine LR | — | Did not produce results |

**Finding**: Temperature annealing underperformed constant temp=1.5.

### Experiment 18: Best Config Long Run (10k iters, eval every 250)
Config: `gpi=2048, bs=256, opp_temp=1.5, 100 games/opp eval`

Full Elo trajectory (100 games/opp, stochastic):
```
 250: 1546    3250: 1893    6250: 1880    9250: 1942
 500: 1627    3500: 1834    6500: 1770
 750: 1692    3750: 1833    6750: 1892
1000: 1699    4000: 1828    7000: 1778
1250: 1789    4250: 1868    7250: 1864
1500: 1862    4500: 1938    7500: 1892
1750: 1793    4750: 1931    7750: 1797
2000: 1853    5000: 1826    8000: 1943  ← Peak
2250: 1805    5250: 1805    8250: 1854
2500: 1821    5500: 1766    8500: 1830
2750: 1878    5750: 1766    8750: 1894
3000: 1799    6000: 1818    9000: 1839
```

- **Peak: 1943** at iter 8000
- **Average (last 10 evals): 1869**
- Oscillation range: 1766–1943 (±90 Elo)
- Training was killed at iter ~9450 (did not complete 10k)
- No best checkpoint was saved (run didn't finish)

---

## What Works (Ranked by Impact)

| Technique | Elo Improvement | Mechanism |
|-----------|----------------|-----------|
| **gpi=2048** | +180 Elo | Better gradient estimates from more data per iter |
| **opp_temp=1.5** | +80 Elo | Diverse board states reduce blind spots |
| **bs=256** | Essential | 8 gradient steps/data point is the sweet spot |
| **clip_eps=0.2** | Essential | Needed for learning; 0.1 too conservative |
| **100 games/opp eval** | N/A | Accurate measurement (±30 vs ±70 Elo noise) |

## What Doesn't Work

| Technique | Why It Fails |
|-----------|-------------|
| Higher entropy (0.01+) | Too much randomness in policy |
| Larger models (512×6, 256×8) | Overfitting with limited data diversity |
| gpi=4096 with any bs | Too many gradient steps, policy overshoots |
| clip_eps=0.1 | Too conservative, agent stops learning |
| 8 PPO epochs | KL divergence too high, policy diverges |
| ppo_epochs=2 | Too few updates, slow convergence |
| Mirror augmentation | Breaks importance sampling ratios → NaN |
| Pinned opponents | Reduces self-play diversity |
| Multi-opponent (4/iter) | Stable but lower ceiling |
| opp_temp=2.0 | Opponent too weak, doesn't teach useful play |
| Draw reward | No significant effect |
| Cosine LR | Marginal improvement (~20 Elo) |
| Temp annealing | Worse than constant temperature |

---

## Why 2100 Wasn't Reached

### 1. Evaluation Ceiling
The reference pool's strongest opponents are at 1585 Elo. To reach 2100, the agent needs ~95%+ win rate against them. At ~1940 Elo, win rates are already ~90%. The marginal improvement from 90%→95% requires eliminating very specific blind spots.

### 2. Architecture Mismatch
The agent (residual MLP with LayerNorm/GELU) plays against pool opponents using a different architecture (plain MLP with ReLU). The agent optimizes for self-play patterns and develops blind spots against the pool's play style, particularly mid-range opponents.

### 3. Training Oscillation
PPO with self-play inherently oscillates because:
- The opponent pool evolves as new snapshots are added
- Each training step changes both the agent and its future opponents
- With `clip_eps=0.2`, the policy can shift significantly in one update
- Conservative clipping (0.1) stops this but also stops learning

### 4. Limited Pool Diversity
18 opponents provide a narrow evaluation signal. The agent can game specific opponent weaknesses rather than learning generally strong play.

---

## Infrastructure Built

1. **Pure C PPO backend** — 13x faster than PyTorch baseline
2. **Elo evaluation system** — Fixed reference pool with per-opponent breakdown
3. **Experiment runner** — Automated training with periodic eval and best checkpoint saving
4. **Opponent temperature** — Novel technique for self-play diversity
5. **Fine-tuning script** — Direct training against reference pool opponents (written but untested)

---

## Final Numbers

| Metric | Value |
|--------|-------|
| Best peak Elo (stochastic, 100g/opp) | **1970** (exp16a, iter 3500) |
| Best sustained avg Elo | **~1870** (exp18, last 10 evals) |
| Best deterministic Elo (100g/opp) | **1908** (exp9b cosine LR) |
| Best config | gpi=2048, bs=256, opp_temp=1.5, clip=0.2, ent=0.001 |
| Architecture | 256×6 residual MLP (366K params) |
| Total experiments | 18 major experiments, ~40 individual runs |
| Training speed | ~1s/iter (2048 games, M2 Max) |

---

## Potential Next Steps (Not Attempted)

1. **Fine-tune against reference pool** — Script written (`finetune_vs_pool.py`), directly addresses blind spots
2. **Larger reference pool** — More diverse evaluation opponents
3. **Population-based training** — Multiple agents evolving together
4. **MCTS-guided policy improvement** — Use search to generate better training targets
5. **Architecture matching** — Train with the same architecture as pool opponents
