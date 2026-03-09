# Connect 4 Self-Play PPO — Road to 2100 Elo

## Goal
Reach **2100 stable Elo** (not just peak) against the fixed reference pool.

## Reference Pool
18 opponents: 16 training snapshots (iter_0 to iter_3000, every 200) + random + heuristic.
- Strongest: iter_3000 (1585 Elo), heuristic (1584 Elo)
- To reach 2100 Elo: need ~95% win rate vs strongest pool members

## Baseline Config
lr=3e-4, ent_coef=0.001, bs=256, H=256, L=6 (366K params), pool=50 uniform, gpi=512, clip_eps=0.2

---

## Experiment 1: Entropy coefficient (2k iters)
| ent_coef | Final Elo | Avg10 | Peak | Result |
|----------|-----------|-------|------|--------|
| 0.001    | 1653      | 1663  | 1722 | **Winner** |
| 0.01     | 1704      | 1605  | 1704 | |
| 0.03     | 1563      | 1538  | 1600 | |
| 0.05     | 1552      | 1529  | 1558 | |
Higher entropy hurts.

## Experiment 2: Draw reward (2k iters)
| draw_reward | Final | Avg10 | Peak | HeurW% |
|-------------|-------|-------|------|--------|
| 0.0         | 1590  | 1570  | 1648 | 81.2%  |
| 0.2         | 1691  | 1631  | 1726 | 62.2%  |
| 0.5         | 1593  | 1608  | 1653 | 66.8%  |
| 1.0         | 1548  | 1557  | 1602 | 23.6%  |
Not significant. Keep baseline.

## Experiment 3: Games per iter + model size (2k iters)
| Config     | Final | Avg10 | Peak | Params |
|------------|-------|-------|------|--------|
| gpi512     | ~1663 | ~1663 | 1722 | 366K   |
| gpi1024    | 1574  | 1604  | 1677 | 366K   |
| **gpi2048**| **1792** | **1769** | **1903** | 366K |
| h512_l6    | 1589  | 1564  | 1650 | 1.39M  |
| h256_l8    | 1607  | 1603  | 1656 | 499K   |
**gpi2048 is a big win.** More data per iter = better gradients.

## Experiment 4: Mixed opponents (3k iters, gpi=2048)
Pinned heuristic net in training pool.
- baseline: 1735 avg
- pinned20%: 1608 avg
- pinned50%: 1525 avg
Pinned opponents hurt. Self-play diversity is better.

## Experiment 5: Long gpi2048 (5k iters)
- Final: 1817, Avg5: 1821, Peak: 1879
- High-precision eval (100 games/opp): **1837 true Elo**
- 100% vs heuristic deterministically

## Experiments 6-7: Cancelled
- exp6 (10k iters bs=256): killed, too slow at 3-4s/iter late training
- exp7 (batch size comparison): killed, pipe deadlock from parallel subprocesses

## Experiment 8: Long runs
### exp8 (gpi=2048, bs=512, 15k): KILLED at iter 1200
- iter 500: 1415, iter 1000: 1534
- bs=512 significantly worse than bs=256 (fewer gradient steps per iter)

### exp8b (gpi=2048, bs=256, 15k): KILLED at iter 3500
- Trajectory: 1478, 1579, 1764, 1748, 1901, 1826, 1836
- Plateaus around 1830 after iter 2500
- **Peak 1901 at iter 2500** (40 games/opp measurement)

## Experiment 9: Breaking the 1830 plateau

### exp9_gpi4096 (bs=1024, 5k): KILLED at iter 1750
- Trajectory: 1590, 1660, 1770, 1764, 1673, 1806, 1694
- Average ~1730, worse than gpi=2048 (fewer gradient updates per data point with bs=1024)

### exp9b_cosine_lr (gpi=2048, cosine LR 3e-4→3e-5): COMPLETED 5k iters
- Trajectory: 1520, 1535, 1669, 1671, 1615, 1701, 1670, 1716, 1702, 1809, 1800, 1719, 1733, 1797, 1783, 1809, 1803, 1851
- **Final: 1851, Avg5: 1837, Peak: 1867** (40 games/opp)
- **High-precision eval (100 games/opp): Stochastic 1890, Deterministic 1908**
- Cosine LR slightly better than constant (1837 vs 1830) but not breakthrough

### exp9b_ppo8 (8 PPO epochs): KILLED at iter 1000
- Trajectory: 1495, 1552, 1477, 1463
- KL too high (0.10-0.13), policy diverges. More epochs = too aggressive.

### exp9b_lr1e3 (lr=1e-3): Not started (killed before)

## Key Eval Discovery
High-precision eval (100 games/opp) of cosine LR best checkpoint:
- **Specific blind spots**: 78% vs iter_1400, 80% vs iter_2600 deterministically
- These are mid-range opponents (Elo 1412, 1454), not the strongest!
- Agent overfits to its own play style; struggles against old MLP architecture

## Experiment 10: Stability + diversity
### exp10_stable (clip=0.1, pool=200, snapshot_interval=5): KILLED at iter 2500
- Trajectory: 1701, 1696, 1704, 1717, 1713
- Stable at ~1710 but NOT improving. 0% vs heuristic by iter 2500!
- clip_eps=0.1 is too conservative — agent stops learning

## Critical Insight
- **clip=0.2 + pool=50**: oscillates but peak ~1900. Best actual performance.
- **clip=0.1 + pool=200**: stable but stuck at 1710. Too conservative.
- The "oscillation" at clip=0.2 is partly eval noise (40 games/opp → ±70 Elo)
- True Elo (100 games/opp) is ~1890, about 60 Elo higher than 40 games/opp avg

## Experiment 11: Not started (superseded by later experiments)

## Experiment 12: Mirror augmentation
Mirror augmentation (left-right flipping Connect 4 board) doubles training diversity.
- exp12_mirror (bs=256, clip=0.2): 1745→1805→1688. Oscillating, no improvement.
- exp12b_mirror_bs512 (bs=512): NaN at iter 231. Importance sampling ratios explode.
- exp12c_mirror_clip01 (bs=256, clip=0.1): NaN at iter ~190. Even conservative clipping fails.
**Mirror augmentation doesn't work** — mirrored transitions break PPO's importance sampling.

## Experiment 13: gpi=4096 with bs=256 — KILLED at iter 1000
- Trajectory (100 games/opp): 1607, 1644
- Much worse than gpi=2048 baseline — too many gradient steps per iter (~1875 vs ~936)
- The policy overshoots on each update

## Experiment 14: Long baseline + ppo_epochs=2
### exp14_long20k (baseline, 20k iters): KILLED at iter 4825
- Trajectory (100 games/opp): 1577, 1655, 1689, 1586, 1671, 1734, 1886, 1863, 1853
- **Peak: 1886** at iter 3500, then oscillates 1850-1890
- Confirms ~1890 ceiling for baseline config

### exp14b_ppo2 (ppo_epochs=2): KILLED at iter 1050
- 1552, 1605 — slower convergence with fewer gradient steps. Not promising.

## Experiment 15: Training diversity approaches
### exp15a_ent_schedule (ent_coef 0.01→0.001 cosine): KILLED at iter 2400
- Trajectory: 1540, 1578, 1586, 1563
- High entropy coefficient (0.01) hurts — too much randomness in policy

### exp15b_multi_opp (4 opponents per iter): KILLED at iter 2500
- Trajectory: **1688**, 1723, 1718, 1716, 1723
- Fast convergence to ~1720 but **lower ceiling** than baseline
- More diverse but weaker signal per opponent

## Experiment 16: Opponent temperature (**BREAKTHROUGH**)
### exp16a_opp_temp15 (opp_temp=1.5): KILLED at iter 5354
- Trajectory: 1628, 1681, 1781, 1780, 1765, 1867, **1970**, 1923, 1957, 1857
- **Peak: 1970** at iter 3500 (new record!)
- Average ~1900, oscillation 1860-1970
- +80 Elo over baseline's peak (1886)

### exp16b_opp_temp20 (opp_temp=2.0): KILLED at iter 2500
- Trajectory: 1675, 1710, 1738, 1760, 1702
- temp=2.0 peaks lower than temp=1.5 — opponent too random, doesn't challenge agent enough

## Key Finding: Opponent Temperature
- **opp_temp=1.5** is the single biggest improvement (+80 Elo peak) after gpi=2048
- Mechanism: stochastic opponents create diverse board states during self-play
- The agent learns to handle unexpected positions, reducing blind spots
- temp=2.0 is too much — opponent becomes too weak to teach useful play

## Experiment 17: Combined approaches
### exp17a_temp_anneal (opp_temp cosine 1.5→1.0): KILLED at iter 2000
- Trajectory: 1529, 1601, 1584, 1632
- Underperformed constant temp=1.5 — curriculum doesn't help

### exp17b, exp17c: Did not produce results (killed early / not started)

## Experiment 18: Best config long run (10k iters, eval every 250)
Config: gpi=2048, bs=256, opp_temp=1.5, 100 games/opp eval
- **Peak: 1943** at iter 8000
- Avg (last 10 evals): 1869
- Oscillation range: 1766–1943
- Killed at iter ~9450, did not complete
- Confirms ~1870 sustained, ~1940 peak with best config

Full trajectory:
| Iter | Elo | Iter | Elo | Iter | Elo |
|------|-----|------|-----|------|-----|
| 250 | 1546 | 3500 | 1834 | 6750 | 1892 |
| 500 | 1627 | 3750 | 1833 | 7000 | 1778 |
| 750 | 1692 | 4000 | 1828 | 7250 | 1864 |
| 1000 | 1699 | 4250 | 1868 | 7500 | 1892 |
| 1250 | 1789 | 4500 | 1938 | 7750 | 1797 |
| 1500 | 1862 | 4750 | 1931 | 8000 | **1943** |
| 1750 | 1793 | 5000 | 1826 | 8250 | 1854 |
| 2000 | 1853 | 5250 | 1805 | 8500 | 1830 |
| 2250 | 1805 | 5500 | 1766 | 8750 | 1894 |
| 2500 | 1821 | 5750 | 1766 | 9000 | 1839 |
| 2750 | 1878 | 6000 | 1818 | 9250 | 1942 |
| 3000 | 1799 | 6250 | 1880 | | |
| 3250 | 1893 | 6500 | 1770 | | |

## Final Summary

### Best Results
- **Peak Elo: 1970** (exp16a, opp_temp=1.5, iter 3500, stochastic 100g/opp)
- **Best sustained: ~1870** (exp18, avg of last 10 evals)
- **Best deterministic: 1908** (exp9b, cosine LR, 100g/opp)
- **Target: 2100** — not reached

### What Works (ranked by impact)
1. **gpi=2048** — +180 Elo (the foundation)
2. **opp_temp=1.5** — +80 Elo peak (biggest single improvement)
3. **bs=256** — more gradient steps per iter is crucial
4. **clip=0.2** — needed for learning; 0.1 too conservative
5. **100 games/opp eval** — accurate Elo measurement

### What Doesn't Work
- Higher entropy coefficient (0.01+), draw reward, bigger models, pinned opponents
- gpi=4096 with any bs (too many gradient steps)
- clip_eps=0.1 (too conservative), 8 PPO epochs (too aggressive)
- Mirror augmentation (breaks importance sampling, NaN)
- Multi-opponent (4 per iter) — stable but lower ceiling
- opp_temp=2.0 (too random, opponent too weak)
- Cosine LR (marginal), ppo_epochs=2 (too few updates)
- Temperature annealing (worse than constant)

---
