# Four-Strategy Comparison (Final M4 Added)

| Method | Training runs | Evaluation seeds | reward ×1000 | NRMSE | Mean WIP | Cap-change points |
|---|---:|---:|---:|---:|---:|---:|
| Fixed CONWIP | 0 | 12 | -116.5701 ± 5.9655 | 0.31025 ± 0.00966 | 15.3696 | 1.00 |
| Coordinated PI (reconstructed) | 0 | 12 | -47.9778 ± 3.9057 | 0.16769 ± 0.01186 | 14.4520 | 129.58 |
| MADQN (original 300 episodes) | 5 | 12 × 5 | -65.6588 ± 12.2601 | 0.19686 ± 0.03543 | 16.8510 | 351.00 |
| Centralized DQN | 5 | 12 × 5 | -62.9914 ± 10.4461 | 0.17949 ± 0.03337 | 20.0859 | 351.82 |
| **M4 independent Q ensemble (extended training)** | **5** | **12** | **-36.8359 ± 1.7177** | **0.12642 ± 0.00677** | **14.6918** | **231.92** |

Reward is higher-is-better (closer to 0); NRMSE is lower-is-better. The final M4 row evaluates an independent Q ensemble of 5 plateau-selected checkpoints on 12 test seeds.
