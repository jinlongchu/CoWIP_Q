# MADQN Single-Factor Ablation Report

## 1. Purpose

This experiment asks how MADQN training and testing change when each module below is added separately, without QMIX or the other improvements.

- **S1: global features + GRU** adds global coordination features to each local agent and uses a GRU over the latest 4 observation frames.
- **S2: 3-step returns + PER** retains feedforward local MADQN and adds only 3-step returns and prioritized experience replay.
- **S3: PI residual guidance** retains feedforward local MADQN, uses the current PI action as a baseline, and learns only `decrease by 1 / hold / increase by 1` residuals.

Each variant uses 5 training seeds, 300 episodes each, evaluated on the same 12 test seeds, for 60 test episodes. S1, S2 and S3 are single-factor comparisons and do not include QMIX.

## 2. Test Results

| Method | Reward ×1000 (higher is better) | NRMSE (lower is better) | Mean total WIP (lower is better) | Cap-change points (lower is smoother) |
|---|---:|---:|---:|---:|
| Current MADQN baseline M0 | -65.659 ± 12.260 | 0.197 ± 0.035 | 16.851 ± 1.841 | 351.0 ± 7.9 |
| S1: global features + GRU | **-50.098 ± 6.954** | **0.153 ± 0.025** | 17.049 ± 1.885 | 351.8 ± 9.1 |
| S2: n-step + PER | -61.590 ± 8.324 | 0.174 ± 0.032 | 20.496 ± 4.062 | 346.1 ± 12.1 |
| S3: PI residual guidance | -52.731 ± 8.353 | 0.172 ± 0.022 | **16.037 ± 0.935** | **245.7 ± 36.5** |
| Current PI (latest reconstruction) | -47.978 ± 3.906 | 0.168 ± 0.012 | 14.452 ± 0.217 | 129.6 ± 11.9 |

## 3. Changes from Current MADQN

- **S1 global features + GRU:** NRMSE falls from 0.197 to 0.153, approximately **22.1%**. Reward improves substantially, but WIP rises slightly and adjustment frequency is nearly unchanged.
- **S2 n-step + PER:** NRMSE falls to 0.174, approximately **11.7%**. WIP rises to 20.50 and cross-seed variation is largest, indicating limited stability under the present budget and hyperparameters.
- **S3 PI residual guidance:** NRMSE falls to 0.172, approximately **12.8%**. WIP falls to 16.04 and cap-change points from 351.0 to 245.7. Actions are smoother, but tracking is worse than S1.

## 4. Interpretation

### S1 is the strongest standalone accuracy improvement

S1 has the lowest NRMSE, 0.153. This suggests that missing system-level coordination information and short-term trends are important bottlenecks. Global features expose other stations' states, while GRU supplies temporal information about throughput, blocking and WIP changes.

### S2 does not provide a clear standalone overall gain

n-step and PER can help with delayed returns and rare useful experiences, but this environment's feedback is not entirely sparse. PER may repeatedly amplify unusual TD-error samples. Mean NRMSE improves moderately while WIP and cross-seed variation rise. S2 is therefore not recommended as a standalone improvement under this configuration.

### S3 mainly improves smoothness rather than replacing PI

S3 substantially lowers WIP and cap-change points, suggesting that PI residual actions constrain large policy changes. However, its NRMSE 0.172 is slightly worse than latest PI's 0.168 and substantially worse than S1. The implementation mainly learns small corrections around PI, not yet consistently effective corrections that outperform it.

## 5. Conclusion

Ranking these standalone additions:

1. **Global features + GRU: highest priority to retain, mainly improving tracking.**
2. **PI residual guidance: potentially useful as a stability constraint, but better residual rewards or adaptive residual magnitude are needed to outperform PI.**
3. **n-step + PER: not recommended alone under this configuration, because seed sensitivity and WIP variation offset benefits.**

None of the three single-factor variants achieves NRMSE < 0.10. S1 is best at **0.153**. Further work toward 0.10 should start from S1 and investigate residual-learning targets and stability, rather than directly adding the current S2.

## 6. Result Files

- [Single-factor summary CSV](comparison_standalone.csv)
- [Results by training and test seed](evaluation_per_run.csv)
- [Training progress and model manifest](training_manifest.json)
