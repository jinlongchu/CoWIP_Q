# MADQN Global Features + GRU + PI Residual Joint Experiment

## 1. Experiment Definition

This combined variant, **S13**, adds the following to MADQN:

- Global coordination features.
- GRU memory over the latest 4 observation frames.
- The current PI action as a baseline, with MADQN outputting a three-level residual: `decrease by 1 / hold / increase by 1`.

S13 excludes QMIX, n-step returns and PER. It uses 5 training seeds, 300 episodes each, evaluated on the same 12 test seeds, for 60 test episodes.

## 2. Comparison

| Method | Reward ×1000 (higher is better) | NRMSE (lower is better) | Mean total WIP (lower is better) | Cap-change points (lower is smoother) |
|---|---:|---:|---:|---:|
| Current MADQN M0 | -65.659 ± 12.260 | 0.197 ± 0.035 | 16.851 ± 1.841 | 351.0 ± 7.9 |
| S1: global features + GRU | -50.098 ± 6.954 | 0.153 ± 0.025 | 17.049 ± 1.885 | 351.8 ± 9.1 |
| S3: PI residual guidance | -52.731 ± 8.353 | 0.172 ± 0.022 | 16.037 ± 0.935 | 245.7 ± 36.5 |
| **S13: global features + GRU + PI residuals** | **-45.453 ± 4.166** | **0.152 ± 0.016** | **15.728 ± 1.192** | **214.6 ± 31.1** |
| Current PI (latest reconstruction) | -47.978 ± 3.906 | 0.168 ± 0.012 | 14.452 ± 0.217 | 129.6 ± 11.9 |

## 3. Main Findings

### 3.1 The combination outperforms both individual variants

S13 has NRMSE **0.1521**, approximately **0.8%** below S1's 0.1533 and **11.4%** below S3's 0.1717. Global temporal features improve tracking, while PI residual constraints reduce unnecessary large changes. The effects are complementary.

S13's aggregate reward is **-45.453**, better than S1's -50.098 and S3's -52.731. Mean WIP and cap-change points are also lower than S1, indicating that PI residual guidance mitigates S1's frequent actions.

### 3.2 The combination improves the main accuracy metric over latest PI

Compared with current PI:

- NRMSE decreases from 0.168 to 0.152, approximately **9.3%** better.
- Reward improves from -47.978 to -45.453.
- Mean WIP increases from 14.452 to 15.728.
- Cap-change points increase from 129.6 to 214.6.

S13 therefore improves tracking and aggregate reward over PI but not inventory or action frequency. Its advantage is accuracy/reward, with weaker smoothness.

### 3.3 NRMSE < 0.10 remains unmet

S13's best mean NRMSE is **0.1521**, still above 0.10. Combining global features, GRU and PI residuals is useful, but reaching below 0.10 requires further investigation of residual-learning targets, PI-weight decay and training stability.

## 4. Suggestions

The S13 results motivate the following follow-up experiments:

1. Adapt PI-residual guidance strength to tracking error, rather than training progress alone.
2. Add a residual-action change penalty or smoothing constraint to reduce cap-change points.
3. Use staged rewards combining tracking error, WIP and action changes to lower WIP while maintaining accuracy.
4. Do not immediately add the then-current PER implementation, because standalone S2 showed substantial seed sensitivity and WIP growth.

## 5. Result Files

- [Joint experiment summary CSV](comparison_standalone.csv)
- [Results by training and test seed](evaluation_per_run.csv)
- [Training progress and model manifest](training_manifest.json)
