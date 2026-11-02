# M4 Reward Improvement Experiment Report

## 1. Method

Starting from the original M4 (global features + GRU + QMIX + n-step + PER + PI residual guidance), ensemble the M4 Q networks from 5 independently trained seeds:

1. Feed each test state to all 5 M4 models.
2. Average the action Q values elementwise for each of the three agents.
3. Select the residual action using the averaged Q values.

This reduces Q-value bias associated with an individual training seed. The environment, reward and action space remain unchanged, and no retraining is required.

## 2. Results

### Original test seeds (14101–14112)

| Method | Reward ×1000 | NRMSE | Mean WIP | Cap-change points |
|---|---:|---:|---:|---:|
| Original single-model M4 (mean over 5 models × 12 seeds) | -43.066 ± 2.970 | 0.1453 | 15.226 | 265.4 |
| M4 Q ensemble | **-39.201 ± 2.259** | **0.1344** | **14.790** | **246.4** |
| M4 Q ensemble + 0.01 residual threshold | -39.696 ± 2.708 | 0.1388 | 14.612 | **193.3** |

Q ensemble versus original M4:

- Reward improves by approximately **3.87 × 1000**, reducing absolute cost by approximately **9.0%**.
- NRMSE decreases from 0.1453 to 0.1344.
- Both mean WIP and cap-change points decrease.

### New seeds not used for tuning (15101–15120)

| Method | Reward ×1000 | NRMSE | Mean WIP | Cap-change points |
|---|---:|---:|---:|---:|
| PI | -46.865 ± 2.838 | 0.1642 | 14.506 | 136.0 |
| Original M4 Q ensemble | **-38.385 ± 2.057** | **0.1310** | 14.864 | 248.3 |
| M4 Q ensemble + 0.01 residual threshold | -38.496 ± 1.971 | 0.1344 | 14.637 | **193.4** |

On the new seeds, the original M4 Q ensemble versus PI:

- Reward improves by approximately **8.48 × 1000**.
- NRMSE decreases by approximately **20.3%**.
- WIP is only approximately 2.5% higher.
- Cap-change points are approximately 1.83 times those of PI.

## 3. Other Trials

M5 was also formally retrained with a 0.01 PI-residual Q-advantage threshold during M4 training. M5 achieves NRMSE 0.1441 but reward -43.573, slightly worse than original M4. The training-time threshold variant is therefore not recommended.

An inference-time threshold substantially reduces action frequency but sacrifices some reward and tracking accuracy. If reward is the priority, use Q ensembling without a threshold.

## 4. Recommendation

Deployment procedure for this historical variant:

1. Retain the 5 trained original M4 models.
2. Average their agent Q values during inference.
3. Select the residual action with the highest averaged Q value.
4. Enable the 0.01 residual threshold if action smoothness later becomes more important.

This was more effective than further stacking n-step, PER or training-time thresholds. The best reward on these new seeds is **-38.385**. Additional random seeds were recommended to assess generalization.

## 5. Raw Result Files

- [Ensemble summary JSON](summary.json)
- [Ensemble results by seed](per_run.csv)
- [Threshold validation on new seeds](../m4_gate_validation/summary.json)
