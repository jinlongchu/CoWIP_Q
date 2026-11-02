# M4 Joint Q Ensemble Reward Improvement Report

## 1. Modification

Starting from the Q ensemble of 5 independent M4 models, also use each model's already-trained QMIX mixer:

1. Enumerate up to 27 residual-action combinations for the 3 agents.
2. Remove combinations that violate cap bounds.
3. Evaluate the joint Q value with each M4 model.
4. Average the joint Q values across the 5 models.
5. Select the legal combination with the highest mean team Q value.

The preceding Q ensemble selected each station's maximum independently and did not fully use M4's team coordination. This change affects only execution-time action selection, not the environment, reward or trained models.

## 2. Comparison

### Original test seeds (14101–14112)

| Method | Reward ×1000 | NRMSE | Mean WIP | Cap-change points |
|---|---:|---:|---:|---:|
| Original single-model M4 | -43.066 | 0.1453 | 15.226 | 265.4 |
| M4 independent Q ensemble | -39.201 ± 2.259 | 0.1344 | 14.790 | 246.4 |
| **M4 joint Q ensemble** | **-38.774 ± 2.821** | **0.1332** | **14.691** | **246.5** |

Joint Q ensembling improves reward by approximately **0.43 × 1000** over independent Q ensembling and by approximately **4.29 × 1000** over original single-model M4, reducing aggregate cost by approximately **10%**.

### New seeds not used for tuning (15101–15120)

| Method | Reward ×1000 | NRMSE | Mean WIP | Cap-change points |
|---|---:|---:|---:|---:|
| PI | -46.865 ± 2.838 | 0.1642 | 14.506 | 136.0 |
| M4 independent Q ensemble | -38.385 ± 2.057 | 0.1310 | 14.864 | 248.3 |
| **M4 joint Q ensemble** | **-38.125 ± 2.228** | **0.1302** | **14.807** | **248.0** |

On the new seeds, joint Q ensembling versus PI:

- Reward improves by approximately **8.74 × 1000**, a relative improvement of approximately **18.7%**.
- NRMSE decreases by approximately **20.7%**.
- WIP is only approximately 2.1% higher.
- Cap-change points are approximately 1.82 times those of PI.

## 3. Conclusion

The best-performing variant in this experiment was:

**M4 + 5-model Q ensemble + QMIX joint-action search**.

It further improves reward and NRMSE over independent Q ensembling without increasing cap-change frequency. This suggests that some previous performance loss arose from not using QMIX joint-value evaluation at execution, rather than insufficient M4 model capacity.

The best reward on these new seeds is **-38.125**, better than the latest PI value of -46.865. Further optimization could investigate joint-Q risk control, action-candidate screening and model distillation, rather than simply adding more n-step/PER modules.

## 4. Result Files

- [Joint ensemble summary JSON](summary.json)
- [Joint ensemble results by seed](per_run.csv)
- [Independent ensemble results](../m4_ensemble_diagnostic/summary.json)
