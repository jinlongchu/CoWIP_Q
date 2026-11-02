# PairV2 Four-Strategy Comparison (MADQN Variant)

The frozen environment uses initial caps (3,3,2), 720-minute warm-up and 3600-minute control, unchanged targets/disturbances, WIP weight 0.05 and action-change weight 0.005. Policies control only three integer cell WIP caps.

| Strategy | n | Reward ×1000 | Normalized RMSE | Mean total WIP | Cap-change decisions |
|---|---:|---:|---:|---:|---:|
| Fixed CONWIP (training-seed scan) | 12 | -116.570 ± 5.965 | 0.310 ± 0.010 | 15.370 ± 0.251 | 1.000 ± 0.000 |
| Coordinated PI (reconstructed) | 12 | -47.978 ± 3.906 | 0.168 ± 0.012 | 14.452 ± 0.217 | 129.583 ± 11.882 |
| MADQN (trained in this experiment) | 60 | -65.659 ± 12.260 | 0.197 ± 0.035 | 16.851 ± 1.841 | 351.000 ± 7.857 |
| Centralized DQN (existing matched-seed training) | 60 | -62.991 ± 10.446 | 0.179 ± 0.033 | 20.086 ± 2.666 | 351.817 ± 6.808 |

Fixed-CONWIP scans and central-DQN training reuse matched-environment/seed results. New MADQN uses training seeds 13101–13105, 300 episodes each, and evaluation seeds 14101–14112. PI uses frozen parameters but was reconstructed because its equations were absent from the source archive.

MADQN comprises three independent local Double-DQNs. Each cell reads its own 45-D observation and selects decrease/hold/increase. Cells share team reward and joint replay. Evaluation deterministically selects each agent's highest-Q legal action.

This is a synthetic SimPy research environment, not an AnyLogic numerical replica or a calibrated real factory. NRMSE≤0.10 must be established by measured test results.
