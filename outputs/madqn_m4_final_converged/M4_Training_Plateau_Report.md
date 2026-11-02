# M4 Independent Q Ensemble: Extended-Training Results

## Conclusions

All 5 random seeds completed at least 300 training episodes, with the best checkpoint retained after validation curves reached a plateau. The validation criterion is no reward improvement for 4 consecutive validation cycles. Seeds 13102, 13103 and 13105 stopped at 750 episodes, 13101 at 900, and 13104 at 1250.

See [learning_curves.svg](learning_curves.svg). Raw curves are in each `run_*` directory's `learning_curve.csv` and `validation.csv`.

## Convergence Overview

| seed | episodes | Best validation episode | Best validation reward ×1000 | Best validation NRMSE |
|---:|---:|---:|---:|---:|
| 13101 | 900 | 700 | -39.6643 | 0.1352 |
| 13102 | 750 | 550 | -38.6856 | 0.1299 |
| 13103 | 750 | 550 | -37.6057 | 0.1307 |
| 13104 | 1250 | 1050 | -37.5522 | 0.1259 |
| 13105 | 750 | 550 | -38.5463 | 0.1341 |

## Final Performance

| Method | split | n | reward×1000 | NRMSE | mean WIP | cap adjustment points |
|---|---|---:|---:|---:|---:|---:|
| Final M4 independent Q ensemble | test | 12 | -36.8359 ± 1.7177 | 0.12642 ± 0.00677 | 14.6918 | 231.92 |
| Final M4 independent Q ensemble | holdout | 20 | -36.3854 ± 2.1638 | 0.12451 ± 0.00858 | 14.7050 | 229.95 |
| Mean of 5 individual checkpoints | test | 60 | -38.6484 ± 2.4624 | 0.13233 ± 0.00986 | 14.8821 | 229.92 |
| Mean of 5 individual checkpoints | holdout | 100 | -38.2049 ± 2.4773 | 0.13051 ± 0.00998 | 14.8995 | 231.53 |

## Comparison with Earlier Results

On the same test seeds, the previous M4 independent Q ensemble obtained reward×1000 = -39.2008 and NRMSE = 0.13442. This extended-training version obtains -36.8359 and 0.12642: reward improves 2.3650 and NRMSE decreases 0.00800.

On 20 holdout seeds, previous M4 obtained reward×1000 = -38.3855 and NRMSE = 0.13098. The extended-training version obtains -36.3854 and 0.12451: reward improves 2.0001 and NRMSE decreases 0.00646.

The current PI baseline obtains reward×1000 = -47.9778 and NRMSE = 0.16769. This M4 independent Q ensemble performs better on test reward and NRMSE.

## Files

- [Learning curves](learning_curves.svg)
- [Final summary](comparison_final.json)
- [Evaluation by seed](evaluation_per_run.csv)
- [Training manifest](training_manifest.json)
