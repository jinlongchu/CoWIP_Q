PairV2 Environment: Four-Strategy Comparison (MADQN Edition)

Four strategies: fixed CONWIP, coordinated PI (parameter-based reconstruction), MADQN and centralized DQN.
MAPPO was removed from this experiment version. The previous archive remains unchanged and separate.

Initial caps (3,3,2), 720-minute warm-up, 3600-minute control and 12 held-out test seeds.
MADQN trains on seeds 13101–13105 for 300 episodes each. DQN, fixed-policy scans and their seeds reuse the previous experiment.

Create a virtual environment, install `requirements-rl.txt`, then run in this directory:

    python run_four_strategies.py --mode all --episodes 300 --runs 5 --workers 1 --output results/madqn_rerun
    python make_comparison.py --results results/madqn_rerun --csv comparison_four_madqn.csv --md Four_Strategies_MADQN.md

MADQN uses three local Double-DQN learners, each with local observations and shared team reward. This is synthetic SimPy research simulation, not an AnyLogic numerical replica or calibrated factory.
`design.json`, checkpoints, training logs and per-seed evaluations jointly document design and results.
