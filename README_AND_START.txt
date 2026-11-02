COWIP_Q: Simulation, Controllers and Experiments
Release date: 2026-11-02

1. Current strategy
M4 independent Q ensemble with PI residual guidance OFF.
Global features, GRU, QMIX, 3-step and PER are retained. Five independently trained models average action Q values.
The English interface connects to the actual Python / SimPy discrete-event simulation.

2. Launch after extraction (Windows)
1. Extract the complete archive into a writable folder. Do not run it inside an archive viewer.
2. On a new computer, install Python 3.10, 3.11 or 3.12. The original experiment used Python 3.10.
3. Double-click setup_windows.cmd. It creates work/.venv_madqn_ablation and installs core dependencies. First installation requires Internet.
4. After installation, double-click start_dashboard.cmd.
5. The browser opens http://127.0.0.1:8767. If occupied, ports 8768–8777 are tried.

Available policies: five-model M4 ensemble, five-model original-MADQN ensemble, coordinated PI and manual caps.
Features: run/pause/step, production-condition changes, real stage/queue states, metric curves and CSV export.
The interface loads existing models for inference. It does not retrain.

On other systems, create a Python 3.10 environment, install requirements.txt and run:
  python work/run_des_dashboard.py --port 8767
Windows double-click launchers are Windows-specific. English PDF builders use the bundled DejaVu Sans font.

3. Verify package integrity
Run from the extracted project root:
  python verify_project.py
This standard-library command checks every file's SHA-256 and size.
After installing core dependencies, run a read-only inference check with the project interpreter:
  work\.venv_madqn_ablation\Scripts\python.exe verify_project.py --simulate
It loads the five latest M4 models and advances one 10-minute decision, without training or changing weights.

4. Environment and source code
work/conwip_four_strategy_v3_madqn/conwip/pair_v2.py
  Original production environment: five stages each for A/B, three double-machine assembly stages and three local CONWIP loops.
work/conwip_four_strategy_v3_madqn/conwip/baseline_algorithms.py
  Original MADQN, centralized DQN and current coordinated PI.
work/run_controlled_m4_ablation.py
  Matched training, validation, early stopping, testing and five-model ensembling.
work/conwip_four_strategy_v3_madqn/run_madqn_ablation.py
  Global summaries, shared GRU, QMIX, PER and related modules.
work/run_des_dashboard.py and work/des_dashboard/
  Local web service and English HTML/CSS/JavaScript.
work/train_mpc_controller.py
  Learned-model MPC training and control.

5. Latest models and experiment results
outputs/controlled_m4_ablation/formal/no_pi/run_13101～run_13105/best_model.pt
  The five latest M4 checkpoints actually used by the dashboard.
outputs/controlled_m4_ablation/formal/original_madqn/
  Matched-retrained original MADQN models.
outputs/m4_no_pi_ablation/
  Five module ablations around the PI-off baseline, training diagnostics and per-episode test records.
outputs/controlled_m4_ablation/comparison_summary.json
  Original MADQN, PI, PI-on M4 and PI-off M4 compared on the same 30 test seeds.
outputs/comparison_tables/
  M4_No_PI_Module_Ablation.xlsx and M4_Ablation_vs_MADQN.xlsx.
outputs/mpc_model_predictive/
  Trained MPC predictors and results.

6. Key experiment conclusions
Latest PI-off M4 five-model ensemble: test mean reward×1000=-35.54014295079746,
NRMSE=0.12301771925080808, mean WIP=14.79844515798786, cap-change points=153.16666666666666.
Matched original-MADQN ensemble: reward×1000=-47.855605026534214, NRMSE=0.13112620026710572.
Matched coordinated PI: reward×1000=-48.30802556705776, NRMSE=0.16852212087623164.
These are test rewards, not exploratory training rewards. One web run is not a multi-seed average.
Early 300-episode experiments, historical extended M4 training and latest matched experiments have different protocols. Do not mix them as a fair matched comparison.
Of five PI-off M4 runs, four meet the validation-plateau stopping rule and one reaches the 1250-episode budget.
NRMSE<0.10 has not been achieved. Existing results have informed version selection and are not fresh external validation.

7. Booklets and learning curves
outputs/M4_Algorithm_Booklets/M4_No_PI_MADQN_Based_Overview.pdf
outputs/M4_Algorithm_Booklets/M4_No_PI_Architecture_and_Improvements.pdf
outputs/project_illustrated_book/From_Production_to_M4_Project_Guide.pdf
training.csv, validation.csv and learning_curve.csv in each run directory retain raw training/validation records.
Historical formal experiments, old models, diagnostic results and small smoke tests are also retained.

8. Portability and dependency boundaries
Machine-bound virtual environments, node_modules, caches, temporary files, QA previews and web-server logs are excluded.
requirements.txt lists core dependencies; requirements-original-environment.txt records original installed versions.
Fresh installations use core version ranges. Historical version records are provenance, not a guarantee that every old version remains publicly downloadable.
PDFs, Excel result tables and figures are included. Viewing them requires no document-generation dependencies.
To regenerate English PDFs, install requirements-reports.txt. Builders use work/fonts/DejaVuSans.ttf rather than Windows-only CJK fonts; its license is included.
The distribution includes the simulation, training, evaluation and PDF-generation source code.
Excel workbooks are distributed as completed result artifacts; their optional development-only builders are not included.
Checkpoint locations in experiment manifests and tables are relative to this project root. Run experiment commands from the root directory.
Model weights, numeric training logs and reported experimental results are unchanged.
The file manifest provides SHA-256 checksums for integrity checking. VERIFICATION_REPORT.json records the checks completed for this release.

9. Important files
PACKAGE_MANIFEST.json: file sizes and SHA-256 hashes.
verify_project.py: integrity check and optional model-inference check after extraction.
setup_windows.cmd: first-time Windows core-dependency installation.
start_dashboard.cmd: launch the English simulation interface.
