# Paired Assembly v2: Dynamic Local WIP Caps from 0 to 8

## 1. Launch

```bash
python -m pip install -r requirements.txt
python pair_v2_dashboard.py
```

Open `http://127.0.0.1:8767`. The historical Windows launcher `start_pair_v2_windows.bat` reuses/creates `.venv`, installs `requirements.txt` and starts the server with that interpreter. First installation needs Internet. Keep its terminal open. The old `pair_dashboard.py` and `run_pair_assembly.py` are v1; their results are not v2 results. For this delivery use the root `start_dashboard.cmd` and README.

## 2. Production Process

Each product requires one A and one B. Each branch passes through five stages into a finite merge buffer. Paired kits then pass through preassembly, assembly and testing. Each assembly stage has two independent machines sharing one FCFS queue.

| Cell | Serial stages | Parallel machines per stage | Baseline mean, minutes/operation |
|---|---:|---|---|
| A branch | 5 | 1,1,1,1,1 | 7,8,7,8,7 |
| B branch | 5 | 1,1,1,1,1 | 8,7,8,7,7 |
| Assembly | 3: preassembly, assembly, testing | 2,2,2 | 12,16,12 |

There are three cells, thirteen stages and sixteen machines. Internal interstage queues have no additional capacity limits but count against cell caps. Each merge buffer holds 3 parts, outside local caps but included in total WIP.

Each cell's nominal bottleneck capacity is 7.5 parts/sets per hour. Raw processing totals are 37 minutes for each A/B branch and 40 per assembly set. Zero-variability critical-WIP estimates are about 4.625, 4.625 and 5. Actual cap selection also depends on variability, pairing, finite buffers and targets, and requires scanning.

These are synthetic research parameters, not calibrated factory data. There are no transport, breakdown, changeover or scrap processes.

## 3. Three Local CONWIP Caps

- A/B caps bound parts queued, processing or blocked after the last stage. A branch slot is returned when a part enters its merge buffer.
- Assembly cap bounds kits across preassembly→assembly→testing. Pairing needs both buffers and a free cap slot. Final testing returns the slot.
- Pair the earliest arrivals from both buffers FIFO. Pairing consumes one part from each and creates one kit.
- cap=0 pauses new releases/pairing; existing work continues. Reducing cap deletes nothing. WIP may temporarily exceed the new cap until natural completion.
- Total part-equivalent WIP = A WIP + B WIP + both buffer counts + 2×assembly-kit WIP. Default maximum normalization is 38.
- There is no global kit-authorization card. Each cap is local to a cell.

## 4. Dynamic Regimes and Default Schedule

Warm up 720 minutes, then control 3600 minutes (60 hours). Decisions every 10 minutes; rolling window 120. Initial caps (3,3,2). Default processing is Gamma with tabulated means and shape α=9. Smaller α increases variation at unchanged mean. Uniform and deterministic distributions are also supported.

Times below start after warm-up:

| Control time | Integer target, sets/hour | Production change |
|---|---:|---|
| 0–600 min | 3 | Baseline means; α=9 in all cells |
| 600–1200 | 6 | Higher target, other parameters unchanged |
| 1200–1800 | 5 | A/B α:9→2; means unchanged; assembly α=9 |
| 1800–2400 | 5 | α restored to 9; B stage 3:8→9.2 minutes |
| 2400–3000 | 5 | B restored; A stage 2:8→9.2 minutes |
| 3000–3600 | 3 | Baseline means/α restored; lower target |

An independent static `assembly_variance` configuration sets assembly α=2, A/B α=9 and target 6 to isolate assembly variability. The historical interface can schedule it at a future control time and change assembly critical-stage multipliers.

During slow A/B stages, nominal bottleneck capacity is about 6.52 sets/hour, above target 5. Under high branch variability, a 12-seed long run (24-hour warm-up, 168-hour measurement) at caps (8,8,8) averaged 5.986 sets/hour. Default dynamic high-variability target is therefore 5. Target 6 remains a separate stress test; its shortage should not all be attributed to the controller.

Targets and individual completions are integer sets/hour and sets. Rolling throughput, integrated references and multi-seed means can be fractional. Do not round statistics used by control merely for display.

## 5. Actions, Observations and Reward

### Actions

PettingZoo agents `cell_0`, `cell_1`, `cell_2` each use `Discrete(3)`: 0 decreases by 1, 1 holds, 2 increases by 1, bounded 0–8. Target, processing times, distributions, Gamma shape and window are external conditions, not actions.

### Observations

Each agent has 45 dimensions, named in `env.observation_names`:

- Local WIP/cap and both merge-buffer inventories.
- Actual rolling set rate, same-window target error, current and historical-window targets.
- Statistics window, control interval and remaining-time fraction.
- Local Gamma parameter, distribution type and uniform half-width.
- Previous interval's local processing, blocking and idle fractions, plus assembly pair-starvation fraction.
- For up to five local stages: current means, machine counts, queues and processing/blocked machine fractions. Pad assembly's two missing stages with zeros.

Idle machines have several causes; not all are starvation. Pair starvation specifically requires idle preassembly equipment, available assembly cap, empty preassembly queue and a missing A or B buffer part. Normalize by preassembly machine count.

`CentralizedPairV2` supplies 27 joint actions and concatenates local observations into 135 dimensions. It is not full SimPy internal state with all remaining service times. Observations exclude future schedules. Future training should randomize change times to reduce schedule memorization.

### Team Reward

```text
r = -[(actual window throughput - historical window target) / max(historical window target, 1)]²
    -0.05 × interval time-averaged part-equivalent WIP / 38
    -0.005 × mean absolute actual cap change across three cells
```

If buffer capacities change, WIP denominator becomes `32 + sum of buffer capacities`. Scheduled boundary crossings weight subinterval endpoint errors: a discrete approximation, not exact continuous error integration. WIP uses event-to-event time integration.

Throughput and reference share the same past window; target changes do not rewrite history. Initial rolling statistics may include warm-up, but formal total completions and mean WIP exclude it. Reward penalizes over- and underproduction. Minimum weighted cost is not equivalent to zero-shortage feasibility.

## 6. API Example

```python
from conwip.pair_v2 import PairV2Env, PairV2Config, CentralizedPairV2

env = PairV2Env()
obs, infos = env.reset(seed=42)
obs, rewards, terminations, truncations, infos = env.step({
    'cell_0': 2, 'cell_1': 1, 'cell_2': 0,
})
# Change external target and variability immediately without resetting WIP.
env.set_conditions(target=5, gamma=(2, 2, 9))
# Future times are control minutes after warm-up.
env.schedule_conditions(650.5, target=6)
# Replace the full stage-mean vector.
means = [list(x) for x in env.current.means]
means[2][1] = 18
env.set_conditions(means=means, keep_future_schedule=False)
```

Service time is sampled when an operation starts. Ongoing operations keep their sampled duration; later operations use new parameters. Future schedules remain unless explicitly canceled. Use `dataclasses.replace(env.cfg, window_minutes=...)` for an immediate window change. Actual throughput and reference both adopt the new window.

Processing randomness is indexed by `(seed, cell, stage, part/kit ID)`, independent of parallel machine ID. Policies use common random quantiles for corresponding operations, transformed through new distributions after parameter changes. Event times and completion order can still differ; full event trajectories are not identical.

## 7. Environment Checks and Reproduction Scans

```bash
python -m pytest -q
# Seven static configurations, including separate stress tests: single-cell caps, 729 joint combinations, order-gated comparisons and dynamic diagnostics.
python run_pair_v2.py --mode all --workers 8
# Independent 12-seed cap-sensitivity check.
python run_pair_v2.py --mode scan --workers 8 --seeds 42 123 2024 6101 6102 6103 6104 6105 6106 6107 6108 6109 --output outputs/pair_v2/sensitivity_validation
python plot_pair_v2.py
```

Historical reports refer to [outputs/pair_v2/RESULTS_English.md](outputs/pair_v2/RESULTS_English.md), which is not part of this delivery. The joint grid used two development seeds, so its minimum is not a validated robust optimum. Dynamic lookup is neither PI/MADRL nor a performance bound; PI needs fair adaptation and tuning for this version.

## 8. Supplementary Order-Gated Releases

`PairV2Config(demand_gated=True)` integrates target rate into integer orders. Each order permits one A and one B release, still subject to local caps. New orders are checked every minute. Unreleased orders are not WIP; actual parts/kits retain the stated accounting. Warm-up uses the same rule.

This supports fixed-cap plus order-gated comparisons. Fixed, PI and MADRL must share the same order rule. It adds no global WIP card and does not turn targets into actions. At low demand, large fixed caps may add no WIP under order gating, so dynamic caps are not automatically beneficial.

## 9. Historical Web Interface and Validation Scope

The historical interface shows 16 machines, stage queues, finite buffers, WIP and throughput; supports run/pause/step, live conditions/caps, scheduled regimes, configuration/trajectory export. That interface is manual, not automatic PI/MADRL. Manual caps can be set directly 0–8; algorithm actions remain ±1 per step. This delivery's separate root dashboard supports trained policies.

Historical backend/API and JavaScript syntax checks passed. Browser access then returned `ERR_BLOCKED_BY_CLIENT`, so that historical round did not complete browser visual/interaction checks or a Windows-machine check.


## Repairing Missing Windows Dependencies Such as simpy

An early v2 launcher used system Python while `install_windows.bat` installed to `.venv`, causing possible `ModuleNotFoundError: No module named simpy`. The revised historical launcher uses `.venv\Scripts\python.exe` consistently and stops on install failure. This delivery uses the root setup launcher.

When running the historical `pair_v2_dashboard.py` in an IDE, select that project's `.venv\Scripts\python.exe` interpreter.


