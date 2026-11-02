"""Run the last two predeclared seeds early to fill available CPU capacity."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_controlled_m4_ablation as exp

def main():
    jobs=[(spec.key,seed,str(exp.OUT),exp.MAX_EPISODES)
          for seed in exp.TRAIN_SEEDS[-2:] for spec in exp.SPECS]
    print(f"Starting {len(jobs)} independent predeclared runs",flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(exp.train_job,j) for j in jobs]
        for future in as_completed(futures):
            print(future.result(),flush=True)

if __name__=="__main__":main()
