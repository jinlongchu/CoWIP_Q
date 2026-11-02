"""Train MADQN and compare four policies on the frozen PairV2 experiment."""
import argparse, csv, hashlib, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from conwip.baseline_algorithms import (
    new_config, select_fixed, FrozenCoordinatedPI, MADQN, MADQNTrainer,
    DQN, DQNTrainer, evaluate_episode, seed_all,
)

TRAIN_SEEDS=list(range(13101,13106))
TEST_SEEDS=list(range(14101,14113))
P=Path(__file__).parent
PACKAGED_BASE=P/'results'/'four_strategy_v2_madqn'
LEGACY_BASE=P.parent/'conwip_four_strategy_v2'/'results'/'four_strategy_v1'

def baseline_results():
    if (PACKAGED_BASE/'dqn').exists() and (PACKAGED_BASE/'fixed_selection.json').exists():return PACKAGED_BASE
    if (LEGACY_BASE/'dqn').exists() and (LEGACY_BASE/'fixed_selection.json').exists():return LEGACY_BASE
    raise FileNotFoundError('Bundled same-seed fixed/DQN baseline results are missing')

def checkpoint_path(value):
    path=Path(value)
    return path if path.is_absolute() else (P/path).resolve()

def dump_json(p,v): Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def dump_csv(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def train_madqn(job):
    run_seed,episodes,output=job
    seed_all(run_seed)
    root=Path(output)/'madqn'/f'run_{run_seed}';root.mkdir(parents=True,exist_ok=False)
    trainer=MADQNTrainer(run_seed,episodes*360);model=trainer.online
    rows=[];start=time.monotonic()
    for ep in range(episodes):
        seed=31_000_000+(run_seed-13101)*1000+ep
        m,_=evaluate_episode(model,seed,learn=True,dqn=trainer)
        rows.append(m.__dict__)
        if (ep+1)%8==0 or ep+1==episodes:
            record=dict(episode=ep+1,train_nrmse=float(np.mean([r['nrmse'] for r in rows[-8:]])),
                        train_reward_x1000=float(np.mean([r['reward_x1000'] for r in rows[-8:]])),
                        epsilon=trainer.eps,seconds=time.monotonic()-start)
            with (root/'training.csv').open('a',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(record))
                if f.tell()==0:w.writeheader()
                w.writerow(record)
            print(f'madqn run={run_seed} episode={ep+1}/{episodes} nrmse={record["train_nrmse"]:.3f} sec={record["seconds"]:.0f}',flush=True)
    torch.save(model.state_dict(),root/'model.pt')
    checkpoint=root/'model.pt'
    try: checkpoint_name=str(checkpoint.resolve().relative_to(P.resolve()))
    except ValueError: checkpoint_name=str(checkpoint.resolve())
    return dict(method='madqn',run_seed=run_seed,episodes=episodes,checkpoint=checkpoint_name,sha256=sha(checkpoint))

def metrics(method,run_seed,seed,m):
    row=m.__dict__.copy();row.update(method=method,run_seed=run_seed,seed=seed);return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['train','evaluate','all'],default='all')
    p.add_argument('--episodes',type=int,default=300);p.add_argument('--runs',type=int,default=5)
    p.add_argument('--workers',type=int,default=5);p.add_argument('--output',default='results/four_strategy_v2_madqn')
    a=p.parse_args();out=Path(a.output).resolve()
    if min(a.episodes,a.runs,a.workers)<1:p.error('episode, run and worker counts must be positive')
    if a.mode in ('train','all'):
        if out.exists():
            if (out/'frozen_models.json').exists():raise FileExistsError(f'Results already contain trained models: {out}')
        else:
            out.mkdir(parents=True)
            previous=baseline_results()
            # Freeze the exact fixed-cap scan and DQN models from the prior same-seed run.
            for filename in ('fixed_selection.json','fixed_training_grid.csv'):
                (out/filename).write_bytes((previous/filename).read_bytes())
            import shutil
            shutil.copytree(previous/'dqn',out/'dqn')
        previous=baseline_results()
        frozen=json.loads((previous/'frozen_models.json').read_text(encoding='utf-8'))
        frozen=[x for x in frozen if x['method']=='dqn']
        for entry in frozen:
            entry['checkpoint']=str((out/'dqn'/f"run_{entry['run_seed']}"/'model.pt').resolve().relative_to(P.resolve()))
        dump_json(out/'design.json',dict(environment=__import__('dataclasses').asdict(new_config()),
            train_seeds=TRAIN_SEEDS,test_seeds=TEST_SEEDS,runs=a.runs,episodes=a.episodes,
            fixed='same-seed frozen selection from prior four-strategy run; cap (3,3,3)',
            pi={'implementation':'reconstructed controller; documented frozen parameter values; source equation unavailable',
                'wip_factor':1.15,'kp':6.0,'ki':.25,'local_gain':.5,'action_deadband':.5,'error_filter':.5,'integral_limit':24.},
            madqn={'algorithm':'three independent local Double-DQN Q-networks; shared team reward; shared replay buffer',
                   'observation':'one 45D local observation per cell','actions':'cap decrement / hold / increment, invalid boundary actions masked',
                   'gamma':.99,'lr':1e-4,'epsilon_start':1.,'epsilon_end':.05,'hidden':128,
                   'replay_capacity':100000,'batch_size':128,'learning_starts':1000,'target_update_steps':1000},
            dqn={'reconstruction':'central 135D MLP; 27 joint actions; replay 100k; batch 128; target update 1000 steps',
                 'gamma':.99,'lr':1e-4,'epsilon_start':1.,'epsilon_end':.05,'hidden':256},
            reproducibility='Training seeds 13101–13105, 300 episodes per run; test seeds 14101–14112. Fixed scan and DQN checkpoints were carried forward unchanged from the same-seed PairV2 run. PI transfer function is reconstructed because supplied snapshot lacks its source.' ,
            source_sha256={str(x.relative_to(P)):sha(x) for x in (P/'conwip'/'pair_v2.py',P/'conwip'/'baseline_algorithms.py',Path(__file__))}))
        jobs=[(s,a.episodes,str(out)) for s in TRAIN_SEEDS[:a.runs]]
        if a.workers==1:
            frozen.extend(train_madqn(job) for job in jobs)
        else:
            with ProcessPoolExecutor(max_workers=min(a.workers,len(jobs))) as pool:
                frozen.extend(pool.map(train_madqn,jobs))
        dump_json(out/'frozen_models.json',frozen)
    if a.mode in ('evaluate','all'):
        if not (out/'fixed_selection.json').exists():raise FileNotFoundError('Run --mode train first')
        fixed=tuple(json.loads((out/'fixed_selection.json').read_text(encoding='utf-8'))['caps'])
        cfg=new_config();rows=[]
        for seed in TEST_SEEDS:
            m,_=evaluate_episode(fixed,seed,cfg);rows.append(metrics('fixed_conwip',-1,seed,m))
            m,_=evaluate_episode(FrozenCoordinatedPI(),seed,cfg);rows.append(metrics('coordinated_pi_reconstructed',-1,seed,m))
        frozen=json.loads((out/'frozen_models.json').read_text(encoding='utf-8'))
        for entry in frozen:
            model_path=checkpoint_path(entry['checkpoint'])
            if sha(model_path)!=entry['sha256']:raise ValueError('frozen checkpoint hash mismatch')
            model=MADQN() if entry['method']=='madqn' else DQN()
            model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True));model.eval()
            for seed in TEST_SEEDS:
                m,_=evaluate_episode(model,seed,cfg,stochastic=False)
                rows.append(metrics(entry['method'],entry['run_seed'],seed,m))
        dump_csv(out/'evaluation_per_run.csv',rows)
        groups={}
        for r in rows:groups.setdefault(r['method'],[]).append(r)
        summary=[]
        for method,rr in groups.items():
            z=dict(method=method,n=len(rr),training_runs=len(set(x['run_seed'] for x in rr if x['run_seed']>=0)),
                   test_seeds=len(set(x['seed'] for x in rr)))
            for k in ('reward_x1000','nrmse','mean_wip','cap_adjustment_points'):
                x=np.array([r[k] for r in rr]);z[k+'_mean']=float(x.mean());z[k+'_sd']=float(x.std(ddof=1))
            summary.append(z)
        dump_csv(out/'comparison_new_environment.csv',summary);dump_json(out/'comparison_new_environment.json',summary)
        print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
