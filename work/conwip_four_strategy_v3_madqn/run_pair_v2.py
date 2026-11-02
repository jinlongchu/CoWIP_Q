"""Reproducible cap sensitivity, complete joint grids, and dynamic demo.

Scans are environment-development diagnostics, NOT held-out algorithm results.
"""
from __future__ import annotations
import argparse,csv,itertools,json,time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from conwip.pair_v2 import BASE_MEANS, Conditions, PairV2Config, PairV2Env


def profiles():
    b=list(BASE_MEANS[1]);b[2]*=1.15
    a=list(BASE_MEANS[0]);a[1]*=1.15
    return {
        'normal_low':Conditions(target=3),
        'normal_high':Conditions(target=6),
        'feed_variance':Conditions(target=6,gamma=(2.,2.,9.)),
        'feed_variance_feasible':Conditions(target=5,gamma=(2.,2.,9.)),
        'b_slow':Conditions(target=5,means=(BASE_MEANS[0],tuple(b),BASE_MEANS[2])),
        'a_slow':Conditions(target=5,means=(tuple(a),BASE_MEANS[1],BASE_MEANS[2])),
        'assembly_variance':Conditions(target=6,gamma=(9.,9.,2.)),
    }


def evaluate(job):
    name,caps,seed,horizon,warmup,gated=job
    cond=profiles()[name]
    cfg=PairV2Config(conditions=cond,schedule=(),initial_caps=tuple(caps),horizon_minutes=horizon,warmup_minutes=warmup,demand_gated=gated)
    e=PairV2Env(cfg);e.reset(seed=seed)
    error=0.;iae=0.;duration=0.
    # Identical endpoint cost and interval weighting to hold-action step(),
    # without building observations and copying info dictionaries on each step.
    while duration<horizon:
        dt=min(cfg.decision_minutes,horizon-duration);e._advance(e.sim.now+dt)
        th,ref=e.rolling_metrics();error+=dt*((th-ref)/max(ref,1.))**2;iae+=abs(th-ref)*dt/60;duration+=dt
    e.assert_invariants();m=e.metrics();mean_error=error/horizon
    return dict(profile=name,cap_a=caps[0],cap_b=caps[1],cap_assembly=caps[2],seed=seed,
                target=cond.target,throughput=m['kits_completed']*60/horizon,
                tracking_cost=mean_error,nrmse=mean_error**.5,iae_kits=iae,
                mean_wip=m['mean_wip_component_equiv'],peak_wip=m['peak_wip_component_equiv'],
                mean_cost=mean_error+cfg.inventory_weight*m['mean_wip_component_equiv']/e.max_total_wip,
                blocked_a=m['blocked_fraction'][0],blocked_b=m['blocked_fraction'][1],
                idle_assembly=m['idle_fraction'][2],pair_shortage=m['assembly_pair_shortage_fraction'],demand_gated=gated)


def writecsv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def aggregate(rows,extra=()):
    groups={}
    keys=('profile','cap_a','cap_b','cap_assembly','demand_gated',*extra)
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for group,rr in groups.items():
        row=dict(zip(keys,group));row['n_seeds']=len(rr)
        for key in ('throughput','mean_wip','mean_cost','nrmse','iae_kits','peak_wip','blocked_a','blocked_b','idle_assembly','pair_shortage'):
            vals=[r[key] for r in rr];row[key]=float(np.mean(vals));row[key+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
        result.append(row)
    return result


def run_tasks(jobs,workers):
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate,jobs,chunksize=4):
            rows.append(row)
            if len(rows)%200==0:print(f'{len(rows)}/{len(jobs)} episodes; {time.monotonic()-start:.1f}s',flush=True)
    return rows


def scan(args,out):
    # Other two caps held at 8; one-factor effects are not uniform-cap effects.
    jobs=[];labels=[]
    for name in profiles():
        for cell in range(3):
            for cap in range(9):
                caps=[8,8,8];caps[cell]=cap
                for seed in args.seeds:
                    jobs.append((name,tuple(caps),seed,args.horizon,args.warmup,False));labels.append(cell)
    rows=run_tasks(jobs,args.workers)
    for row,cell in zip(rows,labels):row['varied_cell']=cell;row['varied_cap']=row[('cap_a','cap_b','cap_assembly')[cell]]
    writecsv(out/'sensitivity_per_seed.csv',rows)
    agg=aggregate(rows,('varied_cell','varied_cap'));writecsv(out/'sensitivity_summary.csv',agg)


def grid(args,out):
    jobs=[(name,caps,seed,args.horizon,args.warmup,False) for name in profiles()
          for caps in itertools.product(range(9),repeat=3) for seed in args.seeds]
    rows=run_tasks(jobs,args.workers);writecsv(out/'joint_grid_per_seed.csv',rows)
    agg=aggregate(rows);writecsv(out/'joint_grid_summary.csv',agg)
    best={name:min((r for r in agg if r['profile']==name),key=lambda r:(r['mean_cost'],r['mean_wip'])) for name in profiles()}
    (out/'best_by_condition.json').write_text(json.dumps(best,indent=2),encoding='utf-8')
    # Common demand gating sensitivity: not an optimized algorithm comparison.
    gatejobs=[(name,(cap,cap,cap),seed,args.horizon,args.warmup,True)
              for name in profiles() for cap in (2,4,6,8) for seed in args.seeds]
    gated=run_tasks(gatejobs,args.workers);writecsv(out/'demand_gated_screen.csv',aggregate(gated))


def dynamic(out,seeds=(42,123,2024)):
    # Illustrative known-schedule controller from static development scan.
    # It is labelled stage_oracle, never PI/MADRL nor a generalization test or performance bound.
    bestpath=out/'best_by_condition.json'
    best=json.loads(bestpath.read_text()) if bestpath.exists() else None
    names=['normal_low','normal_high','feed_variance_feasible','b_slow','a_slow','normal_low']
    methods=['fixed_4','fixed_8']+(['stage_oracle'] if best else [])
    if best:
        # Best *one* fixed cap for known scenario development grid; all periods equal.
        with (out/'joint_grid_summary.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        scores={}
        for r in rows:
            if r['profile'] not in names:continue
            caps=tuple(int(r[k]) for k in ('cap_a','cap_b','cap_assembly'))
            scores[caps]=scores.get(caps,0)+names.count(r['profile'])*float(r['mean_cost'])
        frozen=min(scores,key=scores.get);methods.insert(0,'fixed_selected')
    traces=[];summaries=[]
    for seed in seeds:
        for method in methods:
            e=PairV2Env();e.reset(seed=seed);cost=0.;changes=0;iae=0.
            while e.agents:
                tm=e.sim.now-e.stats_start
                if method=='fixed_selected':goal=frozen
                elif method.startswith('fixed_'):goal=[int(method.split('_')[1])]*3
                else:
                    r=best[names[min(5,int(tm//600))]];goal=[r[k] for k in ('cap_a','cap_b','cap_assembly')]
                old=e.caps.copy();actions={a:1+int(np.sign(g-c)) for a,g,c in zip(e.agents,goal,e.caps)}
                _,rew,_,_,_=e.step(actions);m=e.metrics();cost-=rew['cell_0'];changes+=sum(a!=b for a,b in zip(old,e.caps))
                iae+=abs(m['throughput_moving_kits_per_hour']-m['tracking_target_kits_per_hour'])*e.cfg.decision_minutes/60
                traces.append(dict(seed=seed,method=method,minute=m['control_minutes'],target=m['target_kits_per_hour'],reference=m['tracking_target_kits_per_hour'],throughput=m['throughput_moving_kits_per_hour'],wip=m['total_wip_component_equiv'],cap_a=e.caps[0],cap_b=e.caps[1],cap_assembly=e.caps[2]))
            summaries.append(dict(seed=seed,method=method,return_sum=-cost,iae_kits=iae,mean_wip=m['mean_wip_component_equiv'],kits=m['kits_completed'],cap_changes=changes))
    writecsv(out/'dynamic_traces.csv',traces);writecsv(out/'dynamic_diagnostic.csv',summaries)


def feasibility(args,out):
    seeds=(42,123,2024,*range(6101,6110))
    jobs=[('feed_variance',(8,8,8),seed,10080.,1440.,False) for seed in seeds]
    writecsv(out/'feasibility_long_run.csv',run_tasks(jobs,args.workers))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['scan','grid','dynamic','feasibility','all'],default='scan')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seeds',nargs='+',type=int,default=[101,203])
    p.add_argument('--horizon',type=float,default=1440);p.add_argument('--warmup',type=float,default=720)
    p.add_argument('--output',default='outputs/pair_v2');args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'design.json').write_text(json.dumps(dict(config=asdict(PairV2Config()),profiles={k:asdict(v) for k,v in profiles().items()},scan_seeds=args.seeds,horizon=args.horizon,warmup=args.warmup,purpose='development diagnostics; not held-out controller ranking'),ensure_ascii=False,indent=2),encoding='utf-8')
    if args.mode in ('scan','all'):scan(args,out)
    if args.mode in ('grid','all'):grid(args,out)
    if args.mode in ('dynamic','all'):dynamic(out)
    if args.mode in ('feasibility','all'):feasibility(args,out)

if __name__=='__main__':main()
