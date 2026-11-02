"""Add paired contrasts and the frozen PI baseline to the formal ablation."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

import run_controlled_m4_ablation as exp
from conwip.baseline_algorithms import FrozenCoordinatedPI, evaluate_episode, new_config

OUT = exp.OUT

def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def write_rows(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def bootstrap_mean(values, rng, n=4000):
    a=np.asarray(values,float)
    if len(a)==0: return [float("nan"),float("nan")]
    samples=rng.choice(a,(n,len(a)),replace=True).mean(axis=1)
    return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]

def main():
    rows=read_rows(OUT/"evaluation_per_episode.csv")
    if any(r["config"]=="pi_baseline" for r in rows):
        rows=[r for r in rows if r["config"]!="pi_baseline"]
    # Evaluate PI once on the same unseen test scenarios used by learned agents.
    pi_rows=[]
    for seed in exp.TEST_SEEDS:
        # The PI controller has an integral/filter state, so reset it per episode.
        m,_=evaluate_episode(FrozenCoordinatedPI(),seed,new_config())
        x={"reward_x1000":m.reward_x1000,"nrmse":m.nrmse,"mean_wip":m.mean_wip,
           "cap_adjustment_points":m.cap_adjustment_points,"config":"pi_baseline",
           "label":'Coordinated PI baseline',"method":"pi","split":"test","seed":seed,"run_seed":-1}
        rows.append(x); pi_rows.append(x)
    fields=["config","label","method","split","run_seed","seed","reward_x1000","nrmse","mean_wip","cap_adjustment_points"]
    write_rows(OUT/"evaluation_per_episode.csv",fields,rows)

    # Rebuild concise summaries from the raw episode-level test data.
    summaries=[]
    for spec in exp.SPECS:
        for method in ("single","q_ensemble_5"):
            sub=[r for r in rows if r["config"]==spec.key and r["method"]==method and r["split"]=="test"]
            vals={k:np.asarray([float(r[k]) for r in sub]) for k in ("reward_x1000","nrmse","mean_wip","cap_adjustment_points")}
            item={"config":spec.key,"label":spec.label,"method":method,"split":"test","n":len(sub)}
            for k,a in vals.items():
                item[k+"_mean"]=float(a.mean()); item[k+"_sd"]=float(a.std(ddof=1)) if len(a)>1 else 0.
            if method=="single":
                runmeans={}
                for s in exp.TRAIN_SEEDS:
                    rr=[r for r in sub if int(r["run_seed"])==s]
                    runmeans[s]={k:float(np.mean([float(r[k]) for r in rr])) for k in vals}
                item["training_run_mean"]={k:float(np.mean([v[k] for v in runmeans.values()])) for k in vals}
                item["training_run_sd"]={k:float(np.std([v[k] for v in runmeans.values()],ddof=1)) for k in vals}
            summaries.append(item)
    pi_item={"config":"pi_baseline","label":'Coordinated PI baseline',"method":"pi","split":"test","n":len(pi_rows)}
    for k in ("reward_x1000","nrmse","mean_wip","cap_adjustment_points"):
        a=np.asarray([float(r[k]) for r in pi_rows]);pi_item[k+"_mean"]=float(a.mean());pi_item[k+"_sd"]=float(a.std(ddof=1))
    summaries.append(pi_item)
    (OUT/"comparison_summary.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")

    # Paired contrasts: ablation minus full M4, on identical held-out seeds.
    contrasts=[];rng=np.random.default_rng(93017)
    for method in ("single","q_ensemble_5"):
        full={(int(r["run_seed"]),int(r["seed"])):r for r in rows if r["config"]=="full_m4" and r["method"]==method and r["split"]=="test"}
        for spec in exp.SPECS:
            if spec.key=="full_m4": continue
            target={(int(r["run_seed"]),int(r["seed"])):r for r in rows if r["config"]==spec.key and r["method"]==method and r["split"]=="test"}
            for metric in ("reward_x1000","nrmse"):
                if method=="single":
                    diffs={(rs,es):float(target[(rs,es)][metric])-float(full[(rs,es)][metric]) for rs,es in full if (rs,es) in target}
                    run_ids=exp.TRAIN_SEEDS; env_ids=exp.TEST_SEEDS
                    boot=[]
                    for _ in range(3000):
                        sr=rng.choice(run_ids,len(run_ids),replace=True); se=rng.choice(env_ids,len(env_ids),replace=True)
                        boot.append(np.mean([diffs[(int(a),int(b))] for a in sr for b in se]))
                    values=list(diffs.values()); lo,hi=np.quantile(boot,[.025,.975])
                else:
                    diffs={es:float(target[(-1,es)][metric])-float(full[(-1,es)][metric]) for es in exp.TEST_SEEDS}
                    values=list(diffs.values()); lo,hi=bootstrap_mean(values,rng)
                contrasts.append({"method":method,"ablation":spec.key,"label":spec.label,"metric":metric,
                    "delta_ablation_minus_full":float(np.mean(values)),"ci95_low":float(lo),"ci95_high":float(hi),"paired_n":len(values)})
        # PI is not a trainable M4 ablation, but include a paired reference.
        pi_by_seed={int(r["seed"]):r for r in rows if r["config"]=="pi_baseline" and r["split"]=="test"}
        for metric in ("reward_x1000","nrmse"):
            if method=="single":
                diffs={(rs,es):float(pi_by_seed[es][metric])-float(full[(rs,es)][metric]) for rs,es in full}
                boot=[]
                for _ in range(3000):
                    sr=rng.choice(exp.TRAIN_SEEDS,len(exp.TRAIN_SEEDS),replace=True)
                    se=rng.choice(exp.TEST_SEEDS,len(exp.TEST_SEEDS),replace=True)
                    boot.append(np.mean([diffs[(int(a),int(b))] for a in sr for b in se]))
                values=list(diffs.values());lo,hi=np.quantile(boot,[.025,.975])
            else:
                diffs={es:float(pi_by_seed[es][metric])-float(full[(-1,es)][metric]) for es in exp.TEST_SEEDS}
                values=list(diffs.values());lo,hi=bootstrap_mean(values,rng)
            contrasts.append({"method":method,"ablation":"pi_baseline","label":'Coordinated PI baseline',
                "metric":metric,"delta_ablation_minus_full":float(np.mean(values)),
                "ci95_low":float(lo),"ci95_high":float(hi),"paired_n":len(values)})
    (OUT/"paired_contrasts.json").write_text(json.dumps(contrasts,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"rows":len(rows),"summaries":len(summaries),"contrasts":len(contrasts),"pi":pi_item},ensure_ascii=False))

if __name__=="__main__":main()
