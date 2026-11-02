"""Evaluate existing S13 members, with explicit legacy-mask sensitivity.

No checkpoint or training implementation is modified. Corrected inference
does not undo the legacy mask used during training, so it is a diagnostic.
"""
from pathlib import Path
import sys, json, time, hashlib
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'work/conwip_four_strategy_v3_madqn'))
import run_madqn_standalone as s
import run_madqn_ablation as r

OUT=ROOT/'outputs/s13_q_ensemble_comparison'
SEEDS={'test':list(range(14101,14113)), 'holdout':list(range(15101,15121)),
       'fresh':list(range(29101,29113))}
KEYS=('reward_x1000','nrmse','mean_wip','cap_adjustment_points')

class Ensemble(nn.Module):
    def __init__(self,models,corrected=False):
        super().__init__();self.models=nn.ModuleList(models);self.corrected=corrected
    def forward(self,x,mask):
        if self.corrected:mask=s.standalone_mask(x[:,:,-1],'s13')
        return torch.stack([m(x,mask) for m in self.models]).mean(0)
    def agent_values(self,x,mask):
        return torch.stack([m.agent_values(x,mask) for m in self.models]).mean(0)

def load(paths,is_s13):
    members=[]
    for p in paths:
        m=r.RecurrentAgentQ(72) if is_s13 else r.CTDEModel(72)
        m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));members.append(m.eval())
    return members

def summary(rows):
    out={'n':len(rows)}
    for k in KEYS:
        a=np.array([v[k] for v in rows]);out[k+'_mean']=float(a.mean());out[k+'_sd']=float(a.std(ddof=1))
    return out

def main():
    r.seed_all(29001);OUT.mkdir(parents=True,exist_ok=True)
    p_s=[ROOT/f'outputs/madqn_joint_ablation/s13/run_{seed}/model.pt' for seed in range(13101,13106)]
    p_m=[ROOT/f'outputs/madqn_ablation/m4/run_{seed}/model.pt' for seed in range(13101,13106)]
    p_f=[ROOT/f'outputs/madqn_m4_final_converged/run_{seed}/best_model.pt' for seed in range(13101,13106)]
    sm=load(p_s,True);mm=load(p_m,False);fm=load(p_f,False)
    manifest={'training':'reuse five existing 300-episode S13 checkpoints; no retraining',
      'limitation':'legacy training uses direct-action masks for residual Q; corrected inference is not corrected training',
      'seeds':SEEDS,'sources':[{ 'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in p_s+p_m+p_f]}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Check the corrected mask against the explicit residual-to-direct mapping.
    checks=0
    for cap in range(9):
        for pi in range(3):
            x=torch.zeros(1,3,72);x[:,:,1]=cap/8;x[:,:,69+pi]=1
            actual=s.standalone_mask(x,'s13')[0,0].tolist()
            expected=[0<=pi+a-1<=2 and 0<=cap+pi+a-2<=8 for a in range(3)]
            assert actual==expected,(cap,pi,actual,expected);checks+=1
    results=[];summaries=[];start=time.monotonic()
    variants=[('s13_legacy_q_ensemble',Ensemble(sm),True),
              ('s13_corrected_inference_only',Ensemble(sm,True),True),
              ('m4_300_q_ensemble',Ensemble(mm),False),
              ('m4_final_q_ensemble',Ensemble(fm),False)]
    for name,model,is_s13 in variants:
        model.eval()
        for split,seeds in SEEDS.items():
            group=[]
            for seed in seeds:
                if is_s13:row=s.run_episode('s13',seed,model=model)
                else:
                    r.trainer_model=model;row=r.run_episode('m4',seed)
                row.update(method=name,split=split,seed=seed);results.append(row);group.append(row)
                (OUT/'per_episode.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            item={'method':name,'split':split,**summary(group)};summaries.append(item)
            (OUT/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
            print(json.dumps(item),flush=True)
    (OUT/'verification.json').write_text(json.dumps({'mask_cases_passed':checks,'episodes':len(results),'seconds':time.monotonic()-start},indent=2),encoding='utf-8')
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
