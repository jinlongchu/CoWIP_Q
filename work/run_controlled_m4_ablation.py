"""Controlled component ablation for original MADQN versus full M4.

Configurations are fixed before training. All variants share data seeds,
validation seeds, episode seeds, stopping rule and final test seeds.
Correct PI-residual masks are used consistently for exploration, training,
validation and test. The experiment writes resumable per-run progress markers.
"""
from __future__ import annotations
import argparse, csv, json, math, os, random, sys, time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT/'work'/'conwip_four_strategy_v3_madqn'
sys.path.insert(0,str(REPO))
import run_madqn_ablation as base
from conwip.pair_v2 import PairV2Env
from conwip.baseline_algorithms import FrozenCoordinatedPI,new_config

OUT=ROOT/'outputs'/'controlled_m4_ablation'
TRAIN_SEEDS=list(range(13101,13106))
VALIDATION_SEEDS=list(range(16101,16113))
TEST_SEEDS=list(range(30101,30131))
MIN_EPISODES=300; MAX_EPISODES=1250; EVAL_INTERVAL=50; PATIENCE=4
MIN_REWARD_GAIN=.05; N_AGENTS=3; LOCAL_DIM=45; GLOBAL_DIM=24; ACTIONS=3; SEQ=4; GAMMA=.99

@dataclass(frozen=True)
class Spec:
    key:str; label:str; global_features:bool; gru:bool; qmix:bool; nstep:int; per:bool; pi:bool; original_arch:bool=False

SPECS=[
 Spec('full_m4','Full M4',True,True,True,3,True,True),
 Spec('no_global','M4 without global features',False,True,True,3,True,True),
 Spec('no_gru','M4 without GRU',True,False,True,3,True,True),
 Spec('no_qmix','M4 without QMIX',True,True,False,3,True,True),
 Spec('one_step','M4 without n-step (1-step)',True,True,True,1,True,True),
 Spec('uniform_replay','M4 without PER (uniform replay)',True,True,True,3,False,True),
 Spec('no_pi','M4 without PI residual guidance',True,True,True,3,True,False),
 Spec('original_madqn','Original MADQN (matched retraining)',False,False,False,1,False,False,True),
]
# A second-stage ablation fixes PI residual guidance OFF in every condition.
# The no_pi checkpoints are reused; only the five module-removal variants train.
if os.environ.get('M4_NO_PI_ABLATION') == '1':
 SPECS=[
  Spec('no_pi','New M4 baseline (PI off)',True,True,True,3,True,False),
  Spec('no_global','New M4 without global features',False,True,True,3,True,False),
  Spec('no_gru','New M4 without GRU',True,False,True,3,True,False),
  Spec('no_qmix','New M4 without QMIX',True,True,False,3,True,False),
  Spec('one_step','New M4 without n-step (1-step)',True,True,True,1,True,False),
  Spec('uniform_replay','New M4 without PER (uniform replay)',True,True,True,3,False,False),
 ]
SPEC={x.key:x for x in SPECS}

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)

def context(obs):return base.global_context(obs)

def make_frame(local,pi_actions,spec):
 local=np.asarray(local,dtype=np.float32)
 parts=[local]
 if spec.global_features:parts.append(np.repeat(context(local)[None,:],N_AGENTS,axis=0))
 if spec.pi:
  oh=np.eye(ACTIONS,dtype=np.float32)[np.asarray(pi_actions,dtype=int)]
  parts.append(oh)
 return np.concatenate(parts,axis=1).astype(np.float32)

def pi_offset(spec):return LOCAL_DIM+(GLOBAL_DIM if spec.global_features else 0)

def mask_for(frame,spec):
 """frame: batch x agents x features. PI residual mask maps to direct caps."""
 caps=torch.round(frame[...,1]*8).long()
 if not spec.pi:return torch.stack((caps>0,torch.ones_like(caps,dtype=torch.bool),caps<8),-1)
 pi=frame[...,pi_offset(spec):pi_offset(spec)+3].argmax(-1)
 residual=torch.arange(3,device=frame.device).view(1,1,3)-1
 direct=pi.unsqueeze(-1)+residual
 direct_cap=caps.unsqueeze(-1)+direct-1
 return (direct>=0)&(direct<=2)&(direct_cap>=0)&(direct_cap<=8)

class PolicyNet(nn.Module):
 def __init__(self,spec):
  super().__init__();self.spec=spec
  dim=LOCAL_DIM+(GLOBAL_DIM if spec.global_features else 0)+(ACTIONS if spec.pi else 0)
  self.recurrent=spec.gru
  if self.recurrent:self.core=base.RecurrentAgentQ(dim,hidden=64,prefer_hold=spec.pi)
  else:
   heads=[]
   for _ in range(N_AGENTS):
    if spec.original_arch:
     net=nn.Sequential(nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,3))
     for layer in net:
      if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,math.sqrt(2));nn.init.zeros_(layer.bias)
     nn.init.orthogonal_(net[-1].weight,1.)
    else:net=base.mlp(dim,3,64)
    heads.append(net)
   self.heads=nn.ModuleList(heads)
   if spec.pi and not spec.original_arch:
    for head in self.heads:
     with torch.no_grad():head[-1].bias[1]=.5
  self.mixer=base.QMix(GLOBAL_DIM) if spec.qmix else None
 def values(self,rep,mask):
  if self.recurrent:return self.core(rep,mask)
  frame=rep[:,:,-1,:] if rep.ndim==4 else rep
  q=torch.stack([self.heads[i](frame[:,i]) for i in range(N_AGENTS)],dim=1)
  return q.masked_fill(~mask,-1e9) if mask is not None else q

class TeamTrainer:
 def __init__(self,spec,seed,total_steps):
  self.spec=spec;self.rng=np.random.default_rng(seed);self.count=0;self.total_steps=total_steps;self.eps=1.
  self.model=PolicyNet(spec);self.target=PolicyNet(spec);self.target.load_state_dict(self.model.state_dict());self.target.eval()
  self.optim=torch.optim.Adam(self.model.parameters(),lr=1e-4)
  self.replay=base.Replay(100_000,spec.per);self.pending=deque();self.updates=0
 def _valid(self,frame):
  caps=np.rint(frame[:,1]*8).astype(int)
  if not self.spec.pi:return [[a for a,ok in enumerate((c>0,True,c<8)) if ok] for c in caps]
  off=pi_offset(self.spec);pis=np.argmax(frame[:,off:off+3],axis=-1);out=[]
  for cap,pi in zip(caps,pis):
   out.append([r for r in range(3) if 0<=pi+r-1<=2 and 0<=cap+(pi+r-1)-1<=8])
  return out
 def action(self,rep,state,pi_actions,explore):
  frame=rep[:,-1] if self.spec.gru else rep[:,-1] if rep.ndim==3 else rep
  valid=self._valid(frame)
  if explore:
   self.eps=max(.05,1-.95*self.count/max(self.total_steps,1))
   guidance=max(.05,.5*(1-self.count/max(self.total_steps,1))) if self.spec.pi else 0
   if self.spec.pi and self.rng.random()<guidance:return np.ones(3,dtype=np.int64)
   if self.rng.random()<self.eps:return np.asarray([self.rng.choice(v) for v in valid],dtype=np.int64)
  with torch.no_grad():
   x=torch.as_tensor(rep[None],dtype=torch.float32)
   last=x[:,:,-1] if self.spec.gru else (x[:,:,-1] if x.ndim==4 else x)
   mask=mask_for(last,self.spec);return self.model.values(x,mask)[0].argmax(-1).cpu().numpy().astype(np.int64)
 def _commit(self):
  first=self.pending[0];ret=0.;disc=1.;n=0;terminal=False;ns=first[3];ng=first[4]
  for item in self.pending:
   ret+=disc*item[6];n+=1;ns,ng,terminal=item[3],item[4],item[5]
   if terminal or n>=self.spec.nstep:break
   disc*=GAMMA
  self.replay.add((first[0],first[1],first[2],ret,ns,ng,terminal,GAMMA**n),priority=max(self.replay.priorities,default=1.))
  self.pending.popleft()
 def add(self,rep,state,act,reward,nrep,nstate,done):
  self.count+=1;self.pending.append((rep.copy(),state.copy(),act.copy(),nrep.copy(),nstate.copy(),bool(done),float(reward)))
  # tuple layout adapted by _commit below to keep transition fields explicit
  if done:
   while self.pending:self._commit()
  elif len(self.pending)>=self.spec.nstep:self._commit()
  if len(self.replay)>=1000 and self.count%4==0:self.learn()
 def learn(self):
  beta=min(1.,.4+.6*self.count/max(self.total_steps,1))
  inds,batch,weights=self.replay.sample(self.rng,64,beta)
  reps,states,acts,rews,nreps,nstates,dones,discounts=zip(*batch)
  reps=torch.as_tensor(np.asarray(reps),dtype=torch.float32);nreps=torch.as_tensor(np.asarray(nreps),dtype=torch.float32)
  states=torch.as_tensor(np.asarray(states),dtype=torch.float32);nstates=torch.as_tensor(np.asarray(nstates),dtype=torch.float32)
  acts=torch.as_tensor(np.asarray(acts),dtype=torch.long);rews=torch.as_tensor(rews,dtype=torch.float32)
  dones=torch.as_tensor(dones,dtype=torch.float32);discounts=torch.as_tensor(discounts,dtype=torch.float32)
  weights=torch.as_tensor(weights,dtype=torch.float32)
  def get_mask(rep):
   frame=rep[:,:,-1] if self.spec.gru else (rep[:,:,-1] if rep.ndim==4 else rep)
   return mask_for(frame,self.spec)
  mask=get_mask(reps);nmask=get_mask(nreps)
  values=self.model.values(reps,mask);chosen=values.gather(-1,acts.unsqueeze(-1)).squeeze(-1)
  with torch.no_grad():
   na=self.model.values(nreps,nmask).argmax(-1);nt=self.target.values(nreps,nmask).gather(-1,na.unsqueeze(-1)).squeeze(-1)
   if self.spec.qmix:
    target=rews+discounts*(1-dones)*self.target.mixer(nt,nstates)
   else:target=rews[:,None]+discounts[:,None]*(1-dones[:,None])*nt
  if self.spec.qmix:
   current=self.model.mixer(chosen,states);td=current-target;losses=nn.functional.smooth_l1_loss(current,target,reduction='none')
  else:
   td=chosen-target;losses=nn.functional.smooth_l1_loss(chosen,target,reduction='none').mean(-1)
  loss=(losses*weights).mean();self.optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),10.);self.optim.step()
  if self.spec.per:self.replay.update(inds,np.abs(td.detach().cpu().numpy()).reshape(len(inds),-1).mean(-1)+1e-5)
  self.updates+=1
  if self.updates%250==0:self.target.load_state_dict(self.model.state_dict())

def run_episode(spec,seed,model=None,trainer=None,learn=False):
 env=PairV2Env(new_config());obs,_=env.reset(seed=seed)
 pi=FrozenCoordinatedPI() if spec.pi else None
 local=np.stack([obs[a] for a in env.possible_agents]).astype(np.float32)
 if pi is None:pis=np.asarray([1,1,1],dtype=np.int64)
 else:
  pd=pi.act(env);pis=np.asarray([pd[a] for a in env.possible_agents],dtype=np.int64)
 frame=make_frame(local,pis,spec);hist=deque([frame.copy() for _ in range(SEQ)],maxlen=SEQ)
 rep=np.stack(hist,axis=1).astype(np.float32)
 rews=[];track=[];changes=0
 while env.agents:
  state=base.global_context(local)
  if trainer:codes=trainer.action(rep,state,pis,learn)
  else:
   x=torch.as_tensor(rep[None],dtype=torch.float32);last=x[:,:,-1] if spec.gru else x[:,:,-1]
   mask=mask_for(last,spec)
   with torch.no_grad():codes=model.values(x,mask)[0].argmax(-1).cpu().numpy().astype(np.int64)
  direct=np.clip(pis+codes-1,0,2).astype(np.int64) if spec.pi else codes.copy()
  before=env.caps.copy();nobs,reward,_,_,_=env.step(dict(zip(env.agents,direct.tolist())))
  done=not bool(env.agents);nlocal=np.stack([nobs[a] for a in env.possible_agents]).astype(np.float32)
  if pi is None or done:npis=pis
  else:
   pd=pi.act(env);npis=np.asarray([pd[a] for a in env.possible_agents],dtype=np.int64)
  nf=make_frame(nlocal,npis,spec);nh=deque(hist,maxlen=SEQ);nh.append(nf.copy());nrep=np.stack(nh,axis=1).astype(np.float32)
  if trainer and learn:trainer.add(rep,state,codes,float(reward['cell_0']),nrep,base.global_context(nlocal),done)
  hist=nh;rep=nrep;local=nlocal;pis=npis
  rews.append(float(reward['cell_0']));track.append(float(env.last_reward_parts['tracking_cost']));changes+=int(before!=env.caps)
 return {'reward_x1000':float(np.mean(rews)*1000),'nrmse':float(np.sqrt(np.mean(track))),
         'mean_wip':float(env.metrics()['mean_wip_component_equiv']),'cap_adjustment_points':changes}

def summarize(rows):
 o={'n':len(rows)}
 for k in ('reward_x1000','nrmse','mean_wip','cap_adjustment_points'):
  a=np.asarray([x[k] for x in rows],float);o[k+'_mean']=float(a.mean());o[k+'_sd']=float(a.std(ddof=1)) if len(a)>1 else 0.
 return o
def write_csv(path,fields,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def train_job(job):
 key,seed,output,max_ep=job;spec=SPEC[key];seed_all(seed)
 root=Path(output)/'formal'/key/f'run_{seed}';root.mkdir(parents=True,exist_ok=True)
 if (root/'complete.json').exists():return json.loads((root/'complete.json').read_text())
 trainer=TeamTrainer(spec,seed,MIN_EPISODES*360);rows=[];curves=[];valids=[]
 best_r=-float('inf');best_n=float('inf');best_ep=0;stale=0;converged=False;t0=time.monotonic()
 for ep in range(max_ep):
  es=31_000_000+(seed-13101)*1000+ep
  m=run_episode(spec,es,trainer=trainer,learn=True);m['episode']=ep+1;rows.append(m)
  if (ep+1)%8==0:curves.append({'episode':ep+1,'train_reward_x1000':float(np.mean([x['reward_x1000'] for x in rows[-8:]])),'train_nrmse':float(np.mean([x['nrmse'] for x in rows[-8:]])),'epsilon':trainer.eps,'seconds':time.monotonic()-t0})
  if (ep+1)%EVAL_INTERVAL:continue
  ev=[run_episode(spec,s,model=trainer.model.eval()) for s in VALIDATION_SEEDS]
  sm=summarize(ev);rec={'episode':ep+1,'validation_reward_x1000':sm['reward_x1000_mean'],'validation_reward_sd':sm['reward_x1000_sd'],'validation_nrmse':sm['nrmse_mean'],'validation_nrmse_sd':sm['nrmse_sd'],'validation_mean_wip':sm['mean_wip_mean'],'validation_cap_adjustment_points':sm['cap_adjustment_points_mean'],'epsilon':trainer.eps,'seconds':time.monotonic()-t0}
  valids.append(rec);trainer.model.train()
  better=rec['validation_reward_x1000']>best_r+MIN_REWARD_GAIN
  tie=abs(rec['validation_reward_x1000']-best_r)<=MIN_REWARD_GAIN and rec['validation_nrmse']<best_n-.0002
  if better or tie:
   best_r=rec['validation_reward_x1000'];best_n=rec['validation_nrmse'];best_ep=ep+1;stale=0
   torch.save(trainer.model.state_dict(),root/'best_model.pt')
  else:stale+=1
  # Durable progress at every validation checkpoint.
  write_csv(root/'training.csv',list(rows[0]),rows)
  write_csv(root/'learning_curve.csv',list(curves[0]),curves)
  write_csv(root/'validation.csv',list(valids[0]),valids)
  progress={'config':key,'run_seed':seed,'episode':ep+1,'max_episodes':max_ep,'converged':False,'best_episode':best_ep,'best_validation_reward_x1000':best_r,'stale_validations':stale,'elapsed_seconds':time.monotonic()-t0}
  (root/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
  print(json.dumps(progress),flush=True)
  if ep+1>=MIN_EPISODES and stale>=PATIENCE:converged=True;break
 torch.save(trainer.model.state_dict(),root/'last_model.pt')
 result={'config':key,'label':spec.label,'run_seed':seed,'episodes_completed':len(rows),'converged':converged,'best_episode':best_ep,'best_validation_reward_x1000':best_r,'best_validation_nrmse':best_n,'training_seconds':time.monotonic()-t0,'checkpoint':str((root/'best_model.pt').resolve())}
 (root/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 return result

def load_model(spec,path):
 m=PolicyNet(spec);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));return m.eval()
class QEnsemble(nn.Module):
 def __init__(self,models):super().__init__();self.models=nn.ModuleList(models)
 def values(self,rep,mask):return torch.stack([m.values(rep,mask) for m in self.models]).mean(0)

def _evaluate_group(job):
 torch.set_num_threads(1)
 key,method,run_seed,out_path=job;spec=SPEC[key];out=Path(out_path)
 if method=='single':
  model=load_model(spec,out/'formal'/key/f'run_{run_seed}'/'best_model.pt')
 else:
  roots=[out/'formal'/key/f'run_{s}' for s in TRAIN_SEEDS]
  model=QEnsemble([load_model(spec,p/'best_model.pt') for p in roots]).eval()
 rows=[]
 for split,seeds in [('test',TEST_SEEDS),('validation_audit',VALIDATION_SEEDS)]:
  for seed in seeds:
   item=run_episode(spec,seed,model=model)
   item.update(config=key,label=spec.label,method=method,split=split,
               seed=seed,run_seed=run_seed if method=='single' else -1)
   rows.append(item)
 return rows

def evaluate_all(out):
 allrows=[];summaries=[];member_manifests=[]
 jobs=[]
 for spec in SPECS:
  for seed in TRAIN_SEEDS:jobs.append((spec.key,'single',seed,str(out)))
  jobs.append((spec.key,'q_ensemble_5',-1,str(out)))
 with ProcessPoolExecutor(max_workers=5) as pool:
  futures=[pool.submit(_evaluate_group,j) for j in jobs]
  for f in as_completed(futures):
   allrows.extend(f.result())
   (out/'evaluation_progress.json').write_text(json.dumps({'completed_groups':sum(x.done() for x in futures),'total_groups':len(futures),'rows':len(allrows)},indent=2),encoding='utf-8')
 for spec in SPECS:
  roots=[out/'formal'/spec.key/f'run_{s}' for s in TRAIN_SEEDS]
  manifests=[json.loads((p/'complete.json').read_text()) for p in roots]
  member_manifests.extend(manifests)
  for split in ('test','validation_audit'):
   singles=[x for x in allrows if x['config']==spec.key and x['method']=='single' and x['split']==split]
   pooled=summarize(singles);runmeans=[]
   for s in TRAIN_SEEDS:runmeans.append(summarize([x for x in singles if x['run_seed']==s]))
   pooled['training_run_mean']={k:float(np.mean([z[k+'_mean'] for z in runmeans])) for k in ('reward_x1000','nrmse','mean_wip','cap_adjustment_points')}
   pooled['training_run_sd']={k:float(np.std([z[k+'_mean'] for z in runmeans],ddof=1)) for k in ('reward_x1000','nrmse','mean_wip','cap_adjustment_points')}
   summaries.append({'config':spec.key,'label':spec.label,'method':'single','split':split,**pooled})
   ers=[x for x in allrows if x['config']==spec.key and x['method']=='q_ensemble_5' and x['split']==split]
   summaries.append({'config':spec.key,'label':spec.label,'method':'q_ensemble_5','split':split,**summarize(ers)})
 fields=['config','label','method','split','run_seed','seed','reward_x1000','nrmse','mean_wip','cap_adjustment_points']
 write_csv(out/'evaluation_per_episode.csv',fields,allrows)
 (out/'comparison_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
 (out/'training_manifest.json').write_text(json.dumps(member_manifests,indent=2),encoding='utf-8')

def main():
 p=argparse.ArgumentParser();p.add_argument('--max-episodes',type=int,default=MAX_EPISODES);p.add_argument('--workers',type=int,default=5);p.add_argument('--phase',choices=['formal','evaluate'],default='formal');p.add_argument('--output',default=str(OUT));args=p.parse_args()
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 config={'min_episodes':MIN_EPISODES,'max_episodes':args.max_episodes,'validation_interval':EVAL_INTERVAL,'patience':PATIENCE,'minimum_reward_improvement_x1000':MIN_REWARD_GAIN,'train_seeds':TRAIN_SEEDS,'validation_seeds':VALIDATION_SEEDS,'test_seeds':TEST_SEEDS,'test_fixed_before_training':True,'configs':[asdict(s) for s in SPECS]}
 (out/'protocol.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
 if args.phase=='formal':
  jobs=[(spec.key,seed,str(out),args.max_episodes) for seed in TRAIN_SEEDS for spec in SPECS]
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
   futures=[pool.submit(train_job,j) for j in jobs]
   for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
 evaluate_all(out)
 print('EXPERIMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
