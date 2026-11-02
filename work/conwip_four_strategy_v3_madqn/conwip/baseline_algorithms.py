"""Fixed-cap, reconstructed PairV2 PI, independent MADQN and joint DQN."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import copy
import math
import random
import numpy as np
import torch
from torch import nn
from .pair_v2 import PairV2Env, PairV2Config


def new_config():
    return PairV2Config(initial_caps=(3, 3, 2), inventory_weight=.05,
                        action_weight=.005, demand_gated=False)


@dataclass
class PiParameters:
    wip_factor: float = 1.15
    kp: float = 6.0
    ki: float = .25
    local_gain: float = .5
    action_deadband: float = .5
    error_filter: float = .5
    integral_limit: float = 24.0


class FrozenCoordinatedPI:
    """Reconstruction of the archived PI settings for the three PairV2 cells.

    The historical source file and implementation formula were not provided.
    This controller applies its published parameter values to a workload
    allocation that uses the documented branch and assembly processing times.
    """
    def __init__(self, p=None):
        self.p = p or PiParameters()
        self.integral = self.filtered = 0.0

    def act(self, env):
        th, ref = env.rolling_metrics()
        error = float(np.clip((ref-th)/max(ref, 1.), -1., 1.))
        self.filtered += self.p.error_filter*(error-self.filtered)
        dt = min(env.cfg.decision_minutes, env.cfg.horizon_minutes-(env.sim.now-env.stats_start))
        integ = float(np.clip(self.integral+self.filtered*max(dt,0.)/60.,
                              -self.p.integral_limit, self.p.integral_limit))
        loads = np.asarray([sum(env.current.means[0]), sum(env.current.means[1]),
                            sum(env.current.means[2])], dtype=float)
        target_wip = self.p.wip_factor*ref*loads/60.
        base = target_wip.sum()
        desired_total = float(np.clip(base+self.p.kp*self.filtered+self.p.ki*integ, 0., 24.))
        desired = desired_total*loads/loads.sum()
        ba, bb = (len(b) for b in env.buffers)
        pressure = np.asarray([bb/3-ba/3, ba/3-bb/3,
                               min(1., ba/3,bb/3)-.5*any(s==2 for stage in env.status[2] for s in stage)], dtype=float)
        pressure -= pressure.mean()
        desired = np.clip(desired+self.p.local_gain*pressure, 0., 8.)
        desired_total = float(np.clip(desired_total, 0., 24.))
        for _ in range(10):
            residual=desired_total-desired.sum()
            free=(desired<8-1e-8) if residual>0 else (desired>1e-8)
            if abs(residual)<1e-8 or not free.any(): break
            desired[free] = np.clip(desired[free]+residual/free.sum(),0,8)
        if not ((desired_total>=24-1e-8 and self.filtered>0) or
                (desired_total<=1e-8 and self.filtered<0)):
            self.integral=integ
        return {a:(2 if goal-cap>=self.p.action_deadband else
                   (0 if goal-cap<=-self.p.action_deadband else 1))
                for a,goal,cap in zip(env.agents,desired,env.caps)}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)


def make_mlp(inputs, outputs, hidden=128, final=.01):
    layers=nn.Sequential(nn.Linear(inputs,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,outputs))
    for layer in layers:
        if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,math.sqrt(2));nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight,final)
    return layers


def joint_mask(state):
    # central observation consists of three concatenated 45-value cell frames.
    caps=torch.round(state[:,[1,46,91]]*8).long()
    digits=torch.arange(27,device=state.device)
    a=digits//9;b=(digits//3)%3;c=digits%3
    cell_actions=torch.stack((a,b,c),-1)
    target_caps=caps[:,None,:]+cell_actions[None,:,:]-1
    return ((target_caps>=0)&(target_caps<=8)).all(-1)


class MADQN(nn.Module):
    """Three independent local Q-networks; each receives the common team reward."""
    def __init__(self):
        super().__init__()
        self.q_nets=nn.ModuleList([make_mlp(45,3,128,1.) for _ in range(3)])

    def forward(self,obs):
        # obs is (batch, agents, local_features); disallow cap moves outside [0, 8].
        values=torch.stack([net(obs[:,i]) for i,net in enumerate(self.q_nets)],dim=1)
        caps=torch.round(obs[:,:,1]*8)
        mask=torch.stack((caps>0,torch.ones_like(caps,dtype=torch.bool),caps<8),-1)
        return values.masked_fill(~mask,-1e9)


class MADQNTrainer:
    """Independent Double-DQN learners with shared replay and team reward."""
    def __init__(self,seed,steps):
        self.rng=np.random.default_rng(seed)
        self.online=MADQN();self.target=copy.deepcopy(self.online).eval()
        self.optim=torch.optim.Adam(self.online.parameters(),lr=1e-4)
        self.buffer=deque(maxlen=100_000);self.steps=steps;self.count=0;self.eps=1.

    def action(self,obs,explore):
        valid=[]
        for ob in obs:
            cap=int(round(float(ob[1])*8))
            valid.append([a for a,ok in enumerate((cap>0,True,cap<8)) if ok])
        if explore:
            self.eps=max(.05,1-.95*self.count/max(self.steps,1))
            if self.rng.random()<self.eps:
                return np.asarray([self.rng.choice(v) for v in valid],dtype=np.int64)
        with torch.no_grad():
            q=self.online(torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0))[0]
            return q.argmax(-1).cpu().numpy()

    def add(self,obs,actions,reward,nextobs,done):
        self.buffer.append((obs.copy(),actions.copy(),float(reward),nextobs.copy(),float(done)))
        self.count+=1
        if len(self.buffer)<1000:return
        batch=[self.buffer[i] for i in self.rng.integers(len(self.buffer),size=128)]
        s,a,r,n,d=zip(*batch)
        s=torch.as_tensor(np.asarray(s),dtype=torch.float32);n=torch.as_tensor(np.asarray(n),dtype=torch.float32)
        a=torch.as_tensor(np.asarray(a),dtype=torch.long);r=torch.as_tensor(r,dtype=torch.float32);d=torch.as_tensor(d,dtype=torch.float32)
        q=self.online(s).gather(2,a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_actions=self.online(n).argmax(-1)
            next_q=self.target(n).gather(2,next_actions.unsqueeze(-1)).squeeze(-1)
            y=(r[:,None]+.99*(1-d[:,None])*next_q)
        loss=nn.functional.smooth_l1_loss(q,y)
        self.optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.online.parameters(),10.);self.optim.step()
        if self.count%1000==0:self.target.load_state_dict(self.online.state_dict())


class DQN(nn.Module):
    """Centralized 135-D observation, 27 joint cap actions."""
    def __init__(self):super().__init__();self.net=make_mlp(135,27,256,1.)
    def forward(self,x):return self.net(x).masked_fill(~joint_mask(x),-1e9)


class DQNTrainer:
    def __init__(self,seed,steps):
        self.rng=np.random.default_rng(seed)
        self.online=DQN();self.target=copy.deepcopy(self.online).eval()
        self.optim=torch.optim.Adam(self.online.parameters(),lr=1e-4)
        self.buffer=deque(maxlen=100_000);self.steps=steps;self.count=0;self.eps=1.

    def action(self,state,explore):
        valid=np.flatnonzero(joint_mask(torch.as_tensor(state).unsqueeze(0)).numpy()[0])
        if explore:
            self.eps=max(.05,1-.95*self.count/max(self.steps,1))
            if self.rng.random()<self.eps:return int(self.rng.choice(valid))
        with torch.no_grad():return int(self.online(torch.as_tensor(state).unsqueeze(0)).argmax(-1).item())

    def add(self,s,a,r,n,done):
        self.buffer.append((s.copy(),a,r,n.copy(),float(done)));self.count+=1
        if len(self.buffer)<1000:return
        batch=[self.buffer[i] for i in self.rng.integers(len(self.buffer),size=128)]
        s,a,r,n,d=zip(*batch)
        s=torch.as_tensor(np.asarray(s));n=torch.as_tensor(np.asarray(n));a=torch.as_tensor(a);r=torch.as_tensor(r);d=torch.as_tensor(d)
        q=self.online(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():y=r+.99*(1-d)*self.target(n).max(-1).values
        loss=nn.functional.smooth_l1_loss(q,y)
        self.optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.online.parameters(),10.);self.optim.step()
        if self.count%1000==0:self.target.load_state_dict(self.online.state_dict())


@dataclass
class Metrics:
    run_seed:int
    seed:int
    reward_x1000:float
    nrmse:float
    mean_wip:float
    cap_adjustment_points:int


def evaluate_episode(policy,seed,config=None,stochastic=False,learn=False,dqn=None):
    env=PairV2Env(config or new_config());obs,_=env.reset(seed=seed)
    costs=[];rewards=[];changes=0
    if dqn:
        states=[];actions=[];rews=[];nexts=[];dones=[]
    while env.agents:
        state=env.state().astype(np.float32)
        local_obs=np.stack([obs[a] for a in env.possible_agents]).astype(np.float32)
        if isinstance(policy,MADQN):
            if dqn:
                act=dqn.action(local_obs,learn)
            else:
                act=policy(torch.as_tensor(local_obs).unsqueeze(0))[0].argmax(-1).cpu().numpy()
            acts=dict(zip(env.agents,act.tolist()))
        elif isinstance(policy,DQN):
            if dqn:
                da=dqn.action(state,learn)
            else:
                with torch.no_grad():
                    da=int(policy(torch.as_tensor(state).unsqueeze(0)).argmax(-1).item())
            acts=dict(zip(env.agents,(da//9,(da//3)%3,da%3)))
        elif isinstance(policy,FrozenCoordinatedPI):acts=policy.act(env)
        elif isinstance(policy,tuple):acts={a:1+int(np.sign(g-c)) for a,g,c in zip(env.agents,policy,env.caps)}
        else:raise TypeError(type(policy))
        before=env.caps.copy();nextobs,r,_,_,_=env.step(acts);nextstate=env.state().astype(np.float32)
        done=not bool(env.agents)
        if dqn and learn:
            if isinstance(policy,MADQN):
                next_local=np.stack([nextobs[a] for a in env.possible_agents]).astype(np.float32) if not done else np.zeros_like(local_obs)
                dqn.add(local_obs,act,float(r['cell_0']),next_local,done)
            else:
                dqn.add(state,da,r['cell_0'],nextstate,done)
        obs=nextobs;rewards.append(r['cell_0']);costs.append(env.last_reward_parts['tracking_cost'])
        changes+=int(before!=env.caps)
    metrics=Metrics(-1,seed,float(np.mean(rewards)*1000),float(np.sqrt(np.mean(costs))),
                    float(env.metrics()['mean_wip_component_equiv']),changes)
    return metrics,{}


def select_fixed(jobs):
    rows=[]
    for seed,caps in jobs:
        e=PairV2Env(new_config());e.reset(seed=seed);e.set_caps(caps);score=0.
        while e.sim.now<e.cfg.warmup_minutes+e.cfg.horizon_minutes:
            end=min(e.sim.now+e.cfg.decision_minutes,e.cfg.warmup_minutes+e.cfg.horizon_minutes)
            e._advance(end);th,ref=e.rolling_metrics();score+=10*((th-ref)/max(ref,1))**2
        m=e.metrics();tracking=score/e.cfg.horizon_minutes
        rows.append(dict(seed=seed,caps=tuple(caps),tracking_mse=tracking,mean_wip=m['mean_wip_component_equiv'],
                         objective=tracking+.05*m['mean_wip_component_equiv']/38))
    return rows
