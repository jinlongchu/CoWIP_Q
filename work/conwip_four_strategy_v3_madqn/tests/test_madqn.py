import numpy as np
import torch

from conwip.baseline_algorithms import MADQN, MADQNTrainer


def test_madqn_masks_moves_beyond_cap_limits():
    model=MADQN()
    obs=torch.zeros((2,3,45),dtype=torch.float32)
    obs[0,:,1]=torch.tensor([0.,.5,1.])
    q=model(obs)
    assert q.shape==(2,3,3)
    assert q[0,0,0].item()<-1e8 and q[0,0,2].item()>-1e8
    assert q[0,1,0].item()>-1e8 and q[0,1,2].item()>-1e8
    assert q[0,2,0].item()>-1e8 and q[0,2,2].item()<-1e8


def test_madqn_trainer_emits_only_legal_cell_actions():
    trainer=MADQNTrainer(seed=1,steps=100)
    obs=np.zeros((3,45),dtype=np.float32)
    obs[:,1]=np.array([0.,.5,1.],dtype=np.float32)
    actions=trainer.action(obs,explore=True)
    assert actions.shape==(3,)
    assert actions[0] in (1,2)
    assert actions[1] in (0,1,2)
    assert actions[2] in (0,1)
    nextobs=obs.copy()
    trainer.add(obs,actions,-.1,nextobs,False)
    assert len(trainer.buffer)==1
