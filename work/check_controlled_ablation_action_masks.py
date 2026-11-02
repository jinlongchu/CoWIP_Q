import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import run_controlled_m4_ablation as exp

for key in ('full_m4', 'no_global', 'no_pi', 'original_madqn'):
    spec=exp.SPEC[key]
    for cap in range(9):
        local=np.zeros((3,45),dtype=np.float32)
        local[:,1]=cap/8
        for pi in range(3):
            frame=exp.make_frame(local,np.full(3,pi),spec)
            got=exp.mask_for(torch.as_tensor(frame[None]),spec)[0,0].tolist()
            if spec.pi:
                expected=[0<=pi+r-1<=2 and 0<=cap+(pi+r-1)-1<=8 for r in range(3)]
            else:
                expected=[cap>0,True,cap<8]
            assert got==expected,(key,cap,pi,got,expected)
print('action masks verified for 108 PI/cap combinations and direct-action boundary cases')
