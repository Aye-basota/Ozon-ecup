import numpy as np, pandas as pd, json, os
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
ART=f'{H}/mnt/OZON-E-CUP/artifacts'
a1,a2,Z,names=load_all(); mask=fold_masks(a1); uid=a1.user_id.values; cut=a1.cutoff.values
bl=np.load(f'{ART}/oof_BLOCK4_SAF.npz',allow_pickle=True)
cutcode={c:i for i,c in enumerate(sorted(set(cut)))}
key=np.array([cutcode[c] for c in cut],dtype=np.int64)*10_000_000+uid.astype(np.int64)
o=np.argsort(key); ks=key[o]
k2=np.array([cutcode[str(x)] for x in bl['cutoff']],dtype=np.int64)*10_000_000+bl['uid'].astype(np.int64)
pos=np.searchsorted(ks,k2); act=np.empty(len(key)); act[o[pos]]=bl['activity']
lab=(a1.target_y30.values>0).astype(float)
print('BLOCK4 activity == 1[target_y30>0] ?', float((act==lab).mean()))
cd=pd.read_parquet(f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_CONFIRMATION/fold_2025-10-16.parquet')
m=mask['2025-10-16']
print('vs target_events>0 on 2025-10-16:', float((act[m]==(cd.target_events.values>0)).mean()))
# public-subset RMS(d) uncertainty
d=np.load(f'{OUT}/TEST_d_applied.npy'); rng=np.random.default_rng(7)
s=[np.sqrt((d[rng.choice(250000,50000,replace=False)]**2).mean()) for _ in range(300)]
S0=float(np.sqrt((d**2).mean())); print('RMS(d) full %.9f  50k-subset sd %.3e  rel %.4f%%'%(S0,np.std(s,ddof=1),100*np.std(s,ddof=1)/S0))
RR=1.6461597403364463; dM=-5.4079735154033415e-05
lo,hi=np.percentile(s,[2.5,97.5])
for tag,S in [('point',S0),('p2.5',lo),('p97.5',hi)]:
    cov=(S**2-dM)/2; print(f'{tag:6s} RMS(d)={S:.7f} implied rho={cov/(S*RR):.6f} a*={cov/S**2:.4f} break-even={S/(2*RR):.6f}')
