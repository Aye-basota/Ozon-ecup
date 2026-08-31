import numpy as np, os, json, time
from spanlib import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
E75=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS'
t=time.time(); M=np.array(build_M()); print('M',M.shape,f'{time.time()-t:.1f}s')
S=Span(M,tol=1e-12); print('centered rank',S.rank)
Draw=np.load(f'{E75}/JOINT_A1_365_A2_TEST_raw_correction.npy').astype(np.float64)
Dp=np.load(f'{E75}/JOINT_A1_365_A2_TEST_PERP.npy').astype(np.float64)
pp=S.perp(Draw); Dc=Draw-Draw.mean()
r=dict(vectors=int(M.shape[0]),rank=S.rank,
       perp_fraction_rebuilt=float((pp@pp)/(Dc@Dc)),
       maxabs_vs_stored=float(np.abs(pp-Dp).max()),
       corr_vs_stored=float(np.corrcoef(pp,Dp)[0,1]))
print(json.dumps(r,indent=1)); json.dump(r,open(f'{OUT}/s6a_span.json','w'),indent=1)
print('total',time.time()-t)
