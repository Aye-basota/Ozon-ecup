import numpy as np, pandas as pd, os, json
from spanlib import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
E75=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS'
i=ids(); M=np.array(build_M())
Draw=np.load(f'{E75}/JOINT_A1_365_A2_TEST_raw_correction.npy').astype(np.float64)
Dc=Draw-Draw.mean(); e0=float(Dc@Dc); e0u=float(Draw@Draw)
za=np.log1p(pd.read_csv(f'{H}/mnt/Downloads/SUBMIT_ORTH_ALPHA.csv').set_index('user_id').reindex(i)['predict'].to_numpy(np.float64))
def perp_on(rows):
    A=np.atleast_2d(rows); A=A-A.mean(axis=1,keepdims=True)
    G=A@A.T; w,V=np.linalg.eigh(G); k=w>w.max()*1e-12
    Vi=V[:,k]/np.sqrt(w[k]); c=Vi.T@(A@Dc); return Dc-((Vi@c)@A)
res={}
p=perp_on(za[None,:]); res['span_{1,z_alpha}']=dict(rank=1,perp_centered=float((p@p)/e0),perp_uncentered=float((p+Draw.mean())@(p+Draw.mean())/e0u))
# nested sweep over the 78 vectors, ordered by how much they explain D
A=M-M.mean(axis=1,keepdims=True)
order=np.argsort(-np.abs(A@Dc)/np.sqrt((A*A).sum(1)))
sweep=[]
for k in [1,2,3,5,8,12,20,30,45,60,67,78]:
    rows=M[order[:k]]
    p=perp_on(rows); sweep.append(dict(k=k,perp_centered=float((p@p)/e0)))
res['nested_sweep_best_first']=sweep
p=perp_on(M); res['full_span']=dict(perp_centered=float((p@p)/e0))
# uncentered reference numbers from the stored arrays
Dp=np.load(f'{E75}/JOINT_A1_365_A2_TEST_PERP.npy').astype(np.float64)
res['stored']=dict(RMS_Draw=float(np.sqrt((Draw**2).mean())),RMS_Dperp=float(np.sqrt((Dp**2).mean())),
                   perp_fraction_uncentered=float((Dp@Dp)/(Draw@Draw)))
print(json.dumps(res,indent=1)); json.dump(res,open(f'{OUT}/s8_test_span_sweep.json','w'),indent=1)
