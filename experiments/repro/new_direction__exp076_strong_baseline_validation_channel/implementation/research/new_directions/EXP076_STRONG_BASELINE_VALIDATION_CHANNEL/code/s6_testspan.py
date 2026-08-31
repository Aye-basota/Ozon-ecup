import numpy as np, pandas as pd, json, os, glob
H=os.path.expanduser('~'); ART=f'{H}/mnt/OZON-E-CUP/artifacts'
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
E75=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS'
samp=pd.read_csv(f'{H}/mnt/OZON-E-CUP/data/raw/sample_submit.csv'); ids=samp.user_id.to_numpy()
zc=np.load(f'{H}/mnt/submission_geometry_research/submission_geometry/cache/Z.npz',allow_pickle=True)
assert (zc['user_id']==ids).all()
V=[zc['Z'][i] for i in range(zc['Z'].shape[0])]; src=[f'canonical:{i}' for i in range(zc['Z'].shape[0])]
LOCAL=[(f'{H}/mnt/Downloads/SUBMIT_ORTH_ALPHA.csv','SUBMIT_ORTH_ALPHA'),
 (f'{H}/mnt/Downloads/SUBMIT_ORTH_FINAL.csv','SUBMIT_ORTH_FINAL'),
 (f'{H}/mnt/Downloads/SUBMIT_PUBLIC_EB.csv','SUBMIT_PUBLIC_EB'),
 (f'{H}/mnt/Downloads/SUBMIT_PRIVATE_OPTIMAL.csv','SUBMIT_PRIVATE_OPTIMAL'),
 (f'{H}/mnt/Downloads/SUBMIT_PRIVATE_V2.csv','SUBMIT_PRIVATE_V2'),
 (f'{H}/mnt/e-cup-research-clean/submissions/SUBMIT_NEXT_AFTER_EXP069.csv','SUBMIT_NEXT_AFTER_EXP069'),
 (f'{H}/mnt/e-cup-research-clean/submissions/PROBE_scale097.csv','PROBE_scale097'),
 (f'{H}/mnt/e-cup-research-clean/submissions/my_submit.csv','my_submit'),
 (f'{H}/mnt/e-cup-research-clean/submissions/SUBMIT_v7_newmodel.csv','SUBMIT_v7_newmodel'),
 (f'{H}/mnt/e-cup-research-clean/submissions/SUBMIT_ORTH_ROBUST_H12_INTERP.csv','SUBMIT_ORTH_ROBUST_H12_INTERP'),
 (f'{H}/mnt/e-cup-research-clean/submissions/anchor_diverse_A_combo_mlp_hurdle_w065.csv','anchor_diverse')]
for p,n in LOCAL:
    if not os.path.exists(p): print('MISSING',n); continue
    d=pd.read_csv(p).set_index('user_id').reindex(ids)
    V.append(np.log1p(d['predict'].to_numpy(np.float64))); src.append('local:'+n)
M=np.array(V)                                # (78,250000)
Mc=M-M.mean(axis=1,keepdims=True)
U,S,Vt=np.linalg.svd(Mc,full_matrices=False)
thr=S.max()*1e-8; rank=int((S>thr).sum()); Q=Vt[:rank]           # orthonormal row basis (centered)
print('vectors',M.shape[0],'centered rank',rank,'orthonorm err',float(np.abs(Q@Q.T-np.eye(rank)).max()))
def perp(x):
    xc=x-x.mean(); return xc-(Q@xc)@Q
# 1) EXP075 correction reproduction
Draw=np.load(f'{E75}/JOINT_A1_365_A2_TEST_raw_correction.npy').astype(np.float64)
Dp=np.load(f'{E75}/JOINT_A1_365_A2_TEST_PERP.npy').astype(np.float64)
pp=perp(Draw)
print('perp_fraction rebuilt %.9f  stored-vs-rebuilt maxabs %.3e'%(float((pp@pp)/(( Draw-Draw.mean())@(Draw-Draw.mean()))), float(np.abs(pp-Dp).max())))
# 2) how much of each PRODUCTION MODEL COMPONENT already lies inside the submission span
rows=[]
for f in sorted(glob.glob(f'{ART}/ztest_*.npy')):
    n=os.path.basename(f)[6:-4]
    uf=f'{ART}/uid_{n}.npy'
    z=np.load(f).astype(np.float64)
    if z.shape[0]!=250000: continue
    if os.path.exists(uf):
        u=np.load(uf); 
        if not (u==ids).all():
            o=pd.Series(z,index=u).reindex(ids); 
            if o.isna().any(): continue
            z=o.to_numpy()
    zp=perp(z); zc_=z-z.mean()
    rows.append(dict(component=n, perp_fraction=float((zp@zp)/(zc_@zc_)), rms=float(np.sqrt((zc_**2).mean()))))
df=pd.DataFrame(rows).sort_values('perp_fraction')
df.to_csv(f'{OUT}/s6_component_in_span.csv',index=False)
print(df.to_string(index=False))
np.save(f'{H}/wk/Qspan.npy',Q)
