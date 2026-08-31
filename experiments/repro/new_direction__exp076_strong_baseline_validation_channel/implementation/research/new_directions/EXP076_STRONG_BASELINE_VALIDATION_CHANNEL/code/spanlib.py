import numpy as np, pandas as pd, os
H=os.path.expanduser('~'); WK=f'{H}/wk'
def ids():
    return pd.read_csv(f'{H}/mnt/OZON-E-CUP/data/raw/sample_submit.csv').user_id.to_numpy()
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
def build_M():
    p=f'{WK}/Mspan.npy'
    if os.path.exists(p): return np.load(p,mmap_mode='r')
    i=ids()
    zc=np.load(f'{H}/mnt/submission_geometry_research/submission_geometry/cache/Z.npz',allow_pickle=True)
    assert (zc['user_id']==i).all()
    rows=[np.ascontiguousarray(zc['Z'])]
    ex=[]
    for pth,n in LOCAL:
        d=pd.read_csv(pth).set_index('user_id').reindex(i)
        ex.append(np.log1p(d['predict'].to_numpy(np.float64)))
    M=np.vstack([rows[0],np.array(ex)])
    M-=M.mean(axis=1,keepdims=True)
    np.save(p,M); return M
class Span:
    def __init__(s,M,tol=1e-10):
        G=M@M.T; w,V=np.linalg.eigh(G); thr=w.max()*tol
        s.keep=w>thr; s.rank=int(s.keep.sum())
        s.Vi=V[:,s.keep]/np.sqrt(w[s.keep]); s.M=M
    def coef(s,xc): return (s.Vi.T@(s.M@xc))
    def perp(s,x):
        xc=x-x.mean(); c=s.coef(xc); return xc-((s.Vi@c)@s.M)
