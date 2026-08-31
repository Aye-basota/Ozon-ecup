import numpy as np, pandas as pd, os, json
H=os.path.expanduser('~'); CACHE=f'{H}/wk'
E75=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS'
FOLDS=['2025-09-04','2025-09-18','2025-10-02','2025-10-16']
WEIGHTS={'2025-09-04':1.,'2025-09-18':2.,'2025-10-02':4.,'2025-10-16':8.}
A_JOINT=(0.7462560852846633,0.6466415684754089)   # frozen EXP075 joint coefficients

def load_all():
    a1=pd.read_parquet(f'{E75}/clean_forward_predictions.parquet')
    a2=pd.read_parquet(f'{E75}/a2_clean_forward_predictions.parquet')
    Z=np.load(f'{CACHE}/Zcomp.npy'); meta=json.load(open(f'{CACHE}/Zcomp_names.json'))
    names=meta['names']
    # dedupe near-identical columns
    keep=[]; 
    for j in range(Z.shape[1]):
        dup=False
        for i in keep:
            if np.abs(Z[:,j]-Z[:,i]).max()<1e-9: dup=True; break
        if not dup: keep.append(j)
    Z=Z[:,keep]; names=[names[j] for j in keep]
    return a1,a2,Z,names

def fold_masks(a1):
    return {c:(a1.cutoff.values==c) for c in FOLDS}

def ridge_fit(X,y,alpha):
    XtX=X.T@X; d=np.diag(XtX).mean()
    A=XtX+alpha*d*np.eye(X.shape[1]); A[0,0]=XtX[0,0]        # do not penalise intercept
    return np.linalg.solve(A,X.T@y)

def proj_out(u,B):
    """remove from u the component inside span(B) (B includes a constant column)."""
    c,*_=np.linalg.lstsq(B,u,rcond=None)
    return u-B@c

def rho_of(u,r):
    uc=u-u.mean(); rc=r-r.mean()
    return float(uc@rc/np.sqrt((uc@uc)*(rc@rc)))
