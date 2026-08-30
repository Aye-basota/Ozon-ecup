import numpy as np, pandas as pd, os, json, glob
from spanlib import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
ART=f'{H}/mnt/OZON-E-CUP/artifacts'; i=ids()
def zt(n):
    z=np.load(f'{ART}/ztest_{n}.npy').astype(np.float64)
    uf=f'{ART}/uid_{n}.npy'
    if os.path.exists(uf):
        u=np.load(uf)
        if not (u==i).all(): z=pd.Series(z,index=u).reindex(i).to_numpy()
    return z
SEQ=['SEQ-01','SEQ-C289-S43','SEQ-C289-S44','SEQ-D3A-BASE-S42']
ETX=['ETX-01-S42-DCW','ETX-01-S43-DCW','ETX-01-S44-DCW']
TAB=['S1-CAP','S1-UNC','S1-DIST','S1-E11','RIDGE15','HOLIDAY-YOY-FAST','TIER-A-DIRECT-AVG3-R300',
     'L180_norm0_tb1','LNone_norm0_tb1','L90_norm0_tb1','S1-NORM','S04-A','S04-B']
cols={}; 
for n in SEQ+ETX+TAB:
    try: cols[n]=zt(n)
    except Exception as e: print('skip',n,e)
d=np.load(f'{ART}/BTYD_STABLE_EXP051/test_raw.npz',allow_pickle=True); assert (d['user_id']==i).all()
cols['BTYD_z']=d['z_btyd'].astype(np.float64)
names=list(cols); X=np.column_stack([np.ones(250000)]+[cols[n] for n in names])
za=np.log1p(pd.read_csv(f'{H}/mnt/Downloads/SUBMIT_ORTH_ALPHA.csv').set_index('user_id').reindex(i)['predict'].to_numpy(np.float64))
z37=d['z_strongest'].astype(np.float64)
def fit(y,X,al=1e-6):
    XtX=X.T@X; dd=np.diag(XtX).mean(); A=XtX+al*dd*np.eye(X.shape[1]); A[0,0]=XtX[0,0]
    b=np.linalg.solve(A,X.T@y); res=y-X@b
    return b,float(1-res@res/((y-y.mean())@(y-y.mean())))
out={}
for tag,y in [('SUBMIT_ORTH_ALPHA',za),('EXP037_z_strongest',z37)]:
    b,r2=fit(y,X)
    fam={'SEQ':sum(b[1+names.index(n)] for n in SEQ if n in names),
         'ETX':sum(b[1+names.index(n)] for n in ETX if n in names),
         'TAB':sum(b[1+names.index(n)] for n in TAB if n in names),
         'BTYD':b[1+names.index('BTYD_z')]}
    tot=sum(fam.values())
    out[tag]=dict(R2=r2,total_slope=float(tot),shares={k:float(v/tot) for k,v in fam.items()},
                  raw={k:float(v) for k,v in fam.items()})
    print(tag,'R2=%.6f'%r2, {k:round(v/tot,4) for k,v in fam.items()})
# how much of alpha is NOT explained by the reproducible component bank
b,_=fit(za,X); res=za-X@b
out['alpha_unexplained_rms']=float(np.sqrt((res**2).mean())); out['alpha_rms_centered']=float(np.sqrt(((za-za.mean())**2).mean()))
print('alpha unexplained rms',out['alpha_unexplained_rms'],'of centered rms',out['alpha_rms_centered'])
json.dump(out,open(f'{OUT}/s10_alpha_composition.json','w'),indent=1)
