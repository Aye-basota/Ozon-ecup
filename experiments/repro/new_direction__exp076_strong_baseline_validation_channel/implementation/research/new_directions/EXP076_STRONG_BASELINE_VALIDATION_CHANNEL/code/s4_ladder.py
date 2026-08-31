import numpy as np, json, time
from common import *
t0=time.time()
a1,a2,Z,names=load_all()
print('components after dedupe:',len(names))
tl=a1.target_log.values; cut=a1.cutoff.values
mask=fold_masks(a1)
one=np.ones((len(tl),1))
X=np.hstack([one,Z])
zw=a1.baseline_z.values                                   # weak RFM baseline
z037=Z[:,names.index('BTYD:z_strongest')]                 # frozen production EXP037 blend
res={}
# choose ridge alpha on folds 0..1 -> validate on fold 2 (never uses fold 3)
best=None
for al in [3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]:
    tr=mask[FOLDS[0]]|mask[FOLDS[1]]; va=mask[FOLDS[2]]
    w=ridge_fit(X[tr],tl[tr],al); rm=float(np.sqrt(((tl[va]-X[va]@w)**2).mean()))
    if best is None or rm<best[1]: best=(al,rm)
    print(f'alpha {al:8.5f} fold2 RMSLE {rm:.6f}')
ALPHA=best[0]; print('chosen alpha',ALPHA)
zs=np.full(len(tl),np.nan)
stack={}
for k,c in enumerate(FOLDS):
    m=mask[c]
    if k==0:
        zs[m]=z037[m]; stack[c]={'source':'frozen EXP037 production blend (no earlier fold to fit on)'}
    else:
        tr=np.zeros(len(tl),bool)
        for j in range(k): tr|=mask[FOLDS[j]]
        w=ridge_fit(X[tr],tl[tr],ALPHA)
        zs[m]=X[m]@w
        stack[c]={'source':f'ridge stack fit on {FOLDS[:k]}','alpha':ALPHA,
                  'top':sorted(zip(names,w[1:].tolist()),key=lambda x:-abs(x[1]))[:8]}
rows=[]
for c in FOLDS:
    m=mask[c]
    rows.append(dict(cutoff=c,n=int(m.sum()),
        RMSLE_weak=float(np.sqrt(((tl[m]-zw[m])**2).mean())),
        RMSLE_EXP037=float(np.sqrt(((tl[m]-z037[m])**2).mean())),
        RMSLE_strong=float(np.sqrt(((tl[m]-zs[m])**2).mean()))))
df=pd.DataFrame(rows); print(df.to_string(index=False))
wv=np.array([WEIGHTS[c] for c in FOLDS])
for col in ['RMSLE_weak','RMSLE_EXP037','RMSLE_strong']:
    print(f'wCV {col:14s} {float((df[col].values*wv).sum()/wv.sum()):.6f}')
np.save(f'{CACHE}/z_strong.npy',zs); np.save(f'{CACHE}/z037.npy',z037)
json.dump({'alpha':ALPHA,'stack':stack,'folds':df.to_dict('records'),'names':names},
          open(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out/s4_ladder.json','w'),indent=1,default=str)
print('elapsed',time.time()-t0)
