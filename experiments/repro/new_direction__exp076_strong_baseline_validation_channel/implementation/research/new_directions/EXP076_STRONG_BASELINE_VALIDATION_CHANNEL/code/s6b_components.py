import numpy as np, pandas as pd, os, glob, json
from spanlib import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
ART=f'{H}/mnt/OZON-E-CUP/artifacts'
i=ids(); M=np.array(build_M()); S=Span(M,tol=1e-12); print('rank',S.rank)
rows=[]
for f in sorted(glob.glob(f'{ART}/ztest_*.npy')):
    n=os.path.basename(f)[6:-4]
    z=np.load(f).astype(np.float64)
    if z.shape[0]!=250000: continue
    uf=f'{ART}/uid_{n}.npy'
    if os.path.exists(uf):
        u=np.load(uf)
        if not (u==i).all():
            o=pd.Series(z,index=u).reindex(i)
            if o.isna().any(): continue
            z=o.to_numpy()
    zp=S.perp(z); zc=z-z.mean()
    rows.append(dict(component=n,perp_fraction_centered=float((zp@zp)/(zc@zc)),
                     rms_centered=float(np.sqrt((zc**2).mean()))))
d=np.load(f'{ART}/BTYD_STABLE_EXP051/test_raw.npz',allow_pickle=True)
if (d['user_id']==i).all():
    for k in ['z_strongest','z_btyd']:
        z=d[k].astype(np.float64); zp=S.perp(z); zc=z-z.mean()
        rows.append(dict(component='BTYD:'+k,perp_fraction_centered=float((zp@zp)/(zc@zc)),
                         rms_centered=float(np.sqrt((zc**2).mean()))))
df=pd.DataFrame(rows).sort_values('perp_fraction_centered')
df.to_csv(f'{OUT}/s6b_component_in_span.csv',index=False)
print(df.to_string(index=False))
