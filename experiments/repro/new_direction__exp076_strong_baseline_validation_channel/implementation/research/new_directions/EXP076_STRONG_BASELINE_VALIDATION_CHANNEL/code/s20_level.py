"""Cheap headroom probe: is the frozen submission level right for the TEST window?
Panel-wide mean log1p(30-day GMV) for every 30-day window, straight from the gmv mmap."""
import numpy as np, pandas as pd, datetime as dt, json, os
H=os.path.expanduser('~'); OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
g=np.load(f'{H}/mnt/OZON-E-CUP/data/processed/seq_gmv_v1.npy',mmap_mode='r')
DS=dt.date(2025,1,1); N=g.shape[1]
starts=[d for d in range(0,N-29,7)]
acc=np.zeros(len(starts)); accz=np.zeros(len(starts))
CH=25000
for lo in range(0,250000,CH):
    blk=np.asarray(g[lo:lo+CH])
    cs=np.concatenate([np.zeros((blk.shape[0],1)),np.cumsum(blk,axis=1)],axis=1)
    for i,s in enumerate(starts):
        y=cs[:,s+30]-cs[:,s]
        acc[i]+=np.log1p(np.maximum(y,0)).sum(); accz[i]+=(y<=0).sum()
    del blk,cs
m=acc/250000.; z=accz/250000.
df=pd.DataFrame(dict(window_start=[str(DS+dt.timedelta(days=s)) for s in starts],
                     window_end=[str(DS+dt.timedelta(days=s+29)) for s in starts],
                     mean_log1p_y30=m, frac_zero=z))
df.to_csv(f'{OUT}/s20_level_windows.csv',index=False)
def win(a,b):
    s=(dt.date.fromisoformat(a)-DS).days
    r=df[df.window_start==str(DS+dt.timedelta(days=s))]
    return float(r.mean_log1p_y30.iloc[0]) if len(r) else None
print(df.iloc[::4].to_string(index=False))
