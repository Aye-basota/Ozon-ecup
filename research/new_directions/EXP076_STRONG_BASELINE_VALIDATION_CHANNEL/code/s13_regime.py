import numpy as np, pandas as pd, json, os
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
H=os.path.expanduser('~'); CONF=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_CONFIRMATION'
A=(0.7462560852846633,0.6466415684754089)
rows=[]
for c in ['2025-10-16','2025-10-30','2025-11-13','2025-12-11','2026-01-14']:
    d=pd.read_parquet(f'{CONF}/fold_{c}.parquet')
    r=d.residual.values; u=A[0]*d.u_raw_365.values+A[1]*d.u_raw_A2.values
    B=np.column_stack([np.ones(len(d)),d.baseline_z.values])
    co,*_=np.linalg.lstsq(B,u,rcond=None); up=u-B@co
    act=d.target_events.values>0
    def rho(x,y):
        x=x-x.mean(); y=y-y.mean(); return float(x@y/np.sqrt((x@x)*(y@y)))
    rec=dict(cutoff=c,n=len(d),inactive_frac=float(1-act.mean()),
             sigma_r=float(np.sqrt((r**2).mean())), rho_weak=rho(up,r),
             rho_weak_active_only=rho(up[act],r[act]),
             sigma_r_active=float(np.sqrt((r[act]**2).mean())),
             frac_zero_target=float((d.target_y30.values==0).mean()),
             mean_target_log=float(d.target_log.values.mean()),
             rms_u=float(np.sqrt((u**2).mean())))
    rows.append(rec)
df=pd.DataFrame(rows); df.to_csv(f'{OUT}/s13_regime.csv',index=False)
pd.set_option('display.width',220); print(df.to_string(index=False))
