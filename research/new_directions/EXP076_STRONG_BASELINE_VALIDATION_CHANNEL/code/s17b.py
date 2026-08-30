import numpy as np,pandas as pd,datetime as dt,lightgbm as lgb,json,time,os
src=open('s17_staleness.py').read().split('MODELS=')[0]
exec(src)
CONF=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_CONFIRMATION'
rng=np.random.default_rng(11)
for e2 in ['2025-10-16','2025-11-13']:
    d2=pd.read_parquet(f'{CONF}/fold_{e2}.parquet'); rng.choice(len(d2),60000,replace=False)
ev='2026-01-14'
d=pd.read_parquet(f'{CONF}/fold_{ev}.parquet')
idx=rng.choice(len(d),60000,replace=False); idx.sort(); dd=d.iloc[idx]
rows=np.searchsorted(UID,dd.user_id.values).astype(np.int32)
t=time.time(); X=design(rows,dt.date.fromisoformat(ev)); print('design %.0fs'%(time.time()-t),flush=True)
r=dd.residual.values; zb=dd.baseline_z.values
out=[dict(eval_fold=ev,model='stored_u_raw_fresh',gap_days=0,rho=rho(dd.u_raw_365.values,r,zb)[0])]
m=lgb.Booster(model_file=f'{CONF}/models/A1_365_{ev}.txt')
a,s=rho(m.predict(X),r,zb); out.append(dict(eval_fold=ev,model=f'A1_365@{ev}',gap_days=0,rho=a,rms_perp=s))
print(json.dumps(out,indent=1),flush=True)
json.dump(out,open(f'{H}/wk/s17b.json','w'),indent=1)
