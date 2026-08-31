import numpy as np, pandas as pd, datetime as dt, json, os
H=os.path.expanduser('~'); OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
g=np.load(f'{H}/mnt/OZON-E-CUP/data/processed/seq_gmv_v1.npy',mmap_mode='r'); DS=dt.date(2025,1,1)
W={'2025_jan':('2025-01-15','2025-02-13'),'2025_feb_mar':('2025-02-14','2025-03-15'),
   'test_analogue_2025':('2025-02-14','2025-03-15'),'2026_jan':('2026-01-15','2026-02-13'),
   '2025_dec_jan':('2025-12-15','2026-01-13'),'2024_none':('2025-03-16','2025-04-14')}
idx={k:((dt.date.fromisoformat(a)-DS).days,(dt.date.fromisoformat(b)-DS).days) for k,(a,b) in W.items()}
tot={k:0.0 for k in W}; zer={k:0 for k in W}
for lo in range(0,250000,25000):
    blk=np.asarray(g[lo:lo+25000])
    for k,(s,e) in idx.items():
        y=blk[:,s:e+1].sum(1); tot[k]+=np.log1p(np.maximum(y,0)).sum(); zer[k]+=int((y<=0).sum())
    del blk
res={k:dict(window=W[k],mean_log1p=tot[k]/250000,frac_zero=zer[k]/250000) for k in W}
step=res['2025_feb_mar']['mean_log1p']-res['2025_jan']['mean_log1p']
ratio=res['2025_feb_mar']['mean_log1p']/res['2025_jan']['mean_log1p']
sub_mean=2.329907832630031   # audited mean log1p of the deployed submission
res['forecast']=dict(yoy_additive_step=step,yoy_multiplicative=ratio,
  L_2026_jan=res['2026_jan']['mean_log1p'],
  forecast_TEST_additive=res['2026_jan']['mean_log1p']+step,
  forecast_TEST_multiplicative=res['2026_jan']['mean_log1p']*ratio,
  submission_mean_log1p=sub_mean)
for tag in ['additive','multiplicative']:
    f=res['forecast'][f'forecast_TEST_{tag}']; c=f-sub_mean
    res['forecast'][f'level_error_{tag}']=c
    res['forecast'][f'dMSE_if_corrected_{tag}']=-c*c
    res['forecast'][f'dRMSLE_if_corrected_{tag}']=-c*c/(2*1.646143314225527)
print(json.dumps(res,indent=1)); json.dump(res,open(f'{OUT}/s21_level_exact.json','w'),indent=1)
