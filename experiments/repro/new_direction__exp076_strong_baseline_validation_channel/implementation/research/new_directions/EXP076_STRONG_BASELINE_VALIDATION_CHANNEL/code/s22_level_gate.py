import numpy as np, pandas as pd, datetime as dt, json, os
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
CONF=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_CONFIRMATION'
a1=pd.read_parquet(f'{E75}/clean_forward_predictions.parquet')
means={c:float(a1.target_log.values[a1.cutoff.values==c].mean()) for c in FOLDS}
for c in ['2025-10-30','2025-11-13','2025-12-11','2026-01-14']:
    d=pd.read_parquet(f'{CONF}/fold_{c}.parquet'); means[c]=float(d.target_log.values.mean())
ks=sorted(means); R1=1.646143314225527
w=pd.read_csv(f'{OUT}/s20_level_windows.csv')
def panel_level(start):  # all-250k mean log1p over the 30-day window starting `start`
    r=w[w.window_start==start]; return float(r.mean_log1p_y30.iloc[0]) if len(r) else np.nan
rows=[]
for i,c in enumerate(ks):
    prev=ks[i-1] if i else None
    e_freeze=means[c]-means[prev] if prev else np.nan
    # panel-informed rule: previous fold level + the all-250k panel step between the two target windows
    if prev:
        s0=(dt.date.fromisoformat(prev)+dt.timedelta(days=1)); s1=(dt.date.fromisoformat(c)+dt.timedelta(days=1))
        g0=(s0-dt.date(2025,1,1)).days; g1=(s1-dt.date(2025,1,1)).days
        p0=panel_level(str(dt.date(2025,1,1)+dt.timedelta(days=g0-(g0%7)))); p1=panel_level(str(dt.date(2025,1,1)+dt.timedelta(days=g1-(g1%7))))
        e_panel=means[c]-(means[prev]+(p1-p0))
    else: e_panel=np.nan
    rows.append(dict(cutoff=c,mean_target_log=means[c],err_freeze_prev=e_freeze,err_panel_step=e_panel,
                     dMSE_freeze=e_freeze**2 if prev else np.nan, dMSE_panel=e_panel**2 if prev else np.nan))
df=pd.DataFrame(rows); print(df.to_string(index=False))
o=dict(fold_means=means, level_error_equiv_of_remaining_gap=float(np.sqrt(2.7097878109694022-2.7048785373742925)),
       rms_err_freeze=float(np.sqrt(np.nanmean(df.err_freeze_prev.values**2))),
       rms_err_panel=float(np.sqrt(np.nanmean(df.err_panel_step.values**2))))
o['dRMSLE_if_level_off_by_rms_freeze']=o['rms_err_freeze']**2/(2*R1)
o['dRMSLE_if_level_off_by_rms_panel']=o['rms_err_panel']**2/(2*R1)
print(json.dumps(o,indent=1)); json.dump(o,open(f'{OUT}/s22_level_gate.json','w'),indent=1); df.to_csv(f'{OUT}/s22_level_gate.csv',index=False)
