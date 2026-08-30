import numpy as np, pandas as pd, os, time, json
H=os.path.expanduser('~'); ART=f'{H}/mnt/OZON-E-CUP/artifacts'
E75=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS'
t0=time.time()
a1=pd.read_parquet(f'{E75}/clean_forward_predictions.parquet')
a2=pd.read_parquet(f'{E75}/a2_clean_forward_predictions.parquet')
assert (a1.user_id.values==a2.user_id.values).all() and (a1.cutoff.values==a2.cutoff.values).all()
d=np.load(f'{ART}/oof_S1-E02.npz',allow_pickle=True)
print('rowmatch user', (d['user_id']==a1.user_id.values).all(), 'cutoff',(d['cutoff']==a1.cutoff.values).all())
print('y match max abs', np.abs(d['y'].astype(np.float64)-a1.target_y30.values).max())
print('target_log vs log1p(y)', np.abs(a1.target_log.values-np.log1p(a1.target_y30.values)).max())
print('residual == target_log - baseline_z', np.abs(a1.residual.values-(a1.target_log.values-a1.baseline_z.values)).max())
print('a2 baseline identical', np.abs(a1.baseline_z.values-a2.baseline_z.values).max())
print('elapsed',time.time()-t0)
