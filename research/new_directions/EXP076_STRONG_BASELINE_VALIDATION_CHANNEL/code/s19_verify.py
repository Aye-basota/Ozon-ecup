"""Independent re-derivation of every headline number by a different route."""
import numpy as np, pandas as pd, json, os
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
V={}
R0=1.6461597403364463; R1=1.646143314225527; RT=1.6446514942
samp=pd.read_csv(f'{H}/mnt/OZON-E-CUP/data/raw/sample_submit.csv'); ids=samp.user_id.to_numpy()
L=lambda p: np.log1p(pd.read_csv(p).set_index('user_id').reindex(ids)['predict'].to_numpy(np.float64))
za=L(f'{H}/mnt/Downloads/SUBMIT_ORTH_ALPHA.csv'); ze=L(f'{H}/mnt/e-cup-research-clean/submissions/SUBMIT_EXP075_JOINT_A1_365_A2.csv')
Dp=np.load(f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS/JOINT_A1_365_A2_TEST_PERP.npy').astype(np.float64)
V['submission_identity_max_abs_err']=float(np.abs(ze-np.maximum(za+Dp,0)).max())
d=ze-za; S=float(np.sqrt((d**2).mean()))
# (b) amplitude by 1-D search instead of the closed form
cov=(S*S-(R1**2-R0**2))/2
f=lambda a: -2*a*cov+a*a*S*S
grid=np.linspace(0,1.5,150001); V['a_star_gridsearch']=float(grid[np.argmin(f(grid))]); V['a_star_closed']=cov/S**2
V['implied_rho']=cov/(S*R0); V['break_even_rho']=S/(2*R0)
V['dRMSLE']=R1-R0; V['dMSE']=R1**2-R0**2
# mean-term sensitivity of the covariance estimate
V['mean_d']=float(d.mean()); V['rho_bias_if_mean_r_is_0.05']=float(V['mean_d']*0.05/(S*R0))
# (c) EXP075 weak-baseline rho reproduced from primary parquet
a1,a2,Z,names=load_all(); mask=fold_masks(a1); tl=a1.target_log.values
uj=A_JOINT[0]*a1.u_raw_365.values+A_JOINT[1]*a2.u_raw_A2.values
per={}
for c in FOLDS:
    m=mask[c]; zb=a1.baseline_z.values[m]; r=tl[m]-zb
    B=np.column_stack([np.ones(m.sum()),zb]); per[c]=rho_of(proj_out(uj[m],B),r)
w=np.array([WEIGHTS[c] for c in FOLDS]); V['weak_rho_per_fold']=per
V['weak_rho_recency_weighted']=float(sum(per[c]*WEIGHTS[c] for c in FOLDS)/w.sum())
V['weak_rho_pooled_oracle_reported_by_EXP075']=0.037939815873089
# (e) required rho from the new best
V['required_rho_to_target']=float(np.sqrt(R1**2-RT**2)/R1)
V['required_dRMSLE_to_target']=RT-R1
print(json.dumps(V,indent=1)); json.dump(V,open(f'{OUT}/s19_verify.json','w'),indent=1)
