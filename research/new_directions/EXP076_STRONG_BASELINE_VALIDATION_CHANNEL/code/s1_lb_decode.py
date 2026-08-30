import numpy as np, pandas as pd, json, hashlib, os
H=os.path.expanduser('~')
DL=f'{H}/mnt/Downloads'
SUB=f'{H}/mnt/e-cup-research-clean/submissions'
E75=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS'
SAMP=f'{H}/mnt/OZON-E-CUP/data/raw/sample_submit.csv'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

samp=pd.read_csv(SAMP)
ids=samp['user_id'].to_numpy()
def load(p):
    d=pd.read_csv(p)
    assert (d['user_id'].to_numpy()==ids).all(), p
    return d['predict'].to_numpy(np.float64)

pa=load(f'{DL}/SUBMIT_ORTH_ALPHA.csv')
pe=load(f'{SUB}/SUBMIT_EXP075_JOINT_A1_365_A2.csv')
za=np.log1p(pa); ze=np.log1p(pe)
d=ze-za                                   # actually applied log-space correction
Dperp=np.load(f'{E75}/JOINT_A1_365_A2_TEST_PERP.npy').astype(np.float64)

R0=1.6461597403364463; R1=1.646143314225527
res={}
res['sha_ORTH_ALPHA']=sha(f'{DL}/SUBMIT_ORTH_ALPHA.csv')
res['sha_EXP075']=sha(f'{SUB}/SUBMIT_EXP075_JOINT_A1_365_A2.csv')
clip = ze<=0
res['n_clipped']=int(clip.sum())
res['max_abs_d_minus_Dperp_free']=float(np.abs(d[~clip]-Dperp[~clip]).max())
res['RMS_Dperp_all']=float(np.sqrt((Dperp**2).mean()))
res['RMS_Dperp_free_only']=float(np.sqrt((Dperp[~clip]**2).mean()))
res['RMS_d_applied']=float(np.sqrt((d**2).mean()))
res['mean_d']=float(d.mean()); res['norm_d']=float(np.linalg.norm(d))
res['norm_Dperp']=float(np.linalg.norm(Dperp))

# ---- LB decode on the public subset (RMS of d over public unknown; use full-test value) ----
M0=R0**2; M1=R1**2
dR=R1-R0; dM=M1-M0
S=res['RMS_d_applied']                     # RMS(d)
Rr=R0                                      # RMS of baseline residual on public == its RMSLE
cov=(S**2-dM)/2.0                          # mean(d*r) implied,  r = target_log - z_alpha
rho=cov/(S*Rr)
a_opt=cov/S**2
be=S/(2*Rr)
res.update(dict(delta_RMSLE=dR, delta_MSE=dM, MSE0=M0, MSE1=M1,
                RMS_d_used=S, resid_RMS_public=Rr,
                implied_cov=cov, implied_rho=rho, implied_opt_amplitude=a_opt,
                break_even_rho_applied=be,
                break_even_rho_Dperp_all=res['RMS_Dperp_all']/(2*Rr)))
# ratios
for k,v in dict(clean_forward_joint=0.037939815873089, late_weighted_Spred=0.040508,
                late_weighted_Swide=0.029942, late_latest_Swide=0.032472,
                amplitude_matched=0.035497, break_even=be).items():
    res[f'ratio_realised_over_{k}']=rho/v
print(json.dumps(res,indent=2))
json.dump(res,open('out/s1_lb_decode.json','w'),indent=2)
np.save('out/TEST_d_applied.npy', d)
