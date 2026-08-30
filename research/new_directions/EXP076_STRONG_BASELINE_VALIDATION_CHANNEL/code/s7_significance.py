import numpy as np, json, os, time
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
rng=np.random.default_rng(20260828)
a1,a2,Z,names=load_all()
tl=a1.target_log.values; mask=fold_masks(a1)
zs=np.load(f'{CACHE}/z_strong.npy')
u1=a1.u_raw_365.values; u2=a2.u_raw_A2.values; uj=A_JOINT[0]*u1+A_JOINT[1]*u2
m=mask['2025-10-16']; ones=np.ones((m.sum(),1))
B=np.hstack([ones,zs[m][:,None],Z[m]])
up=proj_out(uj[m],B); r=tl[m]-zs[m]
# kurtosis inflation factor of the product u*r
kap=float((up**2*r**2).mean()/((up**2).mean()*(r**2).mean()))
S=0.05810292133133192; Rr=1.6461597403364463; NPUB=50000; NTOT=250000
out=dict(kappa_product=kap, n_hist=int(m.sum()),
         SE_rho_gauss=1/np.sqrt(NPUB), SE_rho_kappa=float(np.sqrt(kap/NPUB)),
         SE_rho_pub_minus_full=float(np.sqrt(kap/NPUB*(1-NPUB/NTOT))),
         SE_rho_pub_minus_priv=float(np.sqrt(kap*NTOT/(NPUB*(NTOT-NPUB)))))
# direct subsample simulation of a 50k rho estimate on real (u_perp, r) pairs
sub=[]
idx=np.arange(m.sum())
for _ in range(400):
    s=rng.choice(idx,NPUB,replace=False)
    sub.append(rho_of(up[s],r[s]))
sub=np.array(sub); out['sim_rho_mean']=float(sub.mean()); out['sim_rho_sd']=float(sub.std(ddof=1))
out['sim_full_rho']=rho_of(up,r)
# scaled: SD of the deviation of a 50k estimate from the population value
out['SE_50k_empirical']=float(sub.std(ddof=1)/np.sqrt(1-NPUB/m.sum()))*np.sqrt(1-NPUB/NTOT)
rho_obs=0.017930726124852213
for nm,h in [('H0_zero',0.0),('old_clean_forward',0.037939815873089),('old_late_Swide',0.029942),
             ('new_weighted',0.0070938),('new_latest',0.004970),('break_even',S/(2*Rr))]:
    out[f'z_vs_{nm}']=float((rho_obs-h)/out['SE_50k_empirical'])
out['dMSE_obs']=-5.4079735154033415e-05
out['SE_dMSE']=float(2*S*Rr*out['SE_50k_empirical'])
out['z_dMSE_lt_0']=float(out['dMSE_obs']/out['SE_dMSE'])
print(json.dumps(out,indent=1)); json.dump(out,open(f'{OUT}/s7_significance.json','w'),indent=1)
