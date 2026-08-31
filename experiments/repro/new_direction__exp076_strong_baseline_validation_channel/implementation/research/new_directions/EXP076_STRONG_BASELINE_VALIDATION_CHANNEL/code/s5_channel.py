import numpy as np, json, time, os
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
t0=time.time()
a1,a2,Z,names=load_all()
tl=a1.target_log.values; mask=fold_masks(a1)
zw=a1.baseline_z.values
zs=np.load(f'{CACHE}/z_strong.npy'); z037=np.load(f'{CACHE}/z037.npy')
u1=a1.u_raw_365.values; u2=a2.u_raw_A2.values
# frozen EXP075 joint direction, rebuilt from raw candidate outputs exactly as deployed
uj=A_JOINT[0]*u1+A_JOINT[1]*u2
CAND={'A1_365':u1,'A2_CNN':u2,'JOINT':uj}
BASE={'weak_RFM':zw,'EXP037_frozen':z037,'strong_fwd_stack':zs}
rows=[]
for c in FOLDS:
    m=mask[c]; ones=np.ones((m.sum(),1))
    Zm=Z[m]
    for bn,zb in BASE.items():
        r=tl[m]-zb[m]
        Bmin=np.hstack([ones,zb[m][:,None]])                    # EXP075-style minimal span
        Bfull=np.hstack([ones,zb[m][:,None],Zm])                # production-like ensemble span
        for cn,u in CAND.items():
            up_min=proj_out(u[m],Bmin); up_full=proj_out(u[m],Bfull)
            rec=dict(cutoff=c,baseline=bn,cand=cn,n=int(m.sum()),
                     RMSLE_base=float(np.sqrt((r**2).mean())),
                     rho_minproj=rho_of(up_min,r), rho_fullproj=rho_of(up_full,r),
                     rms_u=float(np.sqrt((u[m]**2).mean())),
                     rms_up_min=float(np.sqrt((up_min**2).mean())),
                     rms_up_full=float(np.sqrt((up_full**2).mean())),
                     b_min=float(up_min@r/len(r)), G_min=float(up_min@up_min/len(r)),
                     b_full=float(up_full@r/len(r)), G_full=float(up_full@up_full/len(r)))
            # what the projection removes, and whether it was predictive
            rem=u[m]-up_full
            rec['corr_removed_resid']=rho_of(rem,r); rec['corr_perp_resid']=rho_of(up_full,r)
            rec['perp_fraction_full']=float((up_full@up_full)/(u[m]@u[m]))
            rows.append(rec)
df=pd.DataFrame(rows)
df.to_csv(f'{OUT}/s5_channel_raw.csv',index=False)
p=df[df.cand=='JOINT'].pivot(index='cutoff',columns='baseline',values=['rho_minproj','rho_fullproj','RMSLE_base'])
print(p.to_string())
print('elapsed',time.time()-t0)
