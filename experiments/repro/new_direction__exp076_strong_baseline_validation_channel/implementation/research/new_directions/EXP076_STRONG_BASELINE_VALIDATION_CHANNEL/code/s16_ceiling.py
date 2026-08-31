import numpy as np, pandas as pd, json, os
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
ART=f'{H}/mnt/OZON-E-CUP/artifacts'
a1,a2,Z,names=load_all(); tl=a1.target_log.values; mask=fold_masks(a1)
zm=np.load(f'{CACHE}/z_match.npy'); uid=a1.user_id.values; cut=a1.cutoff.values
cutcode={c:i for i,c in enumerate(sorted(set(cut)))}
key=np.array([cutcode[c] for c in cut],dtype=np.int64)*10_000_000+uid.astype(np.int64)
o=np.argsort(key); ks=key[o]
def al(u2,c2,v):
    k2=np.array([cutcode[str(x)] for x in c2],dtype=np.int64)*10_000_000+u2.astype(np.int64)
    p=np.searchsorted(ks,k2); out=np.empty(len(key)); out[o[p]]=np.asarray(v,float); return out
b=np.load(f'{ART}/BTYD_STABLE_EXP051/oof_raw.npz',allow_pickle=True)
fr=np.load(f'{ART}/oof_FRESH_CONTRAST_MOE.npz',allow_pickle=True)
bl=np.load(f'{ART}/oof_BLOCK4_SAF.npz',allow_pickle=True)
F={k:al(b['user_id'],b['cutoff'],b[k]) for k in ['x','t_x','T','p_alive','expected_count_30','mu_u','rec_buy','w180_days_buy']}
F.update({'fr_'+k:al(fr['uid'],fr['cutoff'],fr[k]) for k in ['w180','rec']})
F.update({'bl_'+k:al(bl['uid'],bl['cutoff'],bl[k]) for k in ['q','nu_c','nu_f','delta_raw']})   # blk 'activity' EXCLUDED: == 1[target_events>0]
u1=a1.u_raw_365.values; u2v=a2.u_raw_A2.values; u180=a1.u_raw_180.values
uj=A_JOINT[0]*u1+A_JOINT[1]*u2v
GROUPS={'EXP075_temporal_all':[u1,u2v,u180],
        'BTYD_RFM_features':[F[k] for k in ['x','t_x','T','p_alive','expected_count_30','mu_u','rec_buy','w180_days_buy']],
        'BLOCK4_conditional_heads':[F['bl_q'],F['bl_nu_c'],F['bl_nu_f'],F['bl_delta_raw']],
        'nonlinear_recalibration':[zm**2,np.clip(zm,0,8)**3,zm*F['p_alive'],zm*F['x'],zm*F['fr_rec'],(F['x']==0).astype(float),
                                   (F['fr_rec']>np.quantile(F['fr_rec'],0.5)).astype(float)],
        'ALL_available':None}
allv=[u1,u2v,u180]+[F[k] for k in F]+[zm**2,np.clip(zm,0,8)**3,zm*F['p_alive'],zm*F['x'],zm*F['fr_rec'],(F['x']==0).astype(float)]
GROUPS['ALL_available']=allv
tr=mask[FOLDS[0]]|mask[FOLDS[1]]|mask[FOLDS[2]]; te=mask[FOLDS[3]]
out={}
for g,V in GROUPS.items():
    X=np.column_stack([np.asarray(v,float) for v in V])
    X=np.nan_to_num(X,posinf=0,neginf=0)
    sd=X.std(0); X=X[:,sd>0]; X=(X-X.mean(0))/X.std(0)
    # residualise every direction against the ensemble span on each side separately
    def resid(m):
        B=np.hstack([np.ones((m.sum(),1)),zm[m][:,None],Z[m]])
        Q,_=np.linalg.qr(B); Xm=X[m]; return Xm-Q@(Q.T@Xm), tl[m]-zm[m]
    Xtr,rtr=resid(tr); Xte,rte=resid(te)
    w=np.linalg.solve(Xtr.T@Xtr+1e-6*np.trace(Xtr.T@Xtr)/Xtr.shape[1]*np.eye(Xtr.shape[1]),Xtr.T@rtr)
    p=Xte@w
    out[g]=dict(k=X.shape[1],rho_heldout_fold4=rho_of(p,rte),
                rho_insample_fold123=rho_of(Xtr@w,rtr))
print(json.dumps(out,indent=1)); json.dump(out,open(f'{OUT}/s16_ceiling.json','w'),indent=1)
