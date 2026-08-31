import numpy as np, pandas as pd, json, os
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
ART=f'{H}/mnt/OZON-E-CUP/artifacts'
a1,a2,Z,names=load_all(); tl=a1.target_log.values; mask=fold_masks(a1)
zm=np.load(f'{CACHE}/z_match.npy'); uid=a1.user_id.values; cut=a1.cutoff.values
u1=a1.u_raw_365.values; u2=a2.u_raw_A2.values; uj=A_JOINT[0]*u1+A_JOINT[1]*u2
cutcode={c:i for i,c in enumerate(sorted(set(cut)))}
key=np.array([cutcode[c] for c in cut],dtype=np.int64)*10_000_000+uid.astype(np.int64)
order=np.argsort(key); ks=key[order]
def align(u2,c2,v):
    k2=np.array([cutcode[str(x)] for x in c2],dtype=np.int64)*10_000_000+u2.astype(np.int64)
    pos=np.searchsorted(ks,k2); assert (ks[pos]==k2).all()
    out=np.empty(len(key),dtype=np.float64); out[order[pos]]=np.asarray(v,dtype=np.float64); return out
b=np.load(f'{ART}/BTYD_STABLE_EXP051/oof_raw.npz',allow_pickle=True)
fr=np.load(f'{ART}/oof_FRESH_CONTRAST_MOE.npz',allow_pickle=True)
bl=np.load(f'{ART}/oof_BLOCK4_SAF.npz',allow_pickle=True)
AB=lambda k: align(b['user_id'],b['cutoff'],b[k])
AF=lambda k: align(fr['uid'],fr['cutoff'],fr[k])
AL=lambda k: align(bl['uid'],bl['cutoff'],bl[k])
F={'btyd_x':AB('x'),'btyd_t_x':AB('t_x'),'btyd_T':AB('T'),'btyd_p_alive':AB('p_alive'),
   'btyd_ecount30':AB('expected_count_30'),'btyd_mu':AB('mu_u'),'btyd_rec_buy':AB('rec_buy'),
   'btyd_w180buy':AB('w180_days_buy'),'fresh_w180':AF('w180'),'fresh_rec':AF('rec'),
   'fresh_d_fresh':AF('d_fresh'),'fresh_d_vol':AF('d_vol'),'blk_activity':AL('activity'),
   'blk_q':AL('q'),'blk_nu_c':AL('nu_c'),'blk_nu_f':AL('nu_f'),'blk_delta_raw':AL('delta_raw')}
CAND={'EXP075_A1_365':u1,'EXP075_A2_CNN':u2,'EXP075_JOINT':uj,'EXP075_A1_180':a1.u_raw_180.values}
# feature directions + segment-wise level recalibration directions (nonlinear in the span)
for k,v in F.items(): CAND['feat:'+k]=v
CAND['nl:z^2']=zm**2; CAND['nl:z^3']=zm**3; CAND['nl:exp(z)-1']=np.expm1(np.clip(zm,0,8))
for k in ['btyd_p_alive','fresh_rec','btyd_x','blk_activity']:
    CAND[f'int:z*{k}']=zm*F[k]
for q,lab in [(0.25,'q25'),(0.5,'q50'),(0.75,'q75')]:
    thr=np.quantile(F['fresh_rec'],q); CAND[f'seg:rec>{lab}']=(F['fresh_rec']>thr).astype(float)
CAND['seg:zero_hist']=(F['btyd_x']==0).astype(float)
rows=[]
for c in ['2025-10-02','2025-10-16']:
    m=mask[c]; ones=np.ones((m.sum(),1))
    r=tl[m]-zm[m]; B=np.hstack([ones,zm[m][:,None],Z[m]])
    Bp=B@np.linalg.pinv(B.T@B)@B.T if False else None
    Q,_=np.linalg.qr(B)
    for nm,v in CAND.items():
        x=v[m].astype(np.float64)
        if not np.isfinite(x).all() or x.std()==0: continue
        xp=x-Q@(Q.T@x)
        if np.sqrt((xp**2).mean())<1e-12: rows.append(dict(cutoff=c,direction=nm,rho_postproj=0.0,note='in span')); continue
        rows.append(dict(cutoff=c,direction=nm,rho_postproj=rho_of(xp,r),
                         perp_frac=float((xp@xp)/(x-x.mean()).dot(x-x.mean()))))
df=pd.DataFrame(rows)
p=df.pivot(index='direction',columns='cutoff',values='rho_postproj').sort_values('2025-10-16',key=abs,ascending=False)
print(p.to_string()); df.to_csv(f'{OUT}/s14_headroom.csv',index=False)
