import numpy as np, pandas as pd, json, os
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
a1,a2,Z,names=load_all(); tl=a1.target_log.values; mask=fold_masks(a1); IX={n:i for i,n in enumerate(names)}
zw=a1.baseline_z.values; zm=np.load(f'{CACHE}/z_match.npy')
u1=a1.u_raw_365.values; u2=a2.u_raw_A2.values; uj=A_JOINT[0]*u1+A_JOINT[1]*u2
S_TEST=0.05810292133133192; RR=1.6461597403364463
rows=[]
for k,c in enumerate(FOLDS):
    m=mask[c]; n=int(m.sum()); ones=np.ones((n,1))
    rw=tl[m]-zw[m]; rs=tl[m]-zm[m]
    Bw=np.hstack([ones,zw[m][:,None]]); Bs=np.hstack([ones,zm[m][:,None]])
    Bfull=np.hstack([ones,zm[m][:,None],Z[m]])
    up_w=proj_out(uj[m],Bw); up_s=proj_out(uj[m],Bs); up_f=proj_out(uj[m],Bfull)
    rec=dict(cutoff=c,n=n,RMSLE_weak=float(np.sqrt((rw**2).mean())),RMSLE_strong=float(np.sqrt((rs**2).mean())),
             rho_weak=rho_of(up_w,rw), rho_strong=rho_of(up_s,rs), rho_strong_postproj=rho_of(up_f,rs),
             perp_frac_full=float((up_f@up_f)/(uj[m]@uj[m])),
             b=float(up_f@rs/n), G=float(up_f@up_f/n),
             b_s=float(up_s@rs/n), G_s=float(up_s@up_s/n))
    rows.append(rec)
df=pd.DataFrame(rows)
# nested amplitude: fold0 fixed at deployed 1.0; later folds use b/G pooled over strictly earlier folds
amp=[1.0]; amp_s=[1.0]
for k in range(1,4):
    amp.append(float(df.b[:k].sum()/df.G[:k].sum())); amp_s.append(float(df.b_s[:k].sum()/df.G_s[:k].sum()))
df['amp_nested']=amp; df['amp_nested_minproj']=amp_s
# delta MSE on THIS fold at the nested amplitude and at the deployed amplitude 1.0
df['dMSE_nested']=[-2*a*b+a*a*g for a,b,g in zip(df.amp_nested,df.b,df.G)]
df['dMSE_unit']  =[-2*b+g for b,g in zip(df.b,df.G)]
df['dRMSLE_nested']=df.dMSE_nested/(2*df.RMSLE_strong)
df['dRMSLE_unit']=df.dMSE_unit/(2*df.RMSLE_strong)
# TEST-scale translation: what this fold's rho implies for the deployed TEST correction at amplitude 1.0
for col in ['rho_weak','rho_strong','rho_strong_postproj']:
    df['dRMSLE_TEST_from_'+col]=(-2*df[col]*S_TEST*RR+S_TEST**2)/(2*RR)
df.to_csv(f'{OUT}/s12_sbvc_folds.csv',index=False)
pd.set_option('display.width',250)
print(df[['cutoff','n','RMSLE_weak','RMSLE_strong','rho_weak','rho_strong','rho_strong_postproj','perp_frac_full',
          'amp_nested','dMSE_unit','dRMSLE_unit','dRMSLE_nested']].to_string(index=False))
w=np.array([WEIGHTS[c] for c in FOLDS])
agg={c:float((df[c].values*w).sum()/w.sum()) for c in ['rho_weak','rho_strong','rho_strong_postproj']}
agg.update({c+'_latest':float(df[c].values[-1]) for c in ['rho_weak','rho_strong','rho_strong_postproj']})
agg['decay_strong_over_weak_wavg']=agg['rho_strong']/agg['rho_weak']
agg['decay_postproj_over_weak_wavg']=agg['rho_strong_postproj']/agg['rho_weak']
print(json.dumps(agg,indent=1)); json.dump(agg,open(f'{OUT}/s12_agg.json','w'),indent=1)
