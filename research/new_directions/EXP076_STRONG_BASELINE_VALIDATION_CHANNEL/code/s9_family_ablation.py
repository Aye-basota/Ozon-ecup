import numpy as np, pandas as pd, json, os
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
a1,a2,Z,names=load_all(); tl=a1.target_log.values; mask=fold_masks(a1)
zw=a1.baseline_z.values
u1=a1.u_raw_365.values; u2=a2.u_raw_A2.values; uj=A_JOINT[0]*u1+A_JOINT[1]*u2
IX={n:i for i,n in enumerate(names)}
SEQFAM=[n for n in names if n.startswith(('SEQ','ETX','S04','FRESH','PT-','BLOCK4'))]
TABFAM=[n for n in names if n.startswith(('S1-','GAP','SAMPLE','RIDGE','MHZ','HOLIDAY','BTYD:z_btyd'))]
SETS={'none (weak RFM only)':[], 'tabular_only':TABFAM, 'sequence_only':SEQFAM, 'all':TABFAM+SEQFAM}
print('tabular:',len(TABFAM),'sequence:',len(SEQFAM))
rows=[]
for c in FOLDS:
    m=mask[c]; ones=np.ones((m.sum(),1)); r_ref=None
    for sn,cols in SETS.items():
        # baseline = ridge stack of weak RFM + this family, fit on strictly earlier folds (fold 0: pooled-honest -> use weak only)
        idx=[IX[n] for n in cols]
        Xall=np.hstack([np.ones((len(tl),1)),zw[:,None]]+([Z[:,idx]] if idx else []))
        k=FOLDS.index(c)
        if k==0: w=None
        else:
            tr=np.zeros(len(tl),bool)
            for j in range(k): tr|=mask[FOLDS[j]]
            w=ridge_fit(Xall[tr],tl[tr],3e-5)
        zb = zw[m] if w is None else Xall[m]@w
        r=tl[m]-zb
        B=np.hstack([ones,zb[:,None]])
        up=proj_out(uj[m],B)
        rows.append(dict(cutoff=c,baseline_set=sn,k_cols=len(cols),
                         RMSLE=float(np.sqrt((r**2).mean())),rho=rho_of(up,r),
                         perp_frac=float((up@up)/(uj[m]@uj[m]))))
df=pd.DataFrame(rows)
df.to_csv(f'{OUT}/s9_family_ablation.csv',index=False)
print(df.pivot(index='cutoff',columns='baseline_set',values=['RMSLE','rho']).to_string())
