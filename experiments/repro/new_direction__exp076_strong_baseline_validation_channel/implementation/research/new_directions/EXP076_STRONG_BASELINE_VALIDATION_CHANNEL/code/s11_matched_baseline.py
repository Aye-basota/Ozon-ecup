import numpy as np, pandas as pd, json, os
from common import *
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
a1,a2,Z,names=load_all(); tl=a1.target_log.values; mask=fold_masks(a1); IX={n:i for i,n in enumerate(names)}
zw=a1.baseline_z.values
u1=a1.u_raw_365.values; u2=a2.u_raw_A2.values; uj=A_JOINT[0]*u1+A_JOINT[1]*u2
CAND={'A1_365':u1,'A2_CNN':u2,'JOINT':uj}
FAM={'SEQ':['SEQ-AVG3','SEQ-D3A-AVG3','SEQ-D3A-BASE-AVG3'],
     'ETX':['ETX-AVG3','ETX-AVG2','ETX-01-S42'],
     'TAB':['S1-E03a','S1-E02','S1-DIST','S1-E11','RIDGE15','HOLIDAY-YOY-FAST','S04-A','S04-B',
            'S1-E10','SAMPLE-TB1-AVG3-R300','MHZ-FULL','GAP-E10-K5-G090-S42','SAMPLE-DENSE-S3-F422-S42-R300'],
     'BTYD':['BTYD:z_btyd']}
FROZEN={'SEQ':{'SEQ-AVG3':1.0},'ETX':{'ETX-AVG3':1.0},
        'TAB':{'S1-E03a':0.10/0.55,'S1-E02':0.20/0.55,'S1-DIST':0.25/0.55},'BTYD':{'BTYD:z_btyd':1.0}}
SH=json.load(open(f'{OUT}/s10_alpha_composition.json'))['SUBMIT_ORTH_ALPHA']['shares']
print('ORTH_ALPHA family shares',SH)
zf={}
for f,cols in FAM.items():
    v=np.full(len(tl),np.nan)
    Xf=np.hstack([np.ones((len(tl),1)),Z[:,[IX[c] for c in cols]]])
    for k,c in enumerate(FOLDS):
        m=mask[c]
        if k==0:
            v[m]=sum(w*Z[m,IX[n]] for n,w in FROZEN[f].items())
        else:
            tr=np.zeros(len(tl),bool)
            for j in range(k): tr|=mask[FOLDS[j]]
            v[m]=Xf[m]@ridge_fit(Xf[tr],tl[tr],3e-5)
    zf[f]=v
zmix=sum(SH[f]*zf[f] for f in FAM)
zmatch=np.full(len(tl),np.nan)
for k,c in enumerate(FOLDS):                 # level calibration from strictly earlier folds only
    m=mask[c]
    if k==0: off=0.0
    else:
        tr=np.zeros(len(tl),bool)
        for j in range(k): tr|=mask[FOLDS[j]]
        off=float((tl[tr]-zmix[tr]).mean())
    zmatch[m]=zmix[m]+off
zs=np.load(f'{CACHE}/z_strong.npy'); z037=Z[:,IX['BTYD:z_strongest']]
rows=[]
for c in FOLDS:
    m=mask[c]; ones=np.ones((m.sum(),1))
    for bn,zb in [('weak_RFM',zw),('EXP037_frozen',z037),('strong_fwd_stack',zs),('ALPHA_composition_matched',zmatch)]:
        r=tl[m]-zb[m]; B=np.hstack([ones,zb[m][:,None]])
        for cn,u in CAND.items():
            up=proj_out(u[m],B)
            rows.append(dict(cutoff=c,baseline=bn,cand=cn,RMSLE=float(np.sqrt((r**2).mean())),
                             rho=rho_of(up,r),b=float(up@r/len(r)),G=float(up@up/len(r)),
                             rms_up=float(np.sqrt((up**2).mean()))))
df=pd.DataFrame(rows); df.to_csv(f'{OUT}/s11_matched.csv',index=False)
print(df[df.cand=='JOINT'].pivot(index='cutoff',columns='baseline',values=['RMSLE','rho']).to_string())
np.save(f'{CACHE}/z_match.npy',zmatch)
for f in FAM: np.save(f'{CACHE}/zf_{f}.npy',zf[f])
