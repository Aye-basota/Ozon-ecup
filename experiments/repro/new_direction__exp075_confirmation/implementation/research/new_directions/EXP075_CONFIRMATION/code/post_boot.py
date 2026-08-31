"""Bootstrap the POST-projection correction (the quantity actually deployed)."""
import json, math, os
import numpy as np, pandas as pd
import frozen_pipeline as A1
from hist_span import build_spans, ortho, double_project
COEF=np.array([0.7462560853,0.6466415685])
WORK="/home/claude/work"; OUT=f"{WORK}/folds"
def boot(d,r,uid,reps=1000,seed=20260828):
    uu,inv=np.unique(uid,return_inverse=True); n=len(uu)
    dd=(r-d)**2-r*r; w=np.full(len(d),1.0/len(d))
    st=np.column_stack([w,w*d,w*r,w*d*d,w*r*r,w*d*r,w*dd])
    cl=np.zeros((n,st.shape[1]))
    for j in range(st.shape[1]): cl[:,j]=np.bincount(inv,weights=st[:,j],minlength=n)
    rng=np.random.default_rng(seed); rho=np.empty(reps); dm=np.empty(reps); pos=0
    while pos<reps:
        m=min(25,reps-pos); c=rng.poisson(1.0,size=(m,n)).astype(float); s=c@cl
        sw=s[:,0]; mu=s[:,1]/sw; mr=s[:,2]/sw
        vu=s[:,3]/sw-mu*mu; vr=s[:,4]/sw-mr*mr; cv=s[:,5]/sw-mu*mr
        rho[pos:pos+m]=cv/np.sqrt(np.maximum(vu*vr,1e-300)); dm[pos:pos+m]=s[:,6]/sw; pos+=m
    return dict(rho_ci=[float(np.quantile(rho,.025)),float(np.quantile(rho,.975))],
                rho_se=float(np.std(rho,ddof=1)),
                dMSE_ci=[float(np.quantile(dm,.025)),float(np.quantile(dm,.975))],
                P_dMSE_lt_0=float(np.mean(dm<0)))
data=A1.CleanData(); rows=[]
for f in sorted(os.listdir(OUT)):
    if not f.startswith("fold_"): continue
    c=f[5:-8]; p=pd.read_parquet(f"{OUT}/{f}").sort_values("user_id")
    ids=p.user_id.to_numpy(np.int64); r=p.residual.to_numpy(float)
    D=COEF[0]*p.u_perp_365.to_numpy(float)+COEF[1]*p.u_perp_A2.to_numpy(float)
    ctx=data.context_features(data.rows(ids),A1.dt.date.fromisoformat(c))
    Sp,Sw=build_spans(ctx,p.baseline_z.to_numpy(float))
    row={"cutoff":c}
    for name,S in (("pred",Sp),("wide",Sw)):
        dp=double_project(D,ortho(S))
        b=boot(dp,r,ids)
        row[f"rho_{name}"]=A1.correlation(dp,r)
        row[f"dMSE_{name}"]=float(np.mean((r-dp)**2-r*r))
        row[f"rho_ci_{name}"]=b["rho_ci"]; row[f"dMSE_ci_{name}"]=b["dMSE_ci"]
        row[f"P_gain_{name}"]=b["P_dMSE_lt_0"]; row[f"rho_se_{name}"]=b["rho_se"]
    rows.append(row); print(json.dumps(row),flush=True)
json.dump(rows,open(f"{WORK}/post_projection_bootstrap.json","w"),indent=2)
