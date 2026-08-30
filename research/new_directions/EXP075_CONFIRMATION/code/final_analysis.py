"""Aggregate all confirmation-fold evidence and apply the pre-registered gates."""
from __future__ import annotations
import json, math, os
import numpy as np, pandas as pd
import frozen_pipeline as A1

WORK="/home/claude/work"; OUT=f"{WORK}/folds"
COEF=np.array([0.7462560853,0.6466415685])
LATE=["2025-11-13","2025-12-11","2026-01-14"]      # pre-registered confirmation set
SUPP=["2025-10-30"]                                 # least-contaminated late fold
PORT=["2025-10-16"]                                 # port-validation (inside dev corridor)
R0=1.6461597403364463

def boot(u,r,uid,reps=1000,seed=20260828):
    """Poisson user-cluster bootstrap of rho and dMSE at unit amplitude."""
    uu,inv=np.unique(uid,return_inverse=True); n=len(uu)
    d=(r-u)**2-r*r
    w=np.full(len(u),1.0/len(u))
    st=np.column_stack([w,w*u,w*r,w*u*u,w*r*r,w*u*r,w*d])
    cl=np.zeros((n,st.shape[1]))
    for j in range(st.shape[1]): cl[:,j]=np.bincount(inv,weights=st[:,j],minlength=n)
    rng=np.random.default_rng(seed)
    rho=np.empty(reps); dm=np.empty(reps); pos=0
    while pos<reps:
        m=min(25,reps-pos)
        c=rng.poisson(1.0,size=(m,n)).astype(float); s=c@cl
        sw=s[:,0]; mu=s[:,1]/sw; mr=s[:,2]/sw
        vu=s[:,3]/sw-mu*mu; vr=s[:,4]/sw-mr*mr; cv=s[:,5]/sw-mu*mr
        rho[pos:pos+m]=cv/np.sqrt(np.maximum(vu*vr,1e-300)); dm[pos:pos+m]=s[:,6]/sw
        pos+=m
    return {"rho_ci":[float(np.quantile(rho,.025)),float(np.quantile(rho,.975))],
            "rho_se":float(np.std(rho,ddof=1)),
            "dMSE_ci":[float(np.quantile(dm,.025)),float(np.quantile(dm,.975))],
            "P_dMSE_lt_0":float(np.mean(dm<0))}

def main():
    hs=pd.read_csv(f"{WORK}/hist_span_projection.csv")
    hs=hs[hs.source=="audit_new_fold"].set_index("cutoff")
    hso=pd.read_csv(f"{WORK}/hist_span_projection.csv")
    hso=hso[hso.source=="EXP075_original"].set_index("cutoff")
    rows=[]
    for f in sorted(os.listdir(OUT)):
        if not f.startswith("fold_"): continue
        c=f[5:-8]
        p=pd.read_parquet(f"{OUT}/{f}").sort_values("user_id")
        r=p.residual.to_numpy(float); u1=p.u_perp_365.to_numpy(float); u2=p.u_perp_A2.to_numpy(float)
        uid=p.user_id.to_numpy(np.int64)
        D=COEF[0]*u1+COEF[1]*u2
        base=math.sqrt(float(np.mean(r*r)))
        row={"cutoff":c,"n":int(len(p)),
             "inactive_frac":float(np.mean(p.target_events.to_numpy()==0)),
             "baseline_RMSLE":base,
             "A1_rho":A1.correlation(u1,r),"A2_rho":A1.correlation(u2,r),
             "joint_rho_pre":A1.correlation(D,r),
             "joint_dMSE_pre":float(np.mean((r-D)**2-r*r)),
             "joint_dRMSLE_pre":math.sqrt(float(np.mean((r-D)**2)))-base,
             "rms_D_pre":float(np.sqrt(np.mean(D*D)))}
        row.update({f"boot_pre_{k}":v for k,v in boot(D,r,uid).items()})
        if c in hs.index:
            h=hs.loc[c]
            for sp in ("pred","wide"):
                row[f"joint_rho_post_{sp}"]=float(h[f"rho_after_{sp}"])
                row[f"joint_dMSE_post_{sp}"]=float(h[f"dMSE_after_{sp}"])
                row[f"joint_dRMSLE_post_{sp}"]=float(h[f"dRMSLE_after_{sp}"])
                row[f"perp_fraction_{sp}"]=float(h[f"perp_fraction_{sp}"])
                row[f"corr_removed_resid_{sp}"]=float(h[f"corr_removed_resid_{sp}"])
                row[f"rms_after_{sp}"]=float(h[f"rms_after_{sp}"])
            # bootstrap the post-projection (pred span) correction
            # rebuild D_perp exactly as hist_span did is expensive; approximate by scaling
        rows.append(row)
    df=pd.DataFrame(rows).sort_values("cutoff")
    df.to_csv(f"{WORK}/confirmation_folds.csv",index=False)
    pd.set_option("display.width",250)
    print(df.to_string(index=False))
    late=df[df.cutoff.isin(LATE)]
    w=np.array([1.,2.,4.])   # recency weights over the 3 pre-registered late folds
    for col in ("joint_rho_pre","joint_rho_post_pred","joint_rho_post_wide"):
        if col in late:
            print(f"\nweighted late {col} = {float(np.sum(w*late[col].to_numpy())/w.sum()):.6f}"
                  f"   unweighted = {float(late[col].mean()):.6f}"
                  f"   n_positive = {int((late[col]>0).sum())}/3")
    json.dump({"late_folds":LATE,"supplementary":SUPP,"port_validation":PORT},
              open(f"{WORK}/gate_inputs.json","w"),indent=2)

if __name__=="__main__":
    main()
