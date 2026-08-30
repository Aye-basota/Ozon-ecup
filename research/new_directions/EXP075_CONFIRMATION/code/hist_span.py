"""Historical analogue of the TEST submission-span projection.

The real TEST correction is projected out of a 78-vector / rank-67 span of full
TEST submissions.  No historical analogue of those files exists, so we rebuild
the largest reproducible equivalent from cutoff-safe quantities:

  S_pred  (10 dims, closest in character to the real bank - every column is a
           *prediction of y30 in log space*):
             clean baseline z; log1p(s*expm1(z)) for s in .64/.97/1.20/1.40
             (the canonical bank contains many such rescaled variants);
             naive-30, naive-90/3, naive-180*30/180, naive-365*30/365 levels;
             a recency-only predictor.
  S_wide  (S_pred + 44 dims): log1p of every channel's 30/90/365-day sum and
           every channel's recency - a deliberately over-aggressive stress span.

Reported per fold, before and after projection: rho with the residual,
Delta MSE at the deployed unit amplitude, retained energy fraction, and the
correlation of the removed component with the residual.
"""
from __future__ import annotations
import datetime as dt, json, math, os, sys
import numpy as np, pandas as pd
import frozen_pipeline as A1

WORK="/home/claude/work"; OUT=f"{WORK}/folds"
UP="/mnt/user-data/uploads/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
COEF=np.array([0.7462560853,0.6466415685])
WIN=A1.WINDOWS; NCH=A1.NCH
GMV=A1.RAW_CHANNELS.index("gmv"); TOORD=A1.RAW_CHANNELS.index("to_ord")

def ctx_cols(ctx):
    """slice context_features layout: [sum_w, cnt_w] * 7 windows, then recency(11), avail(1)"""
    out={}
    p=0
    for w in WIN:
        out[("sum",w)]=ctx[:,p:p+NCH]; p+=NCH
        out[("cnt",w)]=ctx[:,p:p+NCH]; p+=NCH
    out["rec"]=ctx[:,p:p+NCH]; p+=NCH
    return out

def build_spans(ctx, baseline_z):
    c=ctx_cols(ctx)
    g30=c[("sum",30)][:,GMV]; g90=c[("sum",90)][:,GMV]
    g180=c[("sum",180)][:,GMV]; g365=c[("sum",365)][:,GMV]
    recg=c["rec"][:,GMV]
    # the panel stores log1p(gmv) per day, so the window "sum" is a log-sum level proxy
    pred=[baseline_z]
    for s in (0.64,0.97,1.20,1.40):
        pred.append(np.log1p(s*np.expm1(np.maximum(baseline_z,0.0))))
    pred += [g30, g90/3.0, g180*30.0/180.0, g365*30.0/365.0,
             -np.log1p(recg)]
    S_pred=np.column_stack(pred)
    wide=[S_pred]
    for w in (30,90,365):
        wide.append(c[("sum",w)]); wide.append(c[("cnt",w)])
    wide.append(c["rec"])
    S_wide=np.column_stack(wide)
    return S_pred, S_wide

def ortho(S):
    X=np.asarray(S,dtype=np.float64).copy()
    X-=X.mean(axis=0)
    g=X.T@X
    eig,vec=np.linalg.eigh(g)
    th=float(eig.max()*1e-12); keep=eig>th
    Q=X@(vec[:,keep]/np.sqrt(eig[keep]))
    Q,_=np.linalg.qr(Q,mode="reduced")
    return Q

def double_project(d,Q):
    c=np.asarray(d,float)-float(np.mean(d))
    perp=c-Q@(Q.T@c)
    perp=perp-Q@(Q.T@perp)
    perp-=perp.mean()
    return perp

def metrics(d,r,tag):
    base=math.sqrt(float(np.mean(r*r))); cor=math.sqrt(float(np.mean((r-d)**2)))
    return {f"rho_{tag}":A1.correlation(d,r), f"dMSE_{tag}":float(np.mean((r-d)**2-r*r)),
            f"dRMSLE_{tag}":cor-base, f"rms_{tag}":float(np.sqrt(np.mean(d*d)))}

def run(cutoff_str, ids, r, u1, u2, data, res):
    cutoff=dt.date.fromisoformat(cutoff_str)
    rows=data.rows(ids)
    ctx=data.context_features(rows,cutoff)
    # replay a clean baseline_z is not needed: use the fold's own stored baseline_z
    bz=res.pop("_bz")
    Sp,Sw=build_spans(ctx,bz)
    D=COEF[0]*u1+COEF[1]*u2
    row={"cutoff":cutoff_str,"n":int(len(ids))}
    row.update(metrics(D,r,"before"))
    for name,S in (("pred",Sp),("wide",Sw)):
        Q=ortho(S)
        dp=double_project(D,Q)
        par=(D-D.mean())-dp
        row[f"span_{name}_rank"]=int(Q.shape[1])
        row.update({k.replace("_after",f"_after_{name}"):v for k,v in metrics(dp,r,"after").items()})
        row[f"perp_fraction_{name}"]=float(np.mean(dp*dp)/np.mean((D-D.mean())**2))
        row[f"corr_removed_resid_{name}"]=A1.correlation(par,r)
        row[f"max_proj2_{name}"]=float(np.max(np.abs(Q.T@dp)))
    return row

def main():
    data=A1.CleanData()
    rows_out=[]
    # original EXP075 folds from stored OOF
    a1=pd.read_parquet(f"{UP}/clean_forward_predictions.parquet")
    a2=pd.read_parquet(f"{UP}/a2_clean_forward_predictions.parquet")
    d=a1.merge(a2[["user_id","cutoff","u_perp_A2"]],on=["user_id","cutoff"],validate="one_to_one")
    for c in ["2025-09-04","2025-09-18","2025-10-02","2025-10-16"]:
        p=d[d.cutoff==c].sort_values("user_id")
        row=run(c,p.user_id.to_numpy(np.int64),p.residual.to_numpy(float),
                p.u_perp_365.to_numpy(float),p.u_perp_A2.to_numpy(float),data,
                {"_bz":p.baseline_z.to_numpy(float)})
        row["source"]="EXP075_original"
        rows_out.append(row); A1.log(json.dumps(row))
    # new folds
    for f in sorted(os.listdir(OUT)):
        if not f.startswith("fold_"): continue
        c=f[5:-8]
        p=pd.read_parquet(f"{OUT}/{f}").sort_values("user_id")
        row=run(c,p.user_id.to_numpy(np.int64),p.residual.to_numpy(float),
                p.u_perp_365.to_numpy(float),p.u_perp_A2.to_numpy(float),data,
                {"_bz":p.baseline_z.to_numpy(float)})
        row["source"]="audit_new_fold"
        rows_out.append(row); A1.log(json.dumps(row))
    df=pd.DataFrame(rows_out)
    df.to_csv(f"{WORK}/hist_span_projection.csv",index=False)
    print(df.to_string())

if __name__=="__main__":
    main()
