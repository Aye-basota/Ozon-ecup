"""How much does target-window survivorship conditioning move rho?

The late confirmation folds are structurally conditioned on activity inside the
organiser's eligibility window.  To size that effect we re-measure rho on the
CLEAN fold 2025-10-16 restricted to users who are active in the target window -
the same conditioning the late folds carry by construction."""
import math, json, numpy as np, pandas as pd
import frozen_pipeline as A1
COEF=np.array([0.7462560853,0.6466415685])
for c in ("2025-10-16","2025-10-30","2025-11-13","2025-12-11","2026-01-14"):
    try: p=pd.read_parquet(f"/home/claude/work/folds/fold_{c}.parquet")
    except Exception: continue
    r=p.residual.to_numpy(float); D=COEF[0]*p.u_perp_365.to_numpy(float)+COEF[1]*p.u_perp_A2.to_numpy(float)
    act=p.target_events.to_numpy()>0
    def m(mask):
        rr=r[mask]; dd=D[mask]-D[mask].mean()
        b=math.sqrt(float(np.mean(rr*rr))); co=math.sqrt(float(np.mean((rr-dd)**2)))
        return A1.correlation(dd,rr), float(np.mean((rr-dd)**2-rr*rr)), co-b, b
    a=m(np.ones(len(r),bool)); b=m(act)
    print(f"{c}  all n={len(r):>6} inact={100*(1-act.mean()):5.3f}%  "
          f"rho_all={a[0]:.6f} dMSE_all={a[1]:+.6f} baseR_all={a[3]:.6f} | "
          f"rho_active={b[0]:.6f} dMSE_active={b[1]:+.6f} baseR_active={b[3]:.6f}")
