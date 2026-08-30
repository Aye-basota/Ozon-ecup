import math, json, numpy as np, sys
R0 = 1.6461597403364463          # SUBMIT_ORTH_ALPHA public LB
RT = 1.6446514942                # strong-team reference
M0 = R0*R0
RMS_DPERP = 0.05843391232565335  # RMS of the deployed TEST correction (stored artifact)
N = 250000
req_rho2 = (M0 - RT*RT)/M0
print("== Required ==")
print(f"required dRMSLE = {RT-R0:.16f}")
print(f"required dMSE   = {RT*RT-M0:.16f}")
print(f"required rho^2  = {req_rho2:.16f}")
print(f"required rho    = {math.sqrt(req_rho2):.16f}")
print()
print("== Deployed TEST amplitude ==")
print(f"RMS(D_perp)          = {RMS_DPERP:.12f}")
print(f"||D_perp||_2         = {RMS_DPERP*math.sqrt(N):.6f}  (ORTH step for comparison: 11.80)")
print(f"amplitude-matched rho= {RMS_DPERP/R0:.6f}   (rho at which the deployed size is MSE-optimal)")
be = RMS_DPERP/(2*R0)
print(f"break-even rho       = {be:.6f}   (dMSE = 0)")
print()
def dmse(rho): return -2*rho*RMS_DPERP*R0 + RMS_DPERP**2
def score(rho): return math.sqrt(M0 + dmse(rho))
def remaining(rho):
    m = M0 + dmse(rho)
    if m <= RT*RT: return 0.0
    return math.sqrt((m - RT*RT)/m)
extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
rows = [("0 (no transfer)",0.0),("break-even",be),("0.020",0.020),("0.025",0.025),
        ("0.030",0.030),("amplitude-matched 0.03550",RMS_DPERP/R0),
        ("EXP075 historical joint 0.037940",0.03793981587308912)]
for k,v in extra.items(): rows.append((k,float(v)))
print(f"{'scenario':<36}{'rho':>10}{'dMSE':>14}{'dRMSLE':>14}{'approx score':>16}{'remaining rho':>15}")
for name,rho in rows:
    print(f"{name:<36}{rho:>10.5f}{dmse(rho):>14.7f}{score(rho)-R0:>14.7f}{score(rho):>16.10f}{remaining(rho):>15.6f}")
print()
print("Reference: reaching 1.6446514942 from 1.6461597403 needs rho = %.6f against the ALPHA residual." % math.sqrt(req_rho2))
