from decimal import Decimal, getcontext
import math
getcontext().prec = 50

R0 = 1.6461597403364463
R1 = 1.6446514942

MSE0 = R0*R0
MSE1 = R1*R1
dR = R1 - R0
dM = MSE1 - MSE0
rho2 = (MSE0 - MSE1)/MSE0
rho = math.sqrt(rho2)

print(f"current RMSLE           = {R0!r}")
print(f"target  RMSLE           = {R1!r}")
print(f"required Delta RMSLE    = {dR:.16f}")
print(f"current MSE             = {MSE0:.16f}")
print(f"target  MSE             = {MSE1:.16f}")
print(f"required Delta MSE      = {dM:.16f}")
print(f"required rho^2          = {rho2:.16f}")
print(f"required rho            = {rho:.16f}")
print()
# Codex claims
joint_rho = 0.03793981587308912
joint_rho2 = joint_rho**2
print(f"Codex joint rho         = {joint_rho:.10f}  -> rho^2 = {joint_rho2:.12f}")
print(f"claimed  joint rho^2    = 0.001439429628483905  diff={joint_rho2-0.001439429628483905:.3e}")
print(f"round(0.03794)^2        = {0.03794**2:.12f}")
print()
rem2 = rho2 - joint_rho2
print(f"remaining rho^2         = {rem2:.16f}")
print(f"remaining rho           = {math.sqrt(max(rem2,0.0)):.16f}")
print(f"fraction of required rho^2 covered by joint = {joint_rho2/rho2:.6%}")
print()
# what score would joint alone give if fully realised at oracle amplitude
for name,r in [("joint 0.037940",joint_rho),("0.030",0.030),("0.025",0.025),("0.020",0.020)]:
    newmse = MSE0*(1-r*r); print(f"  {name}: newRMSLE={math.sqrt(newmse):.10f} dRMSLE={math.sqrt(newmse)-R0:+.8f}")
print()
# sanity: with old ORTH rho=0.0141 also independent
orth=0.0141
print(f"joint+ORTH indep rho^2  = {joint_rho2+orth**2:.12f}  frac={100*(joint_rho2+orth**2)/rho2:.4f}%")
print(f"remaining rho then      = {math.sqrt(max(rho2-joint_rho2-orth**2,0)):.10f}")
