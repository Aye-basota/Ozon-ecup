# EXP089 — JOINT_V2 Plane Resolution

## Verdict

**V2_OUT_OF_PLANE**

`JOINT_V2` does not identify the second EXP088 tomography axis.  Its old-`w`
component is below the fixed 10% RMS gate, the two-direction matrix fails both
conditioning gates, and the out-of-A1/A2 residual is material.  The old EXP088
`TOMO_PLUS/TOMO_MINUS` files are therefore invalid after the scored span update.

## Artifact and score audit

All six frozen EXP088 arrays/bundles and the four requested exact submissions
were audited.  Every CSV has 250,000 rows, unique `user_id`, exact canonical row
order, finite nonnegative predictions, and a recorded SHA256.  `JOINT_V2` was
recovered by exact SHA256 and copied to `C:\Users\Admin\Desktop\e-cup-research-clean\submissions\SUBMIT_JOINT_V2.csv` without changing
bytes; SHA256 is `211879cb1c79bbbde93d451fca5b61c521b523f989ce42bab62cd3ab87233cba`.

Detailed audit: `artifact_audit.csv`.

## Updated scored span

The manifest contains **81** actually scored files: the 67-file
canonical bank plus 14 exact scored additions.  The sent EXP075 joint, level
probe and `JOINT_V2` are present.  EXP088 PLUS/MINUS, EXP079 A040, H12_INTERP,
NEXT_AFTER_EXP069/PRIVATE_V2 and other PRE-LB files are absent.

The level probe is exactly EXP075 plus a constant in log space; its centered
increment RMS is `1.811e-16`.  It therefore changes the level
measurement but adds no centered submission direction.

Detailed manifest: `scored_span_manifest.csv`.

## JOINT_V2 decomposition

Sequential full-population decomposition of centered realized `d_v2`:

| component | RMS | energy fraction |
| --- | ---: | ---: |
| old sent span excluding unique EXP075 joint | 0.002380714704 | 0.627949% |
| unique realized EXP075 joint | 0.027132332949 | 81.561291% |
| old EXP088 tomography `w` | 0.000805277149 | 0.071846% |
| residual `e` | 0.012653440137 | 17.738915% |

`c_joint=0.468635858950`, `c_w=-0.027412154468`.  Reconstruction RMS is
`2.107e-18` and decomposition R² is
`0.822610854`.

The direct pre-submit A1/A2 fit gives `c_A1=0.332042324890` and
`c_A2=0.312702764790`.  Sent-span + A1/A2 explain
`81.606776%`; remaining RMS is `0.012884691523` and
remaining energy is `18.393224%`.

## A1/A2 plane geometry

In the unit-RMS `[joint_unique,w]` basis, the rows `[EXP075,JOINT_V2]` have
condition number `87.688580` and normalized determinant
`0.029667`.  The old-`w` component RMS divided by centered V2 RMS is
`2.680404%`.  Fixed gates require `<=20`, `>=0.10`, and `>=10%`
respectively; all fail.

The practical-plane gates also fail: explained energy is
`81.606776%` versus `98%`, and residual RMS is
`0.012884691523` versus `max(0.005, 10%*RMS) = 0.005000000000`.

## Leaderboard decoding

The full-250k Gram sanity calculation in the *realized* `[EXP075,V2]` span
predicts a diagnostic optimum at RMSLE `1.645894652126`
with MSE gain `0.000137112501` over V2.  This is
not an A1/A2 solution because V2 carries a material third direction.

With unknown public membership and a 50k finite-population sampling posterior,
the diagnostic additional gain MSE is `0.000136967615`
(`95% [0.000120521472, 0.000154282048]`), corresponding to
Delta RMSLE `-0.000041608337`
(`95% [-0.000046868231, -0.000036612235]`).  This is an
estimate/posterior, not an exact public Gram result.

## Clipping-aware optimum

No plane candidate is authorized.  For completeness, the posterior-median
realized-span diagnostic has `0` clipped rows;
its clipping-induced out-of-span energy is
`0.000000%`.  Its score is not
claimed exact and no CSV was written for it.

## Public vs private expectation

The historical A1/A2 analogue supports the original EXP075 joint, but not the
new V2 out-of-plane residual.  The plane-only V2 coefficients are evaluated on
the frozen four forward folds, 1,000 random 20/80 pseudo-public splits, and a
1,000-replicate user-cluster Poisson bootstrap.  Winner's-curse correction uses
all 81 scored variants.  The resulting verdict is
`HIGH_UNCERTAINTY_OUT_OF_PLANE_NOT_HISTORICALLY_VALIDATED`: public improvement does not identify
private improvement for the new third direction.

## Updated probes or candidate

- PLUS: `C:\Users\Admin\Desktop\e-cup-research-clean\submissions\SUBMIT_EXP089_TOMO_PLUS.csv` — SHA256 `ae07caad5b0a4120cb08b6f0c3c0df566d5a6f5feef2edaeea1dbf8bea48fdae`
- MINUS: `C:\Users\Admin\Desktop\e-cup-research-clean\submissions\SUBMIT_EXP089_TOMO_MINUS.csv` — SHA256 `ab976162e0495db3300dcdd14c96f889228139c41adf2d090a7b9518981cf59b`
- Updated axis RMS: `0.029316898183`; nominal probe RMS: `0.025`.

Exact midpoint/effective-axis arrays and all pre/post-clipping vectors are in
`updated_tomography_vectors.npz` (SHA256
`c7a1e8bda295c0bd53fd2de379c8eb112ed77312d4aac79d63ce9af0767b8880`).  No optimized plane candidate
was created.

## Final conclusion

- Old joint energy share in `JOINT_V2`: **81.5613%**.
- Old tomography-axis energy share: **0.0718%**; RMS contribution ratio **2.6804%**.
- Residual outside sent span + A1/A2: **18.3932%**, RMS **0.012885**.
- `JOINT_V2` did **not** measure the second axis and the A1/A2 plane cannot be
  solved from the existing scores alone.
- Do **not** submit the old EXP088 probes.  Use only the updated EXP089 symmetric
  pair if a further measurement is explicitly requested.

Created CSV SHA256 map:

```json
{
  "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\research\\new_directions\\EXP089_JOINT_V2_PLANE_RESOLUTION\\artifact_audit.csv": "238f76e5834f032e1a5a6c4322f32c3697818caf92bf3c3bbcebb620ea709d02",
  "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\research\\new_directions\\EXP089_JOINT_V2_PLANE_RESOLUTION\\scored_span_manifest.csv": "24a9a8321d3f4aa7378a7a8d8c427f9bdcbaac896e7c231dc0db305e84fa6890",
  "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\research\\new_directions\\EXP089_JOINT_V2_PLANE_RESOLUTION\\plane_coefficients.csv": "67927ed66240fe1f4aad6676791040a247e04ec267ba78e646644a84615ed5b2",
  "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\submissions\\SUBMIT_EXP089_TOMO_PLUS.csv": "ae07caad5b0a4120cb08b6f0c3c0df566d5a6f5feef2edaeea1dbf8bea48fdae",
  "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\submissions\\SUBMIT_EXP089_TOMO_MINUS.csv": "ab976162e0495db3300dcdd14c96f889228139c41adf2d090a7b9518981cf59b"
}
```
