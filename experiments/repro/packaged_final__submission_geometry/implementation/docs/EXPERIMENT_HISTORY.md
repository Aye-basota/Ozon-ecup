# Team-A experiment history

This chronology summarizes found evidence without changing original verdicts.
The exhaustive table is in `experiments/README.md`; source reports live under
`experiments/team_a`, `research/new_directions` and the worktree snapshots.

## Baseline and tabular representation

- `exp_001…exp_006`: established the S1 baseline, panel rule, history-depth and
  normalized-long feature families, then produced the first strong submission.
- `exp_007/008`: radical-minimal and similarity alternatives were diverse but
  substantially worse; low prediction correlation alone was not useful.
- `exp_013/014`: two-part/hurdle and DIST heads. DIST became the durable final
  component; E11 improved its matched control but failed to transfer as a blend.
- `exp_015…020`: fold selection, calibrated weighted CV, capacity, seed and gap
  diagnostics. The 1:2:4:8 four-fold wCV protocol became canonical.

## Dense, calendar and temporal hypotheses

- `exp_021…024`: personal time, dense temporal grids, holiday/YoY and
  multi-horizon hazard. Dense supervision and hazard/count were rejected.
- Independent renewal/domain/calendar worktrees are preserved separately; their
  IDs collide with the main sequence and are therefore not renumbered.

## SEQ and ETX

- `exp_025/026`: sequence encoder and seed averaging. SEQ was valuable as an
  ensemble member rather than a superior ranker.
- `exp_027…030c`: discovered the TEST depth-support shift and fixed production at
  `depth_clip=289`; D3A was validated but not promoted over SEQ-AVG3.
- `exp_032/032b/035`: conditional-fresh/extensive variants and exact SEQ slot
  tests. The hybrid lost diversity; SEQ-AVG3 remained.
- `exp_036/037`: ETX was introduced, its depth/day-of-week inference bug was
  found through a regime gate and corrected with DCW. `exp_037` is the exact
  STRONGEST_CURRENT anchor.

## Residual search and robustness

- `exp_038…045`: funnel auxiliary labels, block residuals, fresh contrast,
  deterministic training, conditional fine-tune and buy-control. Most learned
  their auxiliary task but did not improve direct RMSLE enough.
- `exp_046…058`: tabular refresh, BTYD, selection mismatch, channel Shapley,
  burst/gap, landmark memory, late SSL, state reweighting and fingerprint. Their
  original REJECT/NO_GO conclusions remain intact.
- `exp_059…068`: level/temporal reserve submissions, occurrence revisit,
  platform/event order, latest integration and historical recency Ridge
  provenance. Missing canonical OOF/checkpoints are recorded as blockers.
- `exp_069…071`: requested Team-B B2 integration in the earlier active worktree;
  it is historical and distinct from the later final Team-B delivery vector.

## Submission geometry, ORTH and JOINT

- Submission geometry fitted and shrank the bank of scored TEST submissions in
  log space, producing `SUBMIT_v2_shrunk` (public 1.6467120) and
  `SUBMIT_NEXT_BEST` (1.6466079).
- ORTH research produced `SUBMIT_ORTH_ALPHA` (1.6461597403) and ORTH_FINAL.
- EXP069–EXP075 in `research/new_directions` evaluated BTYD/FRESH, count-value,
  ETX contrast, LWA and new out-of-span A1/A2 signals. EXP075 produced and
  confirmed `SUBMIT_EXP075_JOINT_A1_365_A2`.
- EXP076–EXP089 audited validation transport, forward stacking, level effects,
  adversarial bounds, information loss, occurrence shock and A1/A2 tomography.
  EXP089 confirms that JOINT_V2 contains a material out-of-plane component, but
  does not recover its missing primary build script.

## Team-B integration and finals

- EXP090 audited the delivered Team-B model family and preserved its code,
  validation evidence and TEST direction.
- `SUBMIT_STRONGEST55_TEAMB45` blends Team-B with the canonical EXP037 anchor.
- `SUBMIT_JOINT86_TEAMB14` blends the scored JOINT_V2 anchor with Team-B and has
  the best explicitly recorded final public score, 1.6458200196207617.
