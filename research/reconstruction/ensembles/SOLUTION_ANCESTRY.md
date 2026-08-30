# Solution ancestry and parallel research lines

This graph reconstructs lineage; it does not propose a new ensemble. Solid
edges are backed by primary reports, run artifacts, or manifests. The full
machine-readable graph is `ancestry_edges.csv`.

```mermaid
flowchart LR
  subgraph A[Team A current — tabular to sequence ensemble]
    A001[EXP-001 S1-B0]
    A003[EXP-003 dense cutoffs]
    A004[EXP-004 history-depth arms]
    A005[EXP-005 normalized-long]
    A006[EXP-006 S1-BEST]
    A014[EXP-014 DIST head]
    A025[EXP-025 SEQ-01]
    A026[EXP-026 SEQ seed average]
    A027[EXP-027 depth diagnostic]
    A035[EXP-035 clip289 sequence slot]
    A036[EXP-036 ETX coauthor]
    A037[EXP-037 STRONGEST_CURRENT]
    A047[EXP-047 BG/NBD residual]
    A051[EXP-051 stable production replay]
    A065[EXP-065 integration package]
    A067[EXP-067 latest provenance audit]
    A001 --> A003
    A001 --> A004
    A001 --> A005
    A003 --> A006
    A004 --> A006
    A005 --> A006
    A006 --> A014 --> A025 --> A026 --> A027 --> A035 --> A036 --> A037
    A037 --> A047 --> A051 --> A065
    A037 --> A065 --> A067
  end

  subgraph S2[Team A Strategy 2 — structural target pipeline]
    S209[EXP-009 hybrid QMC/FW aggregation]
    S210[EXP-010 count hurdle]
    S211[EXP-011 monetary K shrink]
    S212[EXP-012 S2-BEST]
    S2SUB[submission_strategy_2.csv]
    S209 --> S210 --> S211 --> S212 --> S2SUB
  end

  subgraph BA[Team B alternate — recency/scaling line]
    BA01[EXP-001 HGBR baseline]
    BA08[EXP-008 recency LightGBM + scale]
    BA09[EXP-009 log ensemble]
    BA11[EXP-011 dense/scale branch]
    BA12[EXP-012 two-fold alignment audit]
    BA16[EXP-016 post-order]
    BA17[EXP-017 distribution head]
    BA01 --> BA08 --> BA09 --> BA11 --> BA12
    BA11 --> BA16 --> BA17
  end

  subgraph TB[Teammate review lineage]
    T0[STRONGEST_CURRENT package]
    T1[fixedstack review run]
    T2[final6h occurrence run]
    T3[extra90 cached-meta run]
    TL[latest.csv assembly]
    T0 --> T1 --> T2 --> T3
    T0 --> TL
    T2 --> TL
  end
```

## What the principal nodes contain

- `S1-BEST` is the log-space blend of the dense-cutoff, normalized-long, and
  history-depth tabular components. Its three calibration-level submissions are
  distinct files but one model recipe.
- `S1-DIST-MIX` introduces the multiclass distribution head. `SEQ-01-MIX` then
  adds a causal TCN. EXP-026 changes seed averaging, blend weights, and the test
  depth policy together; EXP-027 is the diagnostic that separates the depth
  transfer failure from ordinary local CV.
- EXP-035 replaces the sequence slot with a three-seed clip-289 average.
  EXP-036 measures ETX both standalone and as a TCN coauthor. EXP-037 combines
  CAP/UNC/DIST with equal ETX/TCN halves inside the sequence slot and produces
  `submission_STRONGEST_CURRENT.csv`.
- EXP-047 and EXP-051 form a parallel BG/NBD residual/production branch.
  EXP-051 numerically replays EXP-047 OOF but changes the production optimizer
  and test artifacts. EXP-065 packages both EXP-037 and EXP-051; package ACCEPT
  is an integrity verdict, not a new CV improvement.
- The teammate review line starts from the preserved STRONGEST package and
  trains hurdle/occurrence experts before cached meta evaluation. `latest.csv`
  is strongly reconstructed as an artifact, but its reported LB event is not
  SHA-bound and therefore is absent from confirmed leaderboard chronology.

## Other parallel lines

- Team B core is a separate LightGBM two-decision-fold line rooted at
  `team_b_core:EXP-001`. It explores feature groups, calibration, hurdle
  variance, seasonality, and temporal blending. Its later named CSVs are absent;
  reported LB values remain report-only.
- Team B alternate is a separate HGBR/recency/scale line. Single-cutoff and
  two-fold scores are kept in different comparison classes. Its reported LB
  files are also absent from the available checkout.
- The unnumbered S04 LightGBM run is a machine-only auxiliary-supervision line.
  Its three submissions survive and their formulas were reconstructed exactly,
  but the original final-blend manifest is missing.
- Independent renewal, calendar, domain-shift, global-regime, and exact-
  anniversary branches test separate mechanisms against branch-specific
  anchors. Their local numeric IDs collide with other namespaces and are never
  used without prefixes.

## Lineage caveats

An arrow means “used as baseline, parent, component, replay source, or package
input,” not “strictly improved.” Different branches use incompatible validation
protocols. Public LB, local CV, fixed-weight ensemble CV, LOFO, simulation, and
diagnostic AUC are not placed on a common numeric scale.

