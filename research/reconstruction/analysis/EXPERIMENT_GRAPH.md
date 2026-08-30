# Experiment graph

Граф показывает causal/recipe lineage, а не просто хронологию. `delta` относится только к указанному parent и protocol.

```mermaid
flowchart LR
  subgraph A[Team A canonical strong line]
    A003[EXP-003 dense cutoffs<br/>delta -0.00697]
    A005[EXP-005 normalized long<br/>delta -0.00194]
    A006[EXP-006 S1-BEST<br/>delta -0.00993]
    A014[EXP-014 DIST mix<br/>delta -0.00071, 4/4]
    A025[EXP-025 SEQ-01<br/>honest delta -0.00106, 4/4]
    A026[EXP-026 SEQ AVG<br/>delta -0.000601, 4/4]
    A035[EXP-035 sequence slot<br/>delta -0.000575, 4/4]
    A036[EXP-036 ETX coauthor<br/>delta -0.000823, 4/4]
    A037[EXP-037 STRONGEST_CURRENT<br/>wCV 1.747510]
    A003 --> A005 --> A006 --> A014 --> A025 --> A026 --> A035 --> A036 --> A037
  end

  subgraph M[Unabsorbed micro-gains]
    A059[EXP-059 SEQ65<br/>delta -0.000238, 4/4]
    A047[EXP-047 BTYD OOF<br/>nested -0.000269, 4/4]
    A051[EXP-051 BTYD05 production<br/>fixed -0.000321, 4/4]
    A040[EXP-040 FRESH correction<br/>delta -0.000225, 4/4]
    A049[EXP-049 BTYD05+FRESH<br/>delta -0.000547, 3/3]
    C1[NEW fixed compound<br/>SEQ65+BTYD05<br/>delta -0.000563, 4/4]
    C2[NEW OOF triple<br/>SEQ65+BTYD05+FRESH<br/>delta -0.000721, 4/4]
    A037 --> A059
    A037 --> A047 --> A051
    A037 --> A040
    A047 --> A049
    A040 --> A049
    A059 --> C1
    A051 --> C1
    C1 --> C2
    A040 --> C2
  end

  subgraph F[Transfer tests that close tempting reuse]
    A017[EXP-017 rounds<br/>-0.00067 standalone]
    A018[EXP-018 seed AVG3<br/>-0.00062 standalone]
    A046[EXP-046 production factorial<br/>primary -0.000002]
    A017 --> A018 --> A046
    A046 -. no material transfer .-> A037
  end

  subgraph T[Teammate occurrence/table line]
    TF[friend = byte-identical EXP-037]
    TR[ridge/meta/occurrence searches<br/>reported delta -0.00165..-0.00182<br/>3 recent folds]
    TL[latest.csv<br/>.12 friend + .16 meta + .72 raw]
    MISS[canonical OOF missing<br/>for meta/raw]
    TF --> TR --> TL
    TL --> MISS
  end

  subgraph P[Parallel incompatible pipelines]
    S2[S2 structural count/value<br/>CV 1.76831, LB 1.66193]
    BA[Team B alt dist post-order<br/>2-fold 1.708295, LB 1.65463]
    BC[Team B core<br/>calibration transfer failed]
  end

  A037 --> TF
```

## Edge semantics

- Solid arrow: artifact-backed parent/component/recipe relation.
- `EXP-003…037` are absorbed or nested in champion; their historic deltas cannot be summed again.
- `EXP-047→051` is numerical OOF replay with a fixed production recipe, not a second independent gain.
- `EXP-059 + EXP-051 → NEW fixed compound` is the missing historical combination measured in this audit.
- `EXP-040 + EXP-047 → EXP-049` was already tested: total gain positive, interaction antagonistic.
- Dashed `EXP-046→EXP-037` denotes a falsified transfer: standalone rounds/seed gains collapse inside champion.
- Parallel-pipeline nodes have incompatible validation or missing common OOF; no additive edge is asserted.

## Champion composition and ownership

| slot | weight | first decisive evidence | status |
|---|---:|---|---|
| CAP / S1-E03a | 0.10 | `EXP-006`, retained as extrapolation insurance in `EXP-014/025` | absorbed |
| UNC / S1-E02 | 0.20 | `EXP-003/006` | absorbed |
| DIST | 0.25 | `EXP-014` | absorbed |
| ETX-AVG3 | 0.225 | `EXP-036/037` | absorbed |
| SEQ-AVG3 | 0.225 | `EXP-025/026/035/037` | absorbed |

## Compatibility frontier

The only fully pairable, production-supported frontier beyond champion is:

```text
EXP-037
  + fixed sequence-slot reweight from EXP-059
  + fixed 5% BTYD direction from EXP-051
= 1.746947164 wCV
```

FRESH is the next OOF-supported edge, but its production node is missing. Teammate occurrence/meta is a separate exploration branch until canonical OOF is restored.
