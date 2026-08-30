# Compound gains

Цель этого документа — проверить accumulation hypothesis без арифметического сложения независимых карточек.

## Candidate sources

| source | exact change vs `EXP-037` | standalone/fixed delta | folds | already absorbed? | production support |
|---|---|---:|---:|---|---|
| A — `EXP-059 SEQ65` | sequence slot `0.45→0.65`; ETX:SEQ remains 50:50 | `−0.000237898` | 4/4 | no | yes, existing CSV and primitive arrays |
| B — `EXP-051 BTYD05` | `0.95 STRONGEST + 0.05 BTYD` | `−0.000320983` | 4/4 | no | yes, exact test artifact and CSV |
| C — `EXP-040 FRESH` | nested fold-safe FRESH−CLEAN correction | `−0.000224956` | 4/4 | no | **no exact production encoder/heads** |

Все три changes действуют в log-space и используют один canonical `EXP-037` anchor.

## Pairwise compatibility

### A + B: near-perfect additivity

Fixed recipe, без weight search:

```text
SEQ65 = .10 CAP + .10 UNC + .15 DIST + .325 ETX + .325 SEQ
A+B   = .95 * SEQ65 + .05 * BTYD
```

| metric | value |
|---|---:|
| A delta | `−0.000237898` |
| B delta | `−0.000320983` |
| arithmetic sum | `−0.000558881` |
| observed fixed compound | **`−0.000562699`** |
| interaction | **`−0.000003818`** |
| correction Pearson | **`−0.012633`** |
| paired user-cluster SE compound | `0.000058635` |
| test/OOF correction variance ratio | `1.216712` |

Fold deltas: `−0.000988 / −0.000795 / −0.000510 / −0.000478`.

Вывод: A и B не только меняют разные sources; их actual correction vectors почти ортогональны. Это лучший найденный compound.

### B + C: positive, but antagonistic

| metric | value |
|---|---:|
| B delta | `−0.000320983` |
| C delta | `−0.000224956` |
| arithmetic sum | `−0.000545939` |
| observed B+C | `−0.000466940` |
| interaction | `+0.000078999` |
| correction Pearson | `+0.176155` |

`EXP-049` независимо увидел ту же картину на corrected 3-fold protocol: `BTYD05_FRESH1 −0.000547`, 3/3, но interaction около `+0.000041`. Signal совместим, но частично перекрывается.

### A + B + C: stronger OOF, missing production C

| recipe | wCV | delta | folds | paired cluster SE |
|---|---:|---:|---:|---:|
| STRONGEST | 1.747509863 | — | — | — |
| A+B | 1.746947164 | `−0.000562699` | 4/4 | `0.0000586` |
| A+B+C | 1.746788684 | `−0.000721179` | 4/4 | `0.0000737` |

Increment C поверх A+B = `−0.000158480`. Arithmetic A+B+C = `−0.000783837`; observed interaction total `+0.000062658`. Triple честно сильнее fixed pair, но FRESH production inference отсутствует, поэтому это next training experiment, не готовый submission.

## LOFO weight audit

Diagnostic grid:

- sequence slot: `0.45, 0.55, 0.65, 0.75`;
- BTYD weight: `0, .025, .05, .075, .10`;
- outer fold полностью исключён из выбора пары;
- tie tolerance `1e-5` предпочитает меньшее отклонение от baseline.

Результат:

| outer fold | selected sequence | selected BTYD | held-out RMSLE |
|---|---:|---:|---:|
| 09-04 | .65 | .075 | 1.765698 |
| 09-18 | .65 | .075 | 1.759554 |
| 10-02 | .65 | .075 | 1.748117 |
| 10-16 | .75 | .10 | 1.740861 |
| nested aggregate | — | — | **1.746944278** |

Nested delta `−0.000565585` совпадает с fixed A+B `−0.000562699`. Несмотря на это, final no-training submission держит historical weights `.65/.05`; grid используется только как robustness evidence.

## Why other micro-gains are not added

| source | reported gain | exclusion reason |
|---|---:|---|
| `EXP-017/018` rounds + seed averaging | up to `−0.00133` standalone E10 | `EXP-046` exact production factorial: primary transfer `−0.0000024` |
| `EXP-042 ZERO2D` | `−0.0000248` | amount-only better; shuffled p0 identical; mechanism falsified |
| `independent_renewal:EXP-027` | replacement `−0.000416`, total `−0.000266` | occurrence classification worsened 4/4; overlaps BTYD/renewal family; production pairability absent |
| Team B core micro-gains | many `0.0001–0.001` | only 2-fold incompatible pipeline; no common OOF/test artifacts in clean pairable form |
| Team B alt `EXP-016/017` | `−0.000124/−0.000588` | different late-cutoff validation and higher scale; no common OOF with champion |
| teammate Ridge/meta gains | `−0.00165…−0.00182` vs own base | canonical OOF for final components missing; repeated-selection risk |

## Compound ladder

1. **Ready now:** A+B fixed pair, no training.
2. **Conditional next:** reproduce C exactly and add once to frozen A+B.
3. **Do not tune neighbors after LB.** A public result is only a measurement of the locked recipe, not a signal to alter weights.

## Answer

Да, путь к заметному приросту через accumulation существует. Он уже подтверждён offline: два ранее не объединённых gains дали `−0.000563`, а третий OOF source поднимает total до `−0.000721`. Это не арифметическая гипотеза: overlap, interactions, fold signs, paired uncertainty и OOF→test variance были посчитаны напрямую.
