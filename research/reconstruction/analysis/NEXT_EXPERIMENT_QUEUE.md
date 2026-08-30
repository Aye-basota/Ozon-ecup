# Next experiment queue

Очередь отсортирована по Expected Value. После failure threshold эксперимент закрывается; соседние веса/варианты не запускаются автоматически.

## 1. `COMPOUND-AB-FINALIZE` — materialize fixed no-training candidate

- **Exact parent:** `team_a_current:EXP-037`, SHA256 `abc2218b...e04bda`.
- **Exact change:** replace champion composition by `.95*SEQ65 + .05*BTYD`, where SEQ65 is exactly `EXP-059` (`.10/.10/.15/.325/.325`). Apply one final level shift to `2.3293`, floor, `expm1`.
- **Required artifacts:** nine `EXP-059` primitive ztest/uid arrays; `artifacts/BTYD_STABLE_EXP051/test_raw.npz`; five canonical OOF members; sample submission; component SHA manifest.
- **Runtime estimate:** 2–5 min CPU, <1 GB RAM, ~5 MB output.
- **Success threshold:** independent OOF delta `<=−0.00045`, 4/4 folds, max OOF reconstruction error `<=5e-7`, test/OOF correction variance `[0.6,1.4]`, exact 250k UID order.
- **Failure threshold:** delta `>−0.00030`, any losing fold, hash/alignment mismatch, or variance outside gate.
- **What we learn:** whether the best missing compatibility edge survives an independent implementation and is submission-ready.

## 2. `COMPOUND-ABC-FRESH-PROD` — deterministic FRESH production support

- **Exact parent:** frozen output of queue item 1; OOF parent `EXP-037 + EXP-059 + EXP-051`.
- **Exact change:** add only saved `EXP-040` FRESH−CLEAN semantics: deterministic two-sided encoder, GLOBAL correction, alpha=1, donor-safe winsorization, centering, depth clip 289.
- **Required artifacts:** `oof_FRESH_CONTRAST_MOE.npz`, `EXP-040` configs/code/tests, original sequence training data, five encoder trajectories, 30 conditional-head fits, queue-1 manifest.
- **Runtime estimate:** 4–8 GPU-hours + 1–2 CPU-hours; 5–15 GB.
- **Success threshold:** rebuilt FRESH delta within `±0.00005` of `−0.000224956`, triple incremental `<=−0.00010`, ≥3/4 folds, production variance `[0.6,1.4]`.
- **Failure threshold:** parity miss, incremental `>−0.00010`, <3 winning folds, or support/variance failure.
- **What we learn:** whether the OOF-only third microgain is realizable on TEST without trajectory drift.

## 3. `TEAMMATE-CANONICAL-OOF-GATE` — one locked exploration rebuild

- **Exact parent:** canonical `team_a_current:EXP-037` row keys/folds; direction 1 may be evaluated only after standalone common-OOF reconstruction.
- **Exact change:** rebuild row-level OOF for exactly `occ_meta_B` and `occ_raw_X3`; evaluate raw X3, meta B and recorded `.12 friend + .16 meta + .72 raw` only.
- **Required artifacts:** teammate fixedstack/final6h scripts, configs, raw/meta feature matrices, checkpoint bank, `EXP-037` OOF target/row-key hashes.
- **Runtime estimate:** 6–10 compute hours; 10–30 GB.
- **Success threshold:** exact 770,616-row alignment, positive fold-1 delta, ≥3/4 folds, fixed or nested blend gain `<=−0.00030`, test/OOF variance `[0.6,1.4]`.
- **Failure threshold:** missing canonical state, fold 1 not improved, gain `>−0.00030`, or only public-selected benefit.
- **What we learn:** whether the externally strong teammate line contains genuine residual signal or only incomplete-fold/repeated-selection optimism.

## Stop rules for the whole queue

- No LB-driven neighbor search after item 1.
- No new FRESH alpha/segment/architecture search after item 2.
- No broad Ridge/meta search before item 3 reconstructs common OOF.
- If items 2 and 3 fail, research should stop rather than return to saturated calibration, generic residual mapping, rounds/seed tuning or another hurdle variant.
