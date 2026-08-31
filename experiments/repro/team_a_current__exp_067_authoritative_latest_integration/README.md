# exp_067 — AUTHORITATIVE-LATEST-INTEGRATION

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_067_authoritative_latest_integration`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_067_authoritative_latest_integration`
- **Original source:** `experiments/exp_067_authoritative_latest_integration.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** `occ_raw_X3` в bundle отсутствует. Summary validation CSV не использовались
- **Known score:** Public LB `1.64921756224069` найден только в README/text provenance и там
- **Seed:** Training/model/seed отсутствуют. Runner:
- **Postprocessing:** как surrogate; `oof_latest_canonical.npz`, project wCV, сегменты и level audit
- **Submission:** LB upload/new submission: **NO**.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_067 — AUTHORITATIVE-LATEST-INTEGRATION

- **Дата:** 2026-08-25
- **Автор:** A1
- **Коммит:** `a28a71f` + working tree
- **Prefix:** `AUTHORITATIVE_LATEST_AUDIT_20260825_160144_V2`
- **Тип:** read-mostly production integration audit; training = **NONE**

## Гипотеза

Переданный pipeline второй линии может стать authoritative production state,
если финальный TEST CSV точно восстанавливается, public LB имеет надёжный
provenance, CAP lineage доказан, а все три компонента имеют совместимый
canonical four-fold OOF.

## Что изменено относительно базы

Модели и submission не менялись. Добавлен isolated audit runner, который только
читает frozen CSV/manifests/OOF и пишет результаты под уникальным prefix.

## Результат

- `latest.csv` восстановлен в `log1p`-пространстве по exact recipe
  `.12 friend + .16 occ_meta_B + .72 occ_raw_X3`, затем `z=max(z,0)`;
  post-blend level normalization отсутствует.
- 250 000 строк, exact `sample_submit.csv` order, schema `user_id,predict`;
  missing/duplicates/NaN/inf/negative = 0.
- Обязательная проверка
  `max_abs(log1p(source_csv_predict)-reconstructed_z_after_full_policy)` =
  **`8.881784197001252e-16`** при floor `5e-7` — **PASS**.
- Source SHA256 `7ef5b2c58925bd28c5bc7eb83b9cfd4785c608a0c8b2a6d7a3277730cba8e722`;
  independent reconstructed CSV SHA256
  `a9dc2dabdc693cd510c0428c501154898ec11f1e15b0694261471757cae274a1`.
  Файлы численно совпадают до `4.55e-13` raw / `8.88e-16` log, но не побайтно
  из-за CSV serialization текущего pandas writer.
- Общий SHA-manifest: 90/91 MATCH. Единственный mismatch — уже существующий
  `latest/latest_rebuilt.csv`; source latest, три компонента, sample и runner
  совпадают с manifest.
- `friend.csv` byte-identical `STRONGEST_CURRENT`, SHA256 `abc2218b...e04bda`.
  Его canonical OOF доступен: 770 616 unique `(cutoff,user_id)`, размеры
  `188518/191025/193694/197379`, target equality и order после key-alignment PASS.
- Row-level canonical OOF exact production-компонентов `occ_meta_B` и
  `occ_raw_X3` в bundle отсутствует. Summary validation CSV не использовались
  как surrogate; `oof_latest_canonical.npz`, project wCV, сегменты и level audit
  не синтезировались.
- Public LB `1.64921756224069` найден только в README/text provenance и там
  назван значением из переданного внешнего журнала. В трёх относящихся
  `RUN_MANIFEST.json` и SHA-to-score registry его нет; статус
  **`EXTERNALLY_REPORTED`**, фактическая дата LB не зафиксирована.
- Lineage expansion: shared fixed SEQ/ETX anchor 45%, original table core 6.6%,
  candidate table `occ_meta_B` 8.8%, candidate table `occ_raw_X3` 39.6%.
  Прямой fixed CAP coefficient = 1.2%, но дополнительная CAP-зависимость внутри
  learned candidate tables не сводится к доказанному scalar weight.

```text
CAP_LINEAGE = UNKNOWN
PRIVATE_SAFE_STATUS = UNRESOLVED
CANONICAL_OOF = MISSING
```

- CV: не вычислялся для `latest`; canonical component OOF неполон.
- LB upload/new submission: **NO**.

## Вердикт и вывод

**CONTINUE_PROVENANCE.** TEST assembly точно воспроизводится, но public score
не подтверждён надёжным журналом/manifest, total CAP lineage неполон, а
canonical OOF двух поздних компонентов отсутствует. Поэтому
`best_public_observed = latest (EXTERNALLY_REPORTED)`, но
`best_exactly_reproducible` и `research_private_safe_anchor` остаются
`STRONGEST_CURRENT / exp_037`; `latest` не становится CV/LOFO-базой.

## Конфиг прогона

Training/model/seed отсутствуют. Runner:
`python src/authoritative_latest_audit.py --prefix AUTHORITATIVE_LATEST_AUDIT_20260825_160144_V2`.
Focused tests: `python -m pytest src/test_authoritative_latest_audit.py -q` →
**4 passed**. Артефакты: `research/strategies/results/AUTHORITATIVE_LATEST_AUDIT_20260825_160144_V2/`
и `artifacts/AUTHORITATIVE_LATEST_AUDIT_20260825_160144_V2/`.
