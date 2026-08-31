# История production pipeline

Ниже — команды, которые создали компоненты финальной смеси. Все seed берутся из
`src/config.py` или передаются явно как 43/44 поверх базового seed 42.

## 1. Табличные компоненты

`S1-CAP` и `S1-UNC` (`exp_006`):

```bash
python -m src.predict --exp S1-UNC --variant S1-UNC --L 0 --min-history 90 --train-blocks 1
python -m src.predict --exp S1-CAP --variant S1-CAP --L 180 --min-history 90 --train-blocks 1
```

`S1-DIST` (`exp_014`, LightGBM multiclass по 16 бинам, 250 rounds):

```bash
python -m src.predict --exp S1-DIST --variant S1-DIST --L 0 --norm-long \
  --min-history 90 --step 7 --train-blocks 1 --rounds 250 --blend dist
```

Эти версии pipeline сохраняли только production-прогнозы, но не booster weights.

## 2. TCN `SEQ-AVG3`, depth clip 289

Seed 42 (`exp_025`):

```bash
python -m src.seq build
python -m pytest src/test_seq.py -q
python -m src.seq predict --exp SEQ-01 --epochs 4 --depth-clip 289
```

Checkpoint seed 42 исторически не был сохранён той версией pipeline; `ztest_SEQ-01.npy`
сохранён. Для seed 43/44 (`exp_035`) checkpoint-saving уже был включён:

```bash
SEEDS="43 44" bash pipeline/original/MIX9/run_test_models.sh
```

Фактический production-конфиг: dilated causal TCN, hidden 64, 8 блоков, kernel 3,
dropout 0.10, batch 1024, AdamW lr 3e-3 cosine, wd 1e-2, 4 эпохи, 29 cutoff'ов,
`--depth-clip 289`. Реальные логи seed 43/44 включены.

## 3. ETX `ETX-AVG3`, политика DCW

Исходное обучение seed 42 (`exp_036`), затем seed 43/44 (`exp_037`):

```bash
python -m src.etx predict --exp ETX-01-S42 --seed 42 --depth-clip 289
JOBS="T43 T44" bash pipeline/original/ETX2/run_a10.sh
```

После обучения к каждому сохранённому checkpoint применяется согласованный
inference: окно 289 дней, static depth 289 и день недели четверга (`dow-shift -1`):

```bash
python pipeline/original/ETX2/depth_fix.py --mode test \
  --ckpt ETX-01-S42-TEST --depth-clip 289 --dow-shift -1 --exp ETX-01-S42-DCW
python pipeline/original/ETX2/depth_fix.py --mode test \
  --ckpt ETX-01-S43-TEST --depth-clip 289 --dow-shift -1 --exp ETX-01-S43-DCW
python pipeline/original/ETX2/depth_fix.py --mode test \
  --ckpt ETX-01-S44-TEST --depth-clip 289 --dow-shift -1 --exp ETX-01-S44-DCW
```

Конфиг ETX: `d_model=128`, 5 блоков, 8 heads, `head_dim=16`, FFN 384,
dropout 0.10, `n_tok=192`, batch 512, chunk 128, AdamW lr 1.5e-3,
wd 1e-2, 4 эпохи, warmup 500. Seed 42 считался eager на RTX 4060 Ti,
seed 43/44 — с `--compile` на A10. Все три checkpoint'а и production-логи включены.

## 4. Гейты и сборка

Перед отправкой EXP-037 зафиксировал:

- честный LOFO слота: **-0.00092, 4/4**;
- отношение `Var(z_ETX-AVG3-z_SEQ-AVG3)` test/OOF: **0.78x**;
- готовый кандидат против прежнего чемпиона test/OOF: **0.94x**;
- wCV финальной смеси: **1.74751**.

Оригинальная сборка:

```bash
ALPHA=0.5 OUT=submission_STRONGEST_CURRENT.csv \
  bash pipeline/original/ETX2/make_submission.sh
```

Отправлено 2026-08-20. Public LB: **1.6496571**, улучшение к предыдущему
`SEQ-01-MIX` равно **-0.0005193**.

## Требования к полному переобучению

- raw data должны лежать в ожидаемых `src/config.py` путях (в пакет не включены);
- ориентир среды: Python 3.13, NumPy 2.3.4, Polars 1.43.2,
  LightGBM 4.7.0, PyTorch 2.11.0+cu126;
- Linux + CUDA нужен для `--compile`; на Windows этот флаг убрать;
- полный ETX EXP-037: 10 прогонов, две GPU, около 4.7 часа wall-clock.
