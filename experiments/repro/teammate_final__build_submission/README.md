# build_submission

## Catalogue metadata

- **Catalogue ID:** `teammate_final__build_submission`
- **Namespace:** `teammate_final`
- **Experiment ID:** `build_submission`
- **Original source:** `пайплайн сокомандника/friend_original/submission_STRONGEST_CURRENT/pipeline/build_submission.py`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** STRONGEST_CURRENT rebuild
- **Model:** sequence model
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** delta = LEVEL - float(z_mix.mean())
- **Submission:** print(f"OK: {OUTPUT.relative_to(ROOT)} побитово совпадает с отправленным сабмитом")
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# build_submission

Original script: `пайплайн сокомандника/friend_original/submission_STRONGEST_CURRENT/pipeline/build_submission.py`

```python
"""Точная пересборка STRONGEST_CURRENT из сохранённых production-компонент."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "artifacts" / "predictions"
SUB_DIR = ROOT / "submission"
REFERENCE = SUB_DIR / "submission_STRONGEST_CURRENT.csv"
OUTPUT = SUB_DIR / "submission_STRONGEST_CURRENT_rebuilt.csv"
EXPECTED_SHA256 = "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda"
LEVEL = 2.3293

COMPONENTS = (
    ("S1-CAP", 0.100),
    ("S1-UNC", 0.200),
    ("S1-DIST", 0.250),
    ("SEQ-01", 0.075),
    ("SEQ-C289-S43", 0.075),
    ("SEQ-C289-S44", 0.075),
    ("ETX-01-S42-DCW", 0.075),
    ("ETX-01-S43-DCW", 0.075),
    ("ETX-01-S44-DCW", 0.075),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    weights = np.asarray([w for _, w in COMPONENTS], dtype=np.float64)
    assert abs(float(weights.sum()) - 1.0) < 1e-12

    uid_ref = None
    rows = []
    z_parts = []
    for name, weight in COMPONENTS:
        z = np.load(PRED_DIR / f"ztest_{name}.npy")
        uid = np.load(PRED_DIR / f"uid_{name}.npy")
        assert z.ndim == uid.ndim == 1 and len(z) == len(uid) == 250_000
        assert np.isfinite(z).all() and (z >= 0).all(), f"Некорректный z: {name}"
        if uid_ref is None:
            uid_ref = uid
        else:
            assert np.array_equal(uid, uid_ref), f"Другой порядок user_id: {name}"
        z_parts.append(z.astype(np.float64, copy=False))
        rows.append((name, weight, float(z.mean())))

    z_mix = np.average(np.vstack(z_parts), axis=0, weights=weights)
    delta = LEVEL - float(z_mix.mean())
    z_cal = np.maximum(z_mix + delta, 0.0)
    pred = np.maximum(np.expm1(z_cal), 0.0)

    reference = pl.read_csv(REFERENCE)
    assert reference.columns == ["user_id", "predict"] and reference.height == 250_000
    order = reference.select("user_id").with_row_index("_order")
    rebuilt = (
        pl.DataFrame({"user_id": uid_ref, "predict": pred.astype(np.float64)})
        .join(order, on="user_id", how="inner")
        .sort("_order")
        .drop("_order")
    )
    assert rebuilt.height == 250_000 and rebuilt["user_id"].n_unique() == 250_000
    assert np.isfinite(rebuilt["predict"].to_numpy()).all()
    rebuilt.write_csv(OUTPUT, float_precision=6)

    ref_pred = reference["predict"].to_numpy()
    out_pred = pl.read_csv(OUTPUT)["predict"].to_numpy()
    max_error = float(np.max(np.abs(np.log1p(ref_pred) - np.log1p(out_pred))))
    digest = sha256(OUTPUT)

    print("Компоненты:")
    for name, weight, mean_z in rows:
        print(f"  {weight:>5.3f}  {name:<20} mean(z)={mean_z:.6f}")
    print(f"delta={delta:+.8f}; mean(log1p)={float(np.log1p(out_pred).mean()):.6f}")
    print(f"max|log1p(reference)-log1p(rebuilt)|={max_error:.3e}")
    print(f"SHA256 rebuilt: {digest}")
    assert max_error < 1e-6
    assert digest == EXPECTED_SHA256, "Байтовый SHA не совпал с отправленным CSV"
    print(f"OK: {OUTPUT.relative_to(ROOT)} побитово совпадает с отправленным сабмитом")


if __name__ == "__main__":
    main()

```
