from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


HERE = Path(__file__).resolve().parent
OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
PACKET = Path(r"C:\Users\Admin\Desktop\submission_geometry_research\gpt_pro_research_packet")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_frame(frame: pl.DataFrame, columns: list[str], rows: int, key: list[str],
                 allow_diagnostics: bool = False) -> None:
    if allow_diagnostics:
        assert frame.columns[:len(columns)] == columns
    else:
        assert frame.columns == columns
    assert frame.height == rows
    assert frame.select(pl.struct(key).n_unique()).item() == rows
    for name in ("predict", "z_predict", "z_base", "correction"):
        values = frame[name].to_numpy()
        assert np.isfinite(values).all(), name
    assert (frame["predict"].to_numpy() >= 0).all()


def main() -> None:
    oof_columns = ["user_id", "fold", "target", "predict", "z_predict", "z_base",
                   "correction", "candidate_name"]
    test_columns = ["user_id", "predict", "z_predict", "z_base", "correction",
                    "candidate_name"]
    fresh_oof = pl.read_parquet(HERE / "fresh_conditional_OOF.parquet")
    fresh_test = pl.read_parquet(HERE / "fresh_conditional_TEST.parquet")
    combined_oof = pl.read_parquet(HERE / "btyd05_fresh1_OOF.parquet")
    combined_test = pl.read_parquet(HERE / "btyd05_fresh1_TEST.parquet")
    final_csv = pl.read_csv(HERE / "btyd05_fresh1_TEST.csv")
    assert_frame(fresh_oof, oof_columns, 770_616, ["fold", "user_id"], allow_diagnostics=True)
    assert_frame(combined_oof, oof_columns, 770_616, ["fold", "user_id"])
    assert_frame(fresh_test, test_columns, 250_000, ["user_id"], allow_diagnostics=True)
    assert_frame(combined_test, test_columns, 250_000, ["user_id"])
    assert final_csv.columns == ["user_id", "predict"] and final_csv.height == 250_000

    sample = pl.read_csv(OLD / "data" / "raw" / "sample_submit.csv")
    assert np.array_equal(final_csv["user_id"].to_numpy(), sample["user_id"].to_numpy())
    assert np.array_equal(combined_test["user_id"].to_numpy(), sample["user_id"].to_numpy())
    csv_error = float(np.max(np.abs(final_csv["predict"].to_numpy()
                                    - combined_test["predict"].to_numpy())))
    assert csv_error < 1e-12

    aligned_test = pl.read_parquet(PACKET / "07_ALIGNED_TEST.parquet")
    assert np.array_equal(aligned_test["user_id"].to_numpy(), sample["user_id"].to_numpy())
    z_base = np.log1p(aligned_test["pred_exp037_rebuilt"].to_numpy().astype(float))
    z_btyd = np.log1p(aligned_test["pred_btyd"].to_numpy().astype(float))
    d_fresh = fresh_test["correction"].to_numpy().astype(float)
    z_expected = 0.95 * z_base + 0.05 * z_btyd + d_fresh
    p_expected = np.expm1(np.maximum(z_expected, 0.0))
    formula_error = float(np.max(np.abs(p_expected - final_csv["predict"].to_numpy())))
    assert formula_error < 1e-10

    ledger_failures = []
    for line in (HERE / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = digest(HERE / name)
        if actual != expected:
            ledger_failures.append(name)
    assert not ledger_failures

    manifest_failures = []
    with (HERE / "artifact_manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["sha256"]:
                continue
            path = Path(row["path"])
            if not path.exists() or path.stat().st_size != int(row["size"]) or digest(path) != row["sha256"]:
                manifest_failures.append(row["path"])
    assert not manifest_failures

    result = {
        "status": "PASS",
        "oof_rows": combined_oof.height,
        "test_rows": combined_test.height,
        "canonical_test_order": True,
        "csv_vs_parquet_max_abs_error": csv_error,
        "production_formula_max_abs_error": formula_error,
        "checksum_ledger_entries": len((HERE / "checksums.sha256").read_text(encoding="utf-8").splitlines()),
        "checksum_ledger_failures": ledger_failures,
        "artifact_manifest_failures": manifest_failures,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
