# e01_schema

## Catalogue metadata

- **Catalogue ID:** `eda__e01_schema`
- **Namespace:** `eda`
- **Experiment ID:** `e01_schema`
- **Original source:** `research/eda/e01_schema.py`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** EDA experiment/script
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e01_schema

Original script: `research/eda/e01_schema.py`

```python
import pandas as pd
import pyarrow.parquet as pq

P = r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw\train.parquet"

f = pq.ParquetFile(P)
md = f.metadata
print("rows:", md.num_rows, "row_groups:", md.num_row_groups, "cols:", md.num_columns)
print("created_by:", md.created_by)
print()
print("=== SCHEMA ===")
print(f.schema_arrow)
print()
print("=== ROW GROUP SIZES ===")
for i in range(min(md.num_row_groups, 12)):
    rg = md.row_group(i)
    print(i, "rows:", rg.num_rows, "bytes:", rg.total_byte_size)
print()

print("=== HEAD (first row group, 30 rows) ===")
head = f.read_row_group(0).to_pandas().head(30)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 100)
print(head)
print()
print("=== SAMPLE SUBMIT ===")
ss = pd.read_csv(r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw\sample_submit.csv")
print(ss.shape)
print(ss.head())
print(ss.describe())
print("n unique user:", ss["user_id"].nunique() if "user_id" in ss.columns else "n/a")
print("cols:", list(ss.columns))

```
