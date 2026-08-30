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
