"""Determine which aligned-OOF sources have an exact equivalent inside the 65-source
TEST geometry bank (pre-EXP069)."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

GEO = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
GEOM = GEO/"submission_geometry"
sys.path.insert(0, str(GEOM))
from core import load_unique

Z, names, lb, guid = load_unique()
print("== 65 geometry sources (name, public LB) ==")
for i, (n, s) in enumerate(zip(names, lb)):
    print(f"  {i:2d} {s:.10f}  {n}")

at = pq.read_table(GEO/"gpt_pro_research_packet"/"07_ALIGNED_TEST.parquet").to_pandas()
uid = at["user_id"].to_numpy().astype(np.int64)
order = np.argsort(uid); pos = np.searchsorted(uid[order], guid)
assert np.array_equal(uid[order][pos], guid)

print("\n== aligned TEST column -> nearest geometry source ==")
rows = []
for c in at.columns:
    if c == "user_id": continue
    z = np.log1p(at[c].to_numpy().astype(float))[order][pos]
    d = np.sqrt(np.mean((Z - z)**2, axis=1))
    j = int(np.argmin(d))
    rows.append((c, names[j], float(d[j])))
    print(f"  {c:34s} -> {names[j]:58s} rms={d[j]:.3e}")
json.dump([dict(column=a, nearest=b, rms=c) for a,b,c in rows],
          open(Path(r"C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\NEXT_SUBMISSION_AFTER_EXP069")/"_test_col_to_geometry.json","w"), indent=1)
