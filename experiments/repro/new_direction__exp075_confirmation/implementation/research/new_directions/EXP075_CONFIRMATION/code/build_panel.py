import polars as pl, numpy as np, datetime as dt, json, time, os
RAW="/mnt/user-data/uploads/OZON-E-CUP/data/raw/train.parquet"
OUT="/home/claude/work"
DATA_START=dt.date(2025,1,1); DATA_END=dt.date(2026,2,13)
N_DAYS=(DATA_END-DATA_START).days+1
RAW_CHANNELS=["cat","searches","search_to_cart","search_to_ord","cat_to_cart",
              "cat_to_ord","to_cart","to_ord","gmv_search","gmv_cat","gmv"]
T0=time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]",*a,flush=True)
log("N_DAYS",N_DAYS)
uid=(pl.scan_parquet(RAW).select(pl.col("user_id").unique()).collect()
     .to_series().sort().to_numpy().astype(np.int64))
log("users",len(uid))
np.save(f"{OUT}/uid.npy",uid)
panel=np.lib.format.open_memmap(f"{OUT}/panel11.npy",mode="w+",dtype=np.float16,
                                shape=(len(uid),N_DAYS,11))
gmv=np.lib.format.open_memmap(f"{OUT}/gmv.npy",mode="w+",dtype=np.float64,
                              shape=(len(uid),N_DAYS))
# stream by user_id ranges to bound memory
edges=np.linspace(0,len(uid),21).astype(int)
for k in range(20):
    lo,hi=edges[k],edges[k+1]
    ulo,uhi=int(uid[lo]),int(uid[hi-1])
    df=(pl.scan_parquet(RAW)
        .filter((pl.col("user_id")>=ulo)&(pl.col("user_id")<=uhi))
        .select(["user_id","event_date",*RAW_CHANNELS]).collect())
    u=df["user_id"].to_numpy(); ui=np.searchsorted(uid,u)
    di=(df["event_date"].to_numpy()-np.datetime64(DATA_START)).astype("timedelta64[D]").astype(int)
    assert di.min()>=0 and di.max()<N_DAYS
    for j,name in enumerate(RAW_CHANNELS):
        v=df[name].to_numpy()
        v = v.astype(np.float32) if name=="cat" else np.log1p(v.astype(np.float32))
        panel[ui,di,j]=v.astype(np.float16)
    gmv[ui,di]=df["gmv"].to_numpy().astype(np.float64)
    del df,u,ui,di
    log("chunk",k,"rows done")
panel.flush(); gmv.flush()
log("done")
