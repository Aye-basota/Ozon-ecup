import numpy as np, pandas as pd, os, time, json
H=os.path.expanduser('~'); ART=f'{H}/mnt/OZON-E-CUP/artifacts'
E75=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS'
CACHE=f'{H}/wk'; os.makedirs(CACHE,exist_ok=True)
a1=pd.read_parquet(f'{E75}/clean_forward_predictions.parquet')
uid=a1.user_id.values.astype(np.int64); cut=a1.cutoff.values.astype(str)
cutcode={c:i for i,c in enumerate(sorted(set(cut)))}
key=np.array([cutcode[c] for c in cut],dtype=np.int64)*10_000_000 + uid
order=np.argsort(key); key_sorted=key[order]
def reindex(u2,c2,z2):
    k2=np.array([cutcode[str(c)] for c in c2],dtype=np.int64)*10_000_000 + u2.astype(np.int64)
    pos=np.searchsorted(key_sorted,k2)
    if pos.max()>=len(key_sorted) or not (key_sorted[np.clip(pos,0,len(key_sorted)-1)]==k2).all(): return None
    out=np.empty(len(key),dtype=np.float64); out[order[pos]]=z2
    return out
SIMPLE=['S1-E02','S1-E03a','S1-DIST','S1-E10','S1-E11','S1-SEEDAVG5','S1-B0','S1-E01','S1-E03b',
        'SEQ-AVG3','SEQ-D3A-AVG3','SEQ-D3A-BASE-AVG3','ETX-AVG3','ETX-AVG2','ETX-01-S42',
        'PT-FULL-AVG3','PT-OD-AVG3','PT-SHUF-AVG3','RIDGE15','HOLIDAY-YOY-FAST','MHZ-FULL','MHZ-BASE','MHZ-P30','MHZ-SELF',
        'S04-A','S04-B','S04-C','GAP-E02-K5-G090-S42','GAP-E10-K5-G090-S42','GAP-DIST-K5-G060-S42',
        'SAMPLE-TB1-AVG3-R300','SAMPLE-BASELINE-B-AVG3-R300','SAMPLE-DENSE-S3-F422-S42-R300','S1-ROUNDS-R600','S1-ROUNDS-R300']
names=[];cols=[];prov={}
t0=time.time()
for nm in SIMPLE:
    p=f'{ART}/oof_{nm}.npz'
    if not os.path.exists(p): print('MISS',nm); continue
    d=np.load(p,allow_pickle=True); z=d['z'].astype(np.float64)
    if (d['user_id']==uid).all() and (d['cutoff']==cut).all(): v=z; how='direct'
    else:
        v=reindex(d['user_id'],d['cutoff'],z); how='reindexed'
        if v is None: print('KEYFAIL',nm); continue
    if not np.isfinite(v).all(): print('NONFINITE',nm); continue
    names.append(nm); cols.append(v); prov[nm]=dict(file=f'oof_{nm}.npz',align=how)
def add_special(path,ukey,fields,pref):
    d=np.load(path,allow_pickle=True)
    for k in fields:
        v=d[k].astype(np.float64)
        if not ((d[ukey]==uid).all() and (d['cutoff']==cut).all()):
            v=reindex(d[ukey],d['cutoff'],v)
            if v is None: print('KEYFAIL',pref,k); continue
        names.append(pref+k); cols.append(v); prov[pref+k]=dict(file=os.path.basename(path),field=k)
add_special(f'{ART}/BTYD_STABLE_EXP051/oof_raw.npz','user_id',['z_btyd','z_strongest'],'BTYD:')
add_special(f'{ART}/oof_BLOCK4_SAF.npz','uid',['z_new_honest'],'BLOCK4:')
add_special(f'{ART}/oof_FRESH_CONTRAST_MOE.npz','uid',['z_fresh','z_vol','z_clean'],'FRESH:')
Z=np.column_stack(cols)
np.save(f'{CACHE}/Zcomp.npy',Z); json.dump({'names':names,'prov':prov},open(f'{CACHE}/Zcomp_names.json','w'),indent=1)
tl=a1.target_log.values
print(f'{Z.shape} in {time.time()-t0:.1f}s ; weak RFM pooled RMSLE {np.sqrt(((tl-a1.baseline_z.values)**2).mean()):.6f}')
for n,v in sorted([(n,float(np.sqrt(((tl-Z[:,i])**2).mean()))) for i,n in enumerate(names)],key=lambda x:x[1]): print(f'{n:32s} {v:.6f}')
