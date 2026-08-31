"""Cheap staleness pilot: apply the frozen A1-365 model whose training cutoffs match the
deployed TEST model (fold 2025-10-16 -> train cutoffs 2025-07-31..2025-09-11) to later
folds, and compare with the model retrained at that later fold."""
import numpy as np, pandas as pd, datetime as dt, math, json, os, time, lightgbm as lgb
H=os.path.expanduser('~')
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
CONF=f'{H}/mnt/e-cup-research-clean/research/new_directions/EXP075_CONFIRMATION'
PROC=f'{H}/mnt/OZON-E-CUP/data/processed'
DS=dt.date(2025,1,1); WINDOWS=[7,14,30,60,90,180,365]; NCH=11; CH=[1,4,5,6,7,8,9,10,11,12,13]
CTX=len(WINDOWS)*NCH*2+NCH+1
panel=np.load(f'{PROC}/seq_panel_v1.npy',mmap_mode='r'); UID=np.load(f'{PROC}/seq_uid_v1.npy')
def dayi(c): return (c-DS).days
def padded(rows,c,hist=365):
    d=dayi(c); st=d-hist+1; lo=max(0,st)
    out=np.zeros((len(rows),hist,NCH),dtype=np.float32)
    if lo<=d: out[:,lo-st:,:]=np.asarray(panel[rows,lo:d+1,:],dtype=np.float32)[:,:,CH]
    return out
def design(rows,c,chunk=8000):
    nw=math.ceil(365/7); dim=(nw+28)*NCH+CTX
    X=np.empty((len(rows),dim),dtype=np.float32)
    for s in range(0,len(rows),chunk):
        e=min(s+chunk,len(rows)); seq=padded(rows[s:e],c)
        cols=[]
        for w in WINDOWS:
            t=seq[:,-w:,:]; cols += [t.sum(1),(t>0).sum(1,dtype=np.int32).astype(np.float32)]
        nz=seq>0; rec=np.full((e-s,NCH),366.,np.float32)
        has=nz.any(1); rev=nz[:,::-1,:].argmax(1).astype(np.float32); rec[has]=rev[has]
        avail=np.full((e-s,1),min(dayi(c)+1,365),np.float32)
        ctx=np.concatenate([*cols,rec,avail],axis=1)
        pad=nw*7-365
        sw=np.pad(seq,((0,0),(pad,0),(0,0))) if pad else seq
        weekly=sw.reshape(e-s,nw,7,NCH).sum(2)
        X[s:e,:nw*NCH]=weekly.reshape(e-s,-1)
        X[s:e,nw*NCH:(nw+28)*NCH]=seq[:,-28:,:].reshape(e-s,-1)
        X[s:e,(nw+28)*NCH:]=ctx
        del seq,sw,nz,rec,cols,weekly
    return X
def rho(u,r,zb):
    u=u-u.mean(); x=zb-zb.mean(); b=x@u/(x@x); p=u-b*x; p-=p.mean()
    b2=x@p/(x@x); p-=b2*x; p-=p.mean()
    rr=r-r.mean(); return float(p@rr/np.sqrt((p@p)*(rr@rr))), float(np.sqrt((p**2).mean()))
MODELS={c:lgb.Booster(model_file=f'{CONF}/models/A1_365_{c}.txt') for c in
        ['2025-10-16','2025-11-13','2026-01-14']}
rng=np.random.default_rng(11); NSUB=60000; res=[]
t0=time.time()
for ev in ['2025-10-16','2025-11-13','2026-01-14']:
    d=pd.read_parquet(f'{CONF}/fold_{ev}.parquet')
    idx=rng.choice(len(d),min(NSUB,len(d)),replace=False); idx.sort()
    dd=d.iloc[idx]; rows=np.searchsorted(UID,dd.user_id.values).astype(np.int32)
    X=design(rows,dt.date.fromisoformat(ev)); print(ev,'design',X.shape,f'{time.time()-t0:.0f}s',flush=True)
    r=dd.residual.values; zb=dd.baseline_z.values
    rr,_=rho(dd.u_raw_365.values,r,zb); res.append(dict(eval_fold=ev,model='stored_u_raw_fresh',rho=rr,gap_days=0))
    for mc,m in MODELS.items():
        gap=(dt.date.fromisoformat(ev)-dt.date.fromisoformat(mc)).days
        if gap<0: continue
        p=m.predict(X); a,s=rho(p,r,zb)
        res.append(dict(eval_fold=ev,model=f'A1_365@{mc}',gap_days=gap,rho=a,rms_perp=s))
        print(res[-1],flush=True)
    del X
pd.DataFrame(res).to_csv(f'{OUT}/s17_staleness.csv',index=False)
print(pd.DataFrame(res).to_string(index=False))
