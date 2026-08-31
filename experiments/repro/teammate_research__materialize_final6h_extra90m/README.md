# materialize_final6h_extra90m

## Catalogue metadata

- **Catalogue ID:** `teammate_research__materialize_final6h_extra90m`
- **Namespace:** `teammate_research`
- **Experiment ID:** `materialize_final6h_extra90m`
- **Original source:** `пайплайн сокомандника/research_scripts/materialize_final6h_extra90m.py`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** teammate research runner
- **Model:** Ridge, blend
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** rows.sort(key=lambda r:(r["delta"],r["latest_delta"]));save_csv(results/"ALL_EXTRA90_VALIDATION.csv",rows)
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** for p in subs.glob("submission_extra90_*.csv"):zf.write(p,arcname=f"submissions/{p.name}")
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# materialize_final6h_extra90m

Original script: `пайплайн сокомандника/research_scripts/materialize_final6h_extra90m.py`

```python
from __future__ import annotations

"""E-CUP 2026: cheap final diversification/materialization after final6h.

This script NEVER retrains STRONGEST_CURRENT and NEVER trains new raw CAP/UNC/DIST/
SEQ/ETX/occurrence models. It reuses the already-complete OOF/test checkpoint bank
created by continue_best_bas_final6h.py, spends the remaining ~90 minutes only on
cheap meta-occurrence / stacking searches, then materializes FOUR locally-winning
and distributionally different submissions.

Expected location: src/DL/best_bas/ beside continue_best_bas_final6h.py and the
previous fixedstack scripts. Run from repository root.
"""

import argparse
import dataclasses
import datetime as dt
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

FOLDS=("2025-09-04","2025-09-18","2025-10-02","2025-10-16")
FW=np.asarray([1.,2.,4.,8.],dtype=np.float64)
VERSION="extra90m_cached_meta_2026-08-23_001"


def now(): return dt.datetime.now().isoformat(timespec="seconds")
def log(*x): print(f"[{now()}]",*x,flush=True)

def import_module(path:Path,name:str):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def save_csv(path:Path,rows):
    import pandas as pd
    path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(list(rows)).to_csv(path,index=False)

def atomic_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)),encoding="utf-8")
    os.replace(tmp,path)

def align(src_uid,arr,dst_uid):
    src_uid=np.asarray(src_uid,np.int64);dst_uid=np.asarray(dst_uid,np.int64);arr=np.asarray(arr)
    if np.array_equal(src_uid,dst_uid): return arr
    order=np.argsort(src_uid);pos=np.searchsorted(src_uid[order],dst_uid)
    if np.any(pos>=len(src_uid)) or not np.array_equal(src_uid[order][pos],dst_uid): raise ValueError("user_id alignment mismatch")
    return arr[order][pos]

def distance(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);d=a-b
    corr=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else 0.
    return {"corr":corr,"std":float(np.std(d)),"mae":float(np.mean(np.abs(d))),
            "pct02":float(np.mean(np.abs(d)>.02)),"pct05":float(np.mean(np.abs(d)>.05)),"pct10":float(np.mean(np.abs(d)>.10))}

def self_test():
    rng=np.random.default_rng(7);friend=rng.normal(2,.7,5000)
    preds={
        "A":friend+rng.normal(0,.035,5000),
        "B":friend+rng.normal(0,.045,5000),
        "C":friend+.025*np.sign(rng.normal(size=5000)),
        "D":friend+rng.normal(0,.02,5000),
        "E":friend+rng.normal(0,.055,5000),
    }
    rows=[{"name":n,"delta":d,"latest_delta":d*1.1,"wins_recent":3,"family":f} for n,d,f in [
        ("A",-.00165,"stable"),("B",-.00177,"occ"),("C",-.00155,"bias"),("D",-.0014,"ridge"),("E",-.0015,"rawocc")]]
    selected=select_four(rows,preds,friend,"A",{})
    assert len(selected)==4 and "A" in selected and len(set(selected))==4
    print("SELF-TEST OK",selected,flush=True)

def select_four(rows,preds,friend,anchor_name,oldsubs=None):
    """Choose four intentionally different *validated* branches.

    Slot 1: safe exploitation of the LB-confirmed temporal-Ridge family.
    Slot 2: best meta-occurrence candidate.
    Slot 3: best raw occurrence overlay (different error mechanism).
    Slot 4: best stable local-specialist / p-band / super-ridge alternative.

    Within a slot, local quality is primary but candidates nearly identical to
    already-submitted files and to selected slots are penalized.
    """
    oldsubs=oldsubs or {}
    idx={r["name"]:r for r in rows if r["name"] in preds}
    if anchor_name not in idx: raise KeyError(anchor_name)
    eligible=[r for r in rows if r["name"] in preds and r.get("delta",1)<-0.00110 and r.get("wins_recent",0)>=3 and r.get("latest_delta",1)<0]
    selected=[anchor_name]

    def novelty(n,against):
        refs=[preds[x] for x in against if x in preds]+list(oldsubs.values())
        if not refs:return 1.0,0.02,0.10
        ds=[distance(preds[n],z) for z in refs]
        maxcorr=max(d["corr"] for d in ds);minstd=min(d["std"] for d in ds);minp02=min(d["pct02"] for d in ds)
        # 0..1-ish novelty score. Corr alone is too sensitive, so combine it
        # with actual prediction displacement.
        nov=.45*min(1.,minstd/.018)+.35*min(1.,minp02/.12)+.20*min(1.,max(0.,(1-maxcorr)/.00012))
        return nov,minstd,minp02

    def pick(pool,selected_now):
        if not pool:return None
        gain=max(-r["delta"] for r in eligible) if eligible else .001
        scored=[]
        for r in pool:
            nov,sd,p02=novelty(r["name"],selected_now)
            q=(-r["delta"])/gain;latest=min(1.25,max(0.,-r["latest_delta"]/.0020))
            score=.68*q+.17*latest+.15*nov
            scored.append((score,r["name"],sd,p02))
        return max(scored)[1]

    meta=[r for r in eligible if r.get("family") in {"xmeta_risk","xmeta_plain","occurrence_meta_risk","occurrence_meta"}]
    n=pick(meta,selected)
    if n and n not in selected:selected.append(n)

    raw=[r for r in eligible if r.get("family") in {"raw_occ_extra","occurrence_overlay"}]
    n=pick(raw,selected)
    if n and n not in selected:selected.append(n)

    stable=[r for r in eligible if r.get("family") in {"local_bias","hierarchical","candidate_pband","candidate_simplex","super_ridge","super_ridge_recent","super_pband","super_simplex","ridge_predonly","ridge_temporal"} and r["name"] not in selected]
    n=pick(stable,selected)
    if n and n not in selected:selected.append(n)

    # Robust generic fallback, still requiring a locally winning model.
    if len(selected)<4:
        rest=[r for r in eligible if r["name"] not in selected]
        while len(selected)<4 and rest:
            n=pick(rest,selected);selected.append(n);rest=[r for r in rest if r["name"]!=n]
    return selected[:4]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--max-minutes",type=float,default=82.0);ap.add_argument("--threads",type=int,default=max(4,min(10,os.cpu_count() or 8)));ap.add_argument("--reuse-work-dir",type=str,default=None);ap.add_argument("--no-install",action="store_true");ap.add_argument("--self-test",action="store_true");args=ap.parse_args()
    if args.self_test:self_test();return
    t0=time.time();deadline=t0+args.max_minutes*60.
    base=Path(__file__).resolve().parent
    fpath=base/"continue_best_bas_final6h.py"
    if not fpath.exists():raise FileNotFoundError(f"Не найден {fpath.name} рядом с новым файлом")
    F=import_module(fpath,"final6_parent_extra90")
    combo,combo_path,fixed,fixed_path,prev,prev_path=F.discover_parent_scripts(base)
    package=prev.discover_package(base);raw,sample=prev.discover_raw_and_sample(base,package);prev.ensure_dependencies(package,args.no_install)
    work=fixed.discover_work(base,args.reuse_work_dir)
    out=base/"_best_bas_extra90m";results=out/"results";subs=out/"submissions";results.mkdir(parents=True,exist_ok=True);subs.mkdir(parents=True,exist_ok=True)
    ctx=prev.Context(base_dir=base,package=package,pipeline=package/"pipeline",raw=raw,sample=sample,work=work,results=results,submissions=subs,checkpoints=work/"checkpoints",budget=prev.Budget(t0,max(args.max_minutes/60.,1.0),max(args.max_minutes/60.-.15,0),.10))
    prev.configure_pipeline(ctx,args.threads);friend=prev.verify_friend_package(package)
    log("EXTRA90",VERSION,"work",work,"friend_rebuild",friend.get("max_log_error"))
    log("NO raw/base/SEQ/ETX models will be trained. Cached OOF/test only.")

    # Require the eight already trained occurrence families and all fixedstack OOF.
    occ_names=[]
    for cfg in F.OCC_QUEUE:
        ok=all(F.valid_npz(F.occ_fold_path(ctx,cfg.name,f),("user_id","y","p")) for f in FOLDS) and F.valid_npz(F.occ_test_path(ctx,cfg.name),("user_id","p"))
        if ok:occ_names.append(cfg.name)
    if len(occ_names)<6:raise RuntimeError(f"Слишком мало готовых occurrence families: {occ_names}; новый файл принципиально не будет их переобучать")
    log("CACHED OCC",occ_names)

    bank=F.load_full_bank(combo,fixed,prev,ctx);occ_names=F.load_occ_into_bank(bank,ctx,occ_names)
    rows,predpool,combo_recipes,experts=combo.build_primitive_research(fixed,bank,results,"extra90")
    rows,predpool,combo_recipes=combo.build_combo_research(fixed,bank,rows,predpool,combo_recipes,experts,results,"extra90")
    rows,predpool,newrecipes=F.extend_stable_research(combo,fixed,bank,rows,predpool,combo_recipes,results)
    ridx=lambda:{r["name"]:r for r in rows}
    anchor="blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85"
    if anchor not in predpool:raise RuntimeError(f"Не восстановлен обязательный anchor {anchor}")

    # Stable alternate bases that already won 3/3 recent folds.
    stable_base_names=[anchor]
    for n in ("bias_p_spread_cand_pband_stack","cand_pband_stack","hier_trust_bias_ridge_recentpow1p7_s075","superridge_a20_s95"):
        rr=ridx().get(n)
        if n in predpool and rr and rr["delta"]<0 and rr["wins_recent"]>=3:stable_base_names.append(n)
    stable_base_names=stable_base_names[:4]
    log("STABLE BASES",stable_base_names)

    extra_specs={};extra_rows=[]
    # Reproduce the two already selected mechanisms plus several deliberately
    # different cheap meta-occurrence views. No raw model training here.
    configs=[]
    all8=list(occ_names)
    def keep(names):return [n for n in names if n in all8]
    configs += [("all8_p17_l31",all8,1.7,31)]
    configs += [("all8_p23_l31",all8,2.3,31)]
    configs += [("div4_p23_l31",keep(["occ_r10_fast","occ_r14_multiscale","occ_r20_shallow","occ_r24_multiscale"]),2.3,31)]
    configs += [("fast4_p23_l23",keep(["occ_r10_fast","occ_r12_wide","occ_r14_multiscale","occ_r16_bal"]),2.3,23)]
    configs += [("stable4_p17_l31",keep(["occ_r18_wide","occ_r20_shallow","occ_r22_stable","occ_r24_multiscale"]),1.7,31)]
    configs=[c for c in configs if len(c[1])>=2]

    for ci,(tag,names,power,leaves) in enumerate(configs):
        if time.time()>deadline-18*60 and ci>0:
            log("META SEARCH TIME GUARD before",tag);break
        log("META SEARCH",tag,"n",names,"power",power,"leaves",leaves)
        pm=F.walk_meta_occ(bank,names,power=power,leaves=leaves);key=f"p_xmeta_{tag}"
        for f in FOLDS:bank[f][key]=pm[f]
        risk=F.walk_risk_gate(bank,names,power=power)
        # Anchor always; local-bias base only for the most promising configs to
        # create a truly different error pattern without exploding runtime.
        bases=stable_base_names[:2] if tag in {"all8_p23_l31","div4_p23_l31"} else stable_base_names[:1]
        for base_name in bases:
            base_oof=predpool[base_name]
            for gated in (False,True):
                name=f"xmeta_{tag}_{'risk' if gated else 'plain'}__{base_name}"
                p,_=F.walk_occ_candidate(bank,key,base_oof,risk if gated else None,True)
                fam="xmeta_risk" if gated else "xmeta_plain"
                rr=F.score_table(fixed,name,p,bank,fam,rows,notes=f"subset={names};power={power};leaves={leaves};base={base_name}")
                predpool[name]=p;extra_rows.append(rr);extra_specs[name]={"kind":"meta","tag":tag,"occ_names":names,"power":power,"leaves":leaves,"base":base_name,"risk":gated,"pkey":key}
        del pm,risk;gc.collect()

    # Add two raw occurrence overlays with strongest/different temporal behavior.
    for occn in [n for n in ("occ_r10_fast","occ_r20_shallow","occ_r22_stable") if n in occ_names]:
        name=f"xraw_{occn}_adapt__{anchor}";p,_=F.walk_occ_candidate(bank,f"p_{occn}",predpool[anchor],None,True)
        rr=F.score_table(fixed,name,p,bank,"raw_occ_extra",rows,notes=f"raw={occn};base={anchor}");predpool[name]=p;extra_rows.append(rr);extra_specs[name]={"kind":"raw","occ":occn,"base":anchor}

    rows.sort(key=lambda r:(r["delta"],r["latest_delta"]));save_csv(results/"ALL_EXTRA90_VALIDATION.csv",rows)
    log("TOP EXTRA VALIDATION")
    for r in rows[:20]:log(" ",r["name"],r["family"],f"d={r['delta']:+.6f}","recent",r["wins_recent"],"latest",f"{r['latest_delta']:+.6f}")

    # Build test bank only after all OOF research is finished.
    test=F.build_test_bank(combo,fixed,prev,ctx,friend)
    for n in occ_names:
        d=F.load_npz(F.occ_test_path(ctx,n));test[f"p_{n}"]=np.clip(align(d["user_id"],d["p"],test["uid"]),F.EPS,1-F.EPS)
    cache={}
    def final_stable(name):
        if name=="table_core":return np.asarray(test["table_core"],np.float64)
        return F.finalize_stable_candidate(name,combo,fixed,bank,test,predpool,combo_recipes,newrecipes,cache)

    meta_final_cache={};risk_final_cache={}
    def final_extra(name):
        if name not in extra_specs:return final_stable(name)
        sp=extra_specs[name];base_test=final_stable(sp["base"]);base_oof=predpool[sp["base"]]
        if sp["kind"]=="raw":
            ptest=np.asarray(test[f"p_{sp['occ']}"]);z,_=F.final_occ_candidate(bank,test,f"p_{sp['occ']}",ptest,base_oof,base_test,None,None,True);return z
        sig=(tuple(sp["occ_names"]),sp["power"],sp["leaves"])
        ptest_map={n:np.asarray(test[f"p_{n}"]) for n in sp["occ_names"]}
        if sig not in meta_final_cache:meta_final_cache[sig]=F.final_meta_occ(bank,test,sp["occ_names"],ptest_map,power=sp["power"],leaves=sp["leaves"])
        pmeta=meta_final_cache[sig]
        pkey=sp["pkey"]
        risk_oof=None;risk_test=None
        if sp["risk"]:
            rsig=(tuple(sp["occ_names"]),sp["power"])
            # walk risk is cheap enough but cache it by occurrence subset.
            if rsig not in risk_final_cache:
                risk_final_cache[rsig]=(F.walk_risk_gate(bank,sp["occ_names"],sp["power"]),F.final_risk_gate(bank,test,sp["occ_names"],ptest_map,sp["power"]))
            risk_oof,risk_test=risk_final_cache[rsig]
        z,_=F.final_occ_candidate(bank,test,pkey,pmeta,base_oof,base_test,risk_oof,risk_test,True);return z

    # Materialize a quality-controlled shortlist from different families.
    shortlist=[anchor]
    good=[r for r in rows if r["name"] in predpool and r["delta"]<-0.00110 and r["wins_recent"]>=3 and r["latest_delta"]<0]
    # Meta-occurrence: keep several strong variants, but do not let almost identical
    # configurations crowd out genuinely different raw/local-specialist branches.
    seen_tags=set()
    for r in good:
        if r["name"] not in extra_specs or extra_specs[r["name"]].get("kind")!="meta":continue
        tag=extra_specs[r["name"]].get("tag")
        if tag in seen_tags:continue
        shortlist.append(r["name"]);seen_tags.add(tag)
        if len(seen_tags)>=5:break
    # Raw occurrence overlays are required for a different error geometry.
    raws=[r for r in good if r.get("family")=="raw_occ_extra"]
    for r in raws[:2]:
        if r["name"] not in shortlist:shortlist.append(r["name"])
    # Stable local alternatives: different mechanism from both A and occurrence.
    for n in ("bias_p_spread_cand_pband_stack","hier_trust_bias_ridge_recentpow1p7_s075","superridge_a20_s95","cand_pband_stack"):
        rr=ridx().get(n)
        if n in predpool and rr and rr["delta"]<0 and rr["wins_recent"]>=3 and n not in shortlist:shortlist.append(n)
    shortlist=shortlist[:13]
    log("FINALIZE SHORTLIST",shortlist)

    table_preds={};final_preds={};materialized_rows=[]
    for n in shortlist:
        if time.time()>deadline-8*60:
            log("FINALIZATION TIME GUARD before",n);break
        try:
            tb=final_extra(n);zz=fixed.transform_to_friend(np.asarray(friend["z"],np.float64),test["table_core"],tb,1.0)
            table_preds[n]=tb;final_preds[n]=zz
            rr=ridx()[n];materialized_rows.append({"name":n,"family":rr["family"],"delta":rr["delta"],"wins_recent":rr["wins_recent"],"latest_delta":rr["latest_delta"],**{f"friend_{k}":v for k,v in distance(zz,friend["z"]).items()}})
            log("MATERIALIZED",n,f"d={rr['delta']:+.6f}","corr_friend",f"{materialized_rows[-1]['friend_corr']:.6f}")
        except Exception as exc:
            log("MATERIALIZE FAILED",n,repr(exc));traceback.print_exc()
    if anchor not in final_preds:raise RuntimeError("Anchor could not be materialized")

    oldsubs=F.locate_old_submissions(base,np.asarray(friend["uid"],np.int64))
    chosen=select_four([ridx()[n] for n in final_preds],final_preds,np.asarray(friend["z"]),anchor,oldsubs)
    log("SELECTED FOUR",chosen)
    pair=[]
    for i,a in enumerate(chosen):
        for b in chosen[i+1:]:pair.append({"a":a,"b":b,**distance(final_preds[a],final_preds[b])})
    save_csv(results/"MATERIALIZED_CANDIDATES.csv",materialized_rows);save_csv(results/"SELECTED_PAIR_DIVERSITY.csv",pair)

    import pandas as pd
    sample_df=pd.read_csv(sample);suid=sample_df["user_id"].to_numpy(np.int64);final_rows=[]
    for i,n in enumerate(chosen,1):
        z=align(friend["uid"],final_preds[n],suid);pred=np.maximum(np.expm1(np.clip(z,0,20)),0)
        df=pd.DataFrame({"user_id":suid,"predict":pred})
        if len(df)!=250000 or df.user_id.duplicated().any() or df.predict.isna().any() or (df.predict<0).any():raise RuntimeError(f"bad submission {n}")
        fn=subs/f"submission_extra90_{i}_{n}.csv";df.to_csv(fn,index=False);rr=ridx()[n];final_rows.append({"rank":i,"name":n,"family":rr["family"],"delta":rr["delta"],"wins_recent":rr["wins_recent"],"latest_delta":rr["latest_delta"],"file":str(fn),**distance(final_preds[n],friend["z"])})
    save_csv(results/"FINAL_FOUR.csv",final_rows)

    manifest={"version":VERSION,"finished":now(),"runtime_minutes":(time.time()-t0)/60.,"work":str(work),"occ_names":occ_names,"selected":final_rows,"friend_rebuild_error":friend.get("max_log_error")};atomic_json(results/"RUN_MANIFEST.json",manifest)
    report=["E-CUP cached extra90 diversification",f"runtime_minutes={(time.time()-t0)/60.:.1f}","No raw/base/SEQ/ETX model was retrained.","", "Selected four:"]
    for r in final_rows:report.append(f"{r['rank']}. {r['name']} family={r['family']} d={r['delta']:+.6f} latest={r['latest_delta']:+.6f} corr_friend={r['corr']:.6f} std={r['std']:.5f}")
    report += ["","Pair diversity:"]+[f"{r['a']} <> {r['b']}: corr={r['corr']:.6f} std={r['std']:.5f} pct02={r['pct02']:.3f} pct05={r['pct05']:.3f}" for r in pair]
    (results/"REPORT_RU.txt").write_text("\n".join(report),encoding="utf-8")
    bundle=base/f"extra90_REVIEW_BUNDLE_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in results.iterdir():
            if p.is_file() and p.suffix.lower() in {".csv",".json",".txt",".jsonl"}:zf.write(p,arcname=f"results/{p.name}")
        for p in subs.glob("submission_extra90_*.csv"):zf.write(p,arcname=f"submissions/{p.name}")
    log("DONE",f"{(time.time()-t0)/60.:.1f} min","BUNDLE",bundle)

if __name__=="__main__":main()

```
