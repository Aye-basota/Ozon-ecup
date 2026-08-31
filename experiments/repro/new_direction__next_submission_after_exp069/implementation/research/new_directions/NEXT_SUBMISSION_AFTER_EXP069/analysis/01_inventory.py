import hashlib, json, os, csv, sys
from pathlib import Path

REPO = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")
ND = REPO / "research" / "new_directions"
OUT = ND / "NEXT_SUBMISSION_AFTER_EXP069"
OUT.mkdir(parents=True, exist_ok=True)

EXPS = {
 "EXP069": ND/"EXP069_BTYD05_FRESH1_PROD",
 "EXP070": ND/"EXP070_COUNT_VALUE_MOE",
 "EXP071": ND/"EXP071_ETX_FRESH_CONTRAST",
}

def sha256f(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

ROLE = {
 'report.md':'report','config.json':'config','reconnaissance.md':'reconnaissance',
 'fold_metrics.csv':'fold_metrics','nested_selection.csv':'nested_selection',
 'pilot_metrics.json':'pilot_metrics','user_half_metrics.csv':'user_half',
 'bootstrap_metrics.csv':'bootstrap','diversity_oof.csv':'diversity',
 'artifact_manifest.csv':'manifest','checksums.sha256':'checksums',
 'runtime_resources.json':'runtime','test_span_projection.json':'test_span',
 'oof_projection_metrics.json':'oof_projection','production_regime.json':'production_regime',
 'real_vs_vol.csv':'control','real_vs_shuffled.csv':'control','fresh_vs_vol.csv':'control',
 'seq_vs_etx_fresh.csv':'control','encoder_parity.json':'encoder_parity',
 'baseline_parity.json':'baseline_parity','preprocessing_parameters.json':'preprocessing',
 'production_training_audit.json':'production_audit','final_summary.json':'final_summary',
 'oof_analysis_summary.json':'oof_summary','label_audit.csv':'label_audit',
 'class_bin_decision.json':'class_bins','class_distribution.csv':'class_distribution',
 'probability_metrics.csv':'probability_metrics','segment_metrics.csv':'segment_metrics',
}

def role_of(name):
    if name in ROLE: return ROLE[name]
    n = name.lower()
    if n.endswith('.py'): return 'source_code'
    if 'test' in n and (n.endswith('.parquet') or n.endswith('.csv')): return 'TEST_vector'
    if 'oof' in n and (n.endswith('.parquet') or n.endswith('.npz')): return 'OOF_vector'
    if n.endswith('.npz'): return 'cache_npz'
    if n.endswith('.pt'): return 'model_weights'
    if n.endswith('.pyc'): return 'pycache'
    return 'other'

rows = []
audit_notes = []

for exp, d in EXPS.items():
    declared = {}
    ck = d/'checksums.sha256'
    if ck.exists():
        for line in ck.read_text().splitlines():
            line=line.strip()
            if not line: continue
            h, _, name = line.partition('  ')
            declared[name.strip()] = h.strip()
    seen = set()
    for p in sorted(d.rglob('*')):
        if p.is_dir(): continue
        rel = p.relative_to(d).as_posix()
        name = p.name
        role = 'pycache' if '__pycache__' in rel else role_of(name)
        size = p.stat().st_size
        h = sha256f(p)
        parsed_ok = ''
        if p.suffix == '.json':
            try:
                json.loads(p.read_text(encoding='utf-8')); parsed_ok='True'
            except Exception as e:
                parsed_ok='False:'+str(e)
        elif p.suffix == '.csv':
            try:
                with open(p,newline='',encoding='utf-8') as f:
                    r=list(csv.reader(f)); parsed_ok='True:%drows'%len(r)
            except Exception as e:
                parsed_ok='False:'+str(e)
        elif p.suffix == '.parquet':
            try:
                import pyarrow.parquet as pq
                m = pq.ParquetFile(p).metadata
                parsed_ok='True:%dx%d'%(m.num_rows,m.num_columns)
            except Exception as e:
                parsed_ok='False:'+str(e)
        elif p.suffix == '.npz':
            try:
                import numpy as np
                with np.load(p, allow_pickle=False) as z:
                    parsed_ok='True:'+','.join(list(z.keys())[:8])
            except Exception as e:
                parsed_ok='False:'+str(e)
        if rel in declared:
            seen.add(rel)
            note = 'checksum_match' if declared[rel]==h else 'CHECKSUM_MISMATCH declared='+declared[rel]
            if declared[rel]!=h:
                audit_notes.append(exp+': CHECKSUM MISMATCH '+rel)
        else:
            note = 'not_in_checksums'
        rows.append(dict(experiment=exp, artifact_type=role, path=str(p), size_bytes=size,
                         sha256=h, exists='True', parsed_ok=parsed_ok, role=role, notes=note))
    missing = set(declared) - seen
    for m in sorted(missing):
        audit_notes.append(exp+': DECLARED BUT MISSING '+m)
        rows.append(dict(experiment=exp, artifact_type=role_of(m), path=str(d/m), size_bytes=0,
                         sha256='', exists='False', parsed_ok='', role=role_of(m), notes='declared_in_checksums_but_missing'))

EXT = [
   r"C:\Users\Admin\Desktop\submission_geometry_research\submission_geometry\cache\Z.npz",
   r"C:\Users\Admin\Desktop\submission_geometry_research\submission_geometry\cache\Z_meta.json",
   r"C:\Users\Admin\Desktop\submission_geometry_research\submission_geometry\SUBMIT_NEXT_BEST.csv",
   r"C:\Users\Admin\Desktop\submission_geometry_research\current_best\SUBMIT_v2_shrunk.csv",
   r"C:\Users\Admin\Desktop\submission_geometry_research\submissions\last (1).csv",
   r"C:\Users\Admin\Desktop\submission_geometry_research\submission_geometry\score_registry.csv",
   r"C:\Users\Admin\Desktop\submission_geometry_research\submission_geometry\manifest.csv",
   r"C:\Users\Admin\Desktop\submission_geometry_research\gpt_pro_research_packet\06_ALIGNED_OOF.parquet",
   r"C:\Users\Admin\Desktop\submission_geometry_research\gpt_pro_research_packet\07_ALIGNED_TEST.parquet",
   r"C:\Users\Admin\Desktop\submission_geometry_research\scores\submissions.csv",
   r"C:\Users\Admin\Desktop\e-cup-research-clean\artifacts\oof\EXP_037_STRONGEST_CURRENT.parquet",
]
for ps in EXT:
    p = Path(ps)
    if p.exists():
        rows.append(dict(experiment='EXTERNAL', artifact_type=role_of(p.name), path=str(p),
                         size_bytes=p.stat().st_size, sha256=sha256f(p), exists='True',
                         parsed_ok='', role='external_reference', notes='external'))
    else:
        rows.append(dict(experiment='EXTERNAL', artifact_type='', path=str(p), size_bytes=0,
                         sha256='', exists='False', parsed_ok='', role='external_reference', notes='MISSING'))
        audit_notes.append('EXTERNAL: MISSING '+ps)

cols = ['experiment','artifact_type','path','size_bytes','sha256','exists','parsed_ok','role','notes']
with open(OUT/'experiment_inventory.csv','w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)

print('rows=%d'%len(rows))
print('MISMATCH/MISSING notes:')
for n in audit_notes: print('  '+n)
if not audit_notes: print('  (none)')
from collections import Counter
c = Counter((r['experiment'], r['notes'].split(' ')[0]) for r in rows)
for k,v in sorted(c.items()): print(k,v)
