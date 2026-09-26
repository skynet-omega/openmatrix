"""Rebuild sham admission from raw arrays and complete initial state, not flags."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
DONOR=ROOT/'campanas/etapa45_navigation_wind_20260925_40'

def need(ok,label):
    if not ok:raise ValueError(label)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def calculate(folder):
    folder=Path(folder)
    plan=json.loads((HERE/'PLAN.json').read_text())
    for rel,digest in plan['fixed_sha256'].items():need(sha(ROOT/rel)==digest,'Prospective input changed')
    for rel,digest in json.loads((HERE/'SOURCE_LOCK.json').read_text()).items():
        need(sha(ROOT/rel)==digest,'Scientific source changed')
    metadata=json.loads((folder/'RESULT.json').read_text())
    contract=json.loads((folder/'RUN_CONTRACT.json').read_text())
    need(metadata['stage4_admission'] is False and metadata['stage5_admission'] is False
         and metadata['completed_ms']==120 and metadata['error'] is None and metadata['cleanup_errors']==[],
         'Sham execution flags differ from verified scope')
    need(contract['arm']=='sham' and contract['plan_sha256']==sha(HERE/'PLAN.json')
         and contract['source_lock_sha256']==sha(HERE/'SOURCE_LOCK.json')
         and contract['body_commands_from_donor'] is True and contract['neural_output_delivered'] is False,
         'Sham context/flag changed')
    sys.path.insert(0,str(DONOR))
    from checkpoint_compare_utf8 import compare
    initial=compare(DONOR/'navigation_minus_filtered_wind_03/final_state',folder/'prepared_state')
    need(initial['exact'],'Restored initial state differs')
    with np.load(folder/'traces.npz',allow_pickle=False) as z:actual={k:z[k] for k in z.files}
    with np.load(DONOR/'merged_full_01/traces.npz',allow_pickle=False) as z:original={k:z[k] for k in z.files}
    select=(original['fase']=='ensayo') & (original['paso']>=1001) & (original['paso']<=1120)
    need(np.array_equal(actual['paso'],np.arange(1001,1121)),'Incomplete or reordered sham clock')
    fields=sorted(set(original)&set(actual))
    need(set(original)<=set(actual),'Missing original scientific field')
    differences=[]
    text_storage_width=[]
    for key in fields:
        a,b=actual[key],original[key][select]
        need(a.shape==b.shape,'Changed shape: '+key)
        if key=='fase':
            need(a.dtype.kind==b.dtype.kind=='U','Changed phase encoding')
            if a.dtype!=b.dtype:text_storage_width.append({'field':key,'actual':str(a.dtype),'donor':str(b.dtype)})
        else:need(a.dtype==b.dtype,'Changed numerical dtype: '+key)
        if a.dtype.kind in 'fc':need(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite: '+key)
        if not np.array_equal(a,b):
            differences.append({'field':key,'max_abs':float(np.max(np.abs(a-b))) if a.dtype.kind in 'fiu' else None,
                                'first_index':int(np.flatnonzero(np.any(a!=b,axis=tuple(range(1,a.ndim))))[0]) if a.ndim>1 else int(np.flatnonzero(a!=b)[0])})
    witness=json.loads((folder/'EXOGENOUS_WITNESS.json').read_text())
    need(len(witness)==120,'Incomplete input witness')
    rng_stable=all(w['rng_before']==w['rng_after']==witness[0]['rng_before'] for w in witness)
    need(rng_stable,'Live RNG changed unexpectedly; inspect actual noise before proceeding')
    result={'schema':'stage45_sham_raw_verification44_v1',
            'classification':'SHAM_EXACT' if not differences else 'BLOQUEADO_CONTINUIDAD',
            'plan_sha256':sha(HERE/'PLAN.json'),'source_lock_sha256':sha(HERE/'SOURCE_LOCK.json'),
            'actual_trace_sha256':sha(folder/'traces.npz'),'complete_initial_state_exact':True,
            'compared_fields':fields,'different_fields':differences,'phase_unicode_storage_width':text_storage_width,'sham_rows':120,
            'live_brain_rng_unchanged':rng_stable,'stage4_admission':False,'stage5_admission':False}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=calculate(args.folder)
    with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'classification':result['classification'],'differences':result['different_fields']}))
    if result['classification']!='SHAM_EXACT':raise SystemExit(2)

if __name__=='__main__':main()
