"""Portable necessary-check verifier for one frozen-fast MRI real-block result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(a,b):
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
         'state shape/finiteness')
    return float(np.max(np.abs(a-b)/(1e-7+1e-5*np.maximum(np.abs(a),np.abs(b)))))


def verify(root):
    root=Path(root)
    result=json.loads((root/'MRI_REAL_FROZEN_FAST_RESULT.json').read_text())
    need(result['schema']=='mri_real_frozen_fast_result_v1' and
         result['status']=='COMPLETE_DIAGNOSTIC_ONLY','result status')
    need(result['plan_sha256']==sha(root/'MRI_REAL_FROZEN_FAST_PLAN_52.json'),
         'plan identity')
    need(result['source_sha256']==sha(root/'code/probe_mri_real_frozen_fast_v3.py'),
         'provider identity')
    need(result['chatgpt_mri_sha256']==sha(root/'code/mri_event_probe.py'),
         'MRI donor identity')
    path=root/'MRI_REAL_BLOCK_STATES.npz'
    need(result['mri']['states_sha256']==sha(path),'state payload identity')
    with np.load(path,allow_pickle=False) as data:
        need(set(data.files)=={'initial','reference','high','low'},'array names')
        initial,ref,high,low=[data[k] for k in ('initial','reference','high','low')]
    need(initial.shape==ref.shape==high.shape==low.shape==(359373,),
         'state dimension')
    endpoint=normalized(high,ref)
    embedded=normalized(high,low)
    e=np.abs(high-ref)/(1e-7+1e-5*np.maximum(np.abs(high),np.abs(ref)))
    need(np.isclose(endpoint,result['mri']['endpoint_normalized'],rtol=1e-12),
         'endpoint receipt')
    need(np.isclose(embedded,result['mri']['embedded_3_2'],rtol=1e-12),
         'embedded receipt')
    need(result['mri']['full_calls']==5 and result['mri']['events']==7 and
         result['mri']['work_gate_valid'] is False,'scope/calls')
    bad=high.copy();bad[0]=np.nan
    try:normalized(bad,ref)
    except ValueError:pass
    else:raise ValueError('corruption accepted')
    return {'schema':'mri_frozen_capsule_verify_v1','status':'FAIL_FROZEN_FAST',
            'endpoint_normalized':endpoint,'embedded_normalized':embedded,
            'max_index':int(np.argmax(e)),'coordinates_over_one':int((e>1).sum()),
            'defect_reported_unrecomputed':result['mri']['sampled_defect_normalized'],
            'full_calls':5,'state_sha256':sha(path),'corruption_rejected':True,
            'scope':'Endpoint and embedded recomputed from portable arrays; live-oracle defect, parent neutrality, full operator work and speedup require local organism.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();need(not a.out.exists(),'output exists')
    report=verify(a.root)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))
