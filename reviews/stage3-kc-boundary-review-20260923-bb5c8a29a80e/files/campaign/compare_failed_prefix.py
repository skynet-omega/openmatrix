"""Use the preserved failed observer run as an uninstrumented 1-ms KC prefix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
FAILED=HERE/'native_capture_on_01/final_state'
CAPTURED=HERE/'native_capture_on_02/kc/after_001ms.npz'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    run=json.loads((HERE/'native_capture_on_01/RESULT.json').read_text())
    if run['status']!='INCOMPLETE' or run['error']['type']!='AttributeError' or run['checkpoint']['status']!='COMPLETE_SERIALIZED_NOT_RESUME_VALIDATED':
        raise ValueError('Wrong preserved failure')
    meta=json.loads((FAILED/'session.json').read_text())['hybrid']
    results={}
    with np.load(FAILED/'session.npz',allow_pickle=False) as full, np.load(CAPTURED,allow_pickle=False) as compact:
        for group,prefix in (('kc_spatial_state','somatic_'),('kc_axonal_state','axonal_')):
            for key in compact.files:
                if not key.startswith(prefix):continue
                name=key[len(prefix):]
                a=full[meta[group][name]['__array__']]
                b=compact[key]
                results[key]={'shape':list(a.shape),'exact':bool(np.array_equal(a,b,equal_nan=a.dtype.kind in 'fc'))}
    result={'schema':'kc_failed_prefix_neutrality_v1','selected_fields':results,
            'all_exact':all(x['exact'] for x in results.values()),
            'source_sha256':{'failed_session_json':sha(FAILED/'session.json'),
                             'failed_session_npz':sha(FAILED/'session.npz'),
                             'capture_after_1ms':sha(CAPTURED)},
            'scope':'KC fields only. The failed observer advanced one full organism ms and stopped before reading KC; it is not a 2-ms final-state control.'}
    (HERE/'PREFIX_NEUTRALITY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all_exact':result['all_exact'],'fields':len(results)}))
    if not result['all_exact']:raise SystemExit(2)


if __name__=='__main__':main()
