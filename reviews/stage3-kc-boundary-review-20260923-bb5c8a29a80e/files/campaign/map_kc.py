"""Resolve local KC state indices against the frozen anatomical identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'etapa3_dnb05_native_20260923_12'
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10')


def main():
    root=PARENT/'native_sham_20_01/final_state'
    metadata=json.loads((root/'session.json').read_text())
    m=metadata['hybrid']['kc_axonal_manifest']
    with np.load(root/'session.npz',allow_pickle=False) as z:
        ids=z[m['ids']['__array__']].copy()
        rows=z[m['rows']['__array__']].copy()
    nodes=pd.read_parquet(OLD/'nodes.parquet',columns=['bodyId','type'])
    brain_ids=np.load(OLD/'node_ids.npy',allow_pickle=False)
    if not np.array_equal(brain_ids[rows],ids):raise ValueError('KC local/global mapping mismatch')
    types=dict(zip(nodes.bodyId.astype(int),nodes.type.fillna('UNANNOTATED').astype(str)))
    gate=json.loads((PARENT/'GATE.json').read_text())
    boundary=json.loads((HERE/'BOUNDARY.json').read_text())
    local=set()
    for x in gate['kc_errors'].values():local.add(x['index'][0])
    for ms in ('1','2'):
        for x in boundary['state_max_abs_by_ms'][ms].values():local.add(x['index'][0])
    event_ids=(76431,81004,38549)
    for identity in event_ids:
        matches=np.flatnonzero(ids==identity)
        if len(matches)==1:local.add(int(matches[0]))
    mapped={str(i):{'bodyId':int(ids[i]),'global_row':int(rows[i]),'type':types.get(int(ids[i]),'UNANNOTATED')}
            for i in sorted(local)}
    result={'schema':'kc_local_to_anatomical_identity_v1',
            'source_session_json_sha256':hashlib.sha256((root/'session.json').read_bytes()).hexdigest(),
            'source_session_npz_sha256':hashlib.sha256((root/'session.npz').read_bytes()).hexdigest(),
            'node_ids_sha256':hashlib.sha256((OLD/'node_ids.npy').read_bytes()).hexdigest(),
            'local_count':len(ids),'selected_local_to_global':mapped,
            'event_local_indices':{str(identity):int(np.flatnonzero(ids==identity)[0]) if np.sum(ids==identity)==1 else None for identity in event_ids},
            'scope':'Indices for selected KC state maxima and first event producers; anatomy identity only, no causal attribution.'}
    (HERE/'KC_LOCAL_MAPPING.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'selected_count':len(mapped),'event_local_indices':result['event_local_indices']}))


if __name__=='__main__':main()
