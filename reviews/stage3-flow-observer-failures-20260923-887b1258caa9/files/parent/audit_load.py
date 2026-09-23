"""Read-only zero-step audit of current protected stage-3 operator ownership."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
sys.path[:0] = [str(OLD/'work/motor14_20260922'), str(OLD/'work/motor13_20260922')]
from motor_runtime import load


def main() -> None:
    out = HERE/'zero_load_01'
    if out.exists():
        raise FileExistsError(out)
    obj = None
    try:
        obj, diagnostic, plan, Field, field, ports = load(out)
        h = obj.core.hybrid
        b = h.brain
        import pandas as pd
        metadata = pd.read_parquet(OLD/'data/male_v10/nodes.parquet', columns=['bodyId','type','rootSide','somaSide','instance'])
        ids = b.node_ids
        metadata = metadata.set_index('bodyId').reindex(ids)
        types = metadata['type'].fillna('').astype(str).to_numpy()
        names = ['ORN_DM1','DM1_lPN','MBON32','LAL170','LAL171','DNa02']
        groups = {name: np.flatnonzero(types == name) for name in names}
        groups['DNa02'] = np.flatnonzero(np.char.startswith(types.astype(str), 'DNa02'))
        selected = np.unique(np.concatenate(list(groups.values())))
        dynamic = np.asarray(getattr(h,'_dynamic_cache',{}).get('rows',[]))
        general = np.asarray(getattr(h,'_general_rows',[]))
        orn = np.asarray(getattr(h,'_orn_rows',[]))
        pvlp = np.asarray(getattr(h,'_pvlp_rows',[]))
        records = []
        for row in selected:
            sl = slice(b.W.indptr[row],b.W.indptr[row+1])
            pre = b.W.indices[sl]
            records.append({
                'bodyId':int(ids[row]),'row':int(row),'type':types[row],
                'rootSide':str(metadata.iloc[row]['rootSide']),
                'somaSide':str(metadata.iloc[row]['somaSide']),
                'instance':str(metadata.iloc[row]['instance']),
                'visual_mode':bool(h.visual_mask[row]),
                'incoming_edges':int(len(pre)),
                'from_PN10208':int(np.count_nonzero(ids[pre] == 10208)),
                'in_dynamic_targets':bool(np.isin(row,dynamic)),
                'in_general_PN_targets':bool(np.isin(row,general)),
                'in_orn_override_targets':bool(np.isin(row,orn)),
                'in_pvlp_override_targets':bool(np.isin(row,pvlp)),
                'incoming_signed_weight_sum':float(np.sum(h.weights64[sl])),
                'incoming_signed_weight_abs_sum':float(np.sum(np.abs(h.weights64[sl]))),
            })
        report = {
            'schema':'stage3_zero_step_operator_audit_v1',
            'scope':'Load current protected organism once; no neural/body advance and no intervention.',
            'checkpoint':plan['checkpoint'],
            'checkpoint_manifest_sha256':hashlib.sha256((OLD/plan['checkpoint']/'manifest.json').read_bytes()).hexdigest(),
            'hybrid_class':h.__class__.__name__,
            'hybrid_mro':[c.__name__ for c in h.__class__.__mro__],
            'time_ns':int(h.time_ns),
            'pending_sensors':obj.core.pending_sensors.tolist(),
            'n_neurons':int(b.n_neurons),
            'stored_edges':int(b.W.nnz),
            'groups':{name:ids[rows].tolist() for name,rows in groups.items()},
            'selected':records,
            'pn_scope':h.pn_online_manifest.get('scope'),
            'pn_general_scope':h.pn_online_manifest.get('general_outputs',{}).get('scope'),
            'orn_terminal_scope':h.pn_online_manifest.get('orn_peripheral_terminal',{}).get('scope'),
            'source_route_counts':{'dynamic_target_rows':int(len(dynamic)),'general_PN_target_rows':int(len(general)),'ORN_override_rows':int(len(orn)),'PVLP_override_rows':int(len(pvlp))},
            'units_warning':'Signed W times inherited transmission times caps is an abstract input to target/rate operator, not ionic current pA. Dynamic targets have different receptor equations.',
        }
        (out/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'hybrid_class':report['hybrid_class'],'pending_sensors':report['pending_sensors'],'source_route_counts':report['source_route_counts'],'selected':records},ensure_ascii=False,indent=2))
    finally:
        if obj is not None:
            obj.close()


if __name__ == '__main__':
    main()
