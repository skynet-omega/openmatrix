"""Recompute the static direct-PN coverage from its lossless projection."""
from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent
def need(ok,msg):
    if not ok:raise ValueError(msg)
def verify():
    with np.load(H/'reference/pn_coverage_projection.npz',allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    out=json.loads((H/'PN_COVERAGE.json').read_text());summary=[]
    need(len(d['PN_ids'])==out['author_annotated_ALPN']==686,'ALPN population')
    for k,tid in enumerate(d['DN_ids']):
        n=d['counts'][k];sel=n>0;w=d['weight'][k,sel];n=n[sel];present=d['present'][k,sel];sg=d['sign'][sel]
        denom=int(n.sum());ratios=w/n if len(n) else np.empty(0)
        summary.append(dict(DN_id=int(tid),ALPN_direct_edges=int(sel.sum()),synapses=denom,
            active_coefficient_coverage=None if not denom else int(n[w!=0].sum())/denom,
            missing_edges=int((~present).sum()),sign_inconsistent=int((np.sign(w)!=sg).sum()),
            weight_per_synapse_range=None if not len(n) else [float(ratios.min()),float(ratios.max())]))
    need(summary==out['summary'],'Static PN coverage reconstruction')
    return dict(coverage_reconstructed=True,DN_targets=6,ALPN_population=686,new_neural_ms=0,
        scope='Extracted static projection; full anatomy hashes retained separately')
if __name__=='__main__':print(json.dumps(verify()))
