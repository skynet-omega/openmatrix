"""Extract the first two weighted CSR influence layers from the frozen block."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAP=ROOT/'motor_nuevo/multirate_real_20260924_01/capture_01'


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    output=HERE/'EVENT_CONE_CAPSULE.npz'
    manifest=HERE/'EVENT_CONE_CAPSULE_MANIFEST.json'
    need(not output.exists() and not manifest.exists(),'unique output')
    report=json.loads((HERE/'EVENT_CONE_RESULT.json').read_text())
    mapping=json.loads((CAP/'EFFECTIVE_ARRAY_PATHS.json').read_text())
    with np.load(CAP/'effective_gpu_arrays.npz',allow_pickle=False) as z:
        ptr=z[mapping['cuda/indptr']]
        indices=z[mapping['cuda/indices']]
        weights=z[mapping['cuda/weights']]
    sources=np.asarray(report['sources'],dtype=np.int32)
    active=weights!=0
    def select(rows):
        positions=np.flatnonzero(active & np.isin(indices,rows))
        posts=(np.searchsorted(ptr,positions,side='right')-1).astype(np.int32)
        return indices[positions].astype(np.int32),posts,weights[positions]
    pre1,post1,weight1=select(sources)
    first=np.setdiff1d(np.unique(post1),sources).astype(np.int32)
    pre2,post2,weight2=select(first)
    need(len(first)==report['layers'][0]['new_neurons'] and
         len(pre1)==report['event_source_outgoing_active_edges'] and
         len(pre2)==report['layers'][0]['outgoing_active_edges_from_new'],
         'cone layer mismatch')
    np.savez_compressed(output,source_rows=sources,first_hop_rows=first,
        edge1_pre=pre1,edge1_post=post1,edge1_weight=weight1,
        edge2_pre=pre2,edge2_post=post2,edge2_weight=weight2,
        target_row=np.asarray([report['target_row']],dtype=np.int32))
    data={'schema':'event_cone_capsule_manifest_v1',
          'capsule_sha256':sha(output),'capsule_bytes':output.stat().st_size,
          'builder_sha256':sha(Path(__file__)),
          'result_sha256':sha(HERE/'EVENT_CONE_RESULT.json'),
          'full_graph_sha256':sha(CAP/'effective_gpu_arrays.npz'),
          'n_source_rows':len(sources),'n_first_hop_rows':len(first),
          'n_edge1':len(pre1),'n_edge2':len(pre2),
          'scope':'First two layers only, from nonzero base CSR; excludes specialized effective operator and neuron dynamics.'}
    manifest.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(data))


if __name__=='__main__':main()
