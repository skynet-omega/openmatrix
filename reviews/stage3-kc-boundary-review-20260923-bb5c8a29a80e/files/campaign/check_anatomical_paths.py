"""Check whether one divergent KC is anatomically isolated from the motor reader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

HERE=Path(__file__).resolve().parent
DATA=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10')
SOURCE_IDS=(81004,544736,45199)
TARGET_IDS=(10065,10118)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_to(pre,source,target):
    if target==source:return [source]
    if pre[target]<0:return None
    out=[target]
    while out[-1]!=source:
        out.append(int(pre[out[-1]]))
    return out[::-1]


def main():
    ids=np.load(DATA/'node_ids.npy',allow_pickle=False)
    table=pd.read_parquet(DATA/'nodes.parquet',columns=['bodyId','type'])
    types=dict(zip(table.bodyId.astype(int),table.type.fillna('UNANNOTATED').astype(str)))
    with np.load(DATA/'counts_pre_post.npz',allow_pickle=False) as z:
        indptr=z['indptr'].copy();indices=z['indices'].copy();counts=z['data'].copy()
    n=len(ids);lookup={int(k):i for i,k in enumerate(ids)}
    graph=csr_matrix((counts.astype(np.int32),indices,indptr),shape=(n,n))
    rows={}
    for minimum in (1,5,10):
        chosen=graph.copy()
        chosen.data=np.where(chosen.data>=minimum,1,0).astype(np.int8)
        chosen.eliminate_zeros()
        paths={}
        for source_id in SOURCE_IDS:
            root=lookup[source_id]
            _,pre=breadth_first_order(chosen,root,directed=True,return_predecessors=True)
            paths[str(source_id)]={}
            for target_id in TARGET_IDS:
                path=path_to(pre,root,lookup[target_id])
                paths[str(source_id)][str(target_id)]=None if path is None else [
                    {'id':int(ids[i]),'type':types.get(int(ids[i]),'UNANNOTATED'),
                     'synapses_to_next':int(graph[i,path[j+1]]) if j+1<len(path) else None}
                    for j,i in enumerate(path)]
        rows[str(minimum)]={'remaining_edges':int(chosen.nnz),'paths':paths}
    result={'schema':'kc81004_to_dnb05_anatomical_reachability_v1',
            'source':{'node_ids_sha256':sha(DATA/'node_ids.npy'),
                      'graph_sha256':sha(DATA/'counts_pre_post.npz'),
                      'nodes_sha256':sha(DATA/'nodes.parquet')},
            'presynaptic_ids':SOURCE_IDS,'targets':TARGET_IDS,
            'directed_paths_by_min_synapses':rows,
            'scope':'Anatomical paths in male v1.0 counts. Does not establish sign, effective transmission, timing, magnitude, or that this specific KC error affected DNb05 in the simulation.'}
    (HERE/'KC_MOTOR_REACHABILITY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{s:{t:None if v is None else [x['id'] for x in v] for t,v in targets.items()} for s,targets in row['paths'].items()} for k,row in rows.items()}))


if __name__=='__main__':main()
