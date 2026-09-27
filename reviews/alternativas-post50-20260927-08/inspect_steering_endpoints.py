"""Exploratory named steering types in existing snapshots; no neural simulation."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,time,resource
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
ROOT=H.parents[1]
TYPES=['DNa01','DNa02','DNb05','DNb06','DNg13']
ARMS=['sham','dm1','profile','permuted']
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(20,21))
out=H/'STEERING_ENDPOINTS.json';need(not out.exists(),'Preserve prior result')
nodes_path=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/nodes.parquet')
nodes=pd.read_parquet(nodes_path).sort_values('node_index')
need(np.array_equal(nodes.node_index.to_numpy(),np.arange(len(nodes))),'Node order')
selected=nodes[nodes['type'].isin(TYPES)].copy();rows=selected.node_index.to_numpy()
need(len(selected)==10,'Five named bilateral pairs required')
source_hashes={str(nodes_path):sha(nodes_path)};values={};qsel={}
for arm in ARMS:
    folder=ROOT/'campanas/etapa45_composicion_20260927_48'/arm
    state=folder/'final_state';manifest=json.loads((state/'MANIFEST.json').read_text())
    for name in ['published.json','published.npz','session.json','session.npz']:
        p=state/name;expected=manifest['files'][name]
        expected=expected['sha256'] if isinstance(expected,dict) else expected
        need(sha(p)==expected,'Frozen snapshot hash '+arm+'/'+name)
        source_hashes[str(p.relative_to(ROOT))]=expected
    meta=json.loads((state/'published.json').read_text())
    session=json.loads((state/'session.json').read_text())
    with np.load(state/'session.npz',allow_pickle=False) as z:
        q=z[session['hybrid']['state']['__array__']][:len(nodes)]
        vi= z[session['hybrid']['visual_ids']['__array__']]
        need(not np.intersect1d(vi,selected.bodyId.to_numpy()).size,'Selected DN is visual specialized')
        need(q.shape==(len(nodes),) and np.isfinite(q).all(),'Finite complete native vector')
        qsel[arm]=q[rows].copy()
    with np.load(folder/'blocks/02901_03000ms/traces.npz',allow_pickle=False) as z:
        actual=z['DN_q_actual'][-1]
        for k,tid in enumerate([10045,10056,10118,10065]):
            need(q[np.flatnonzero(nodes.bodyId.to_numpy()==tid)[0]]==actual[k],'Published/trace DN mismatch')
    values[arm]=dict(time_ns=meta['time_ns'],selected_q=qsel[arm].tolist())
table=[]
for t in TYPES:
    pair=selected[selected['type']==t]
    need(sorted(pair.somaSide.tolist())==['L','R'],'Pair sides '+t)
    l=int(np.flatnonzero((selected['type']==t)&(selected.somaSide=='L'))[0])
    r=int(np.flatnonzero((selected['type']==t)&(selected.somaSide=='R'))[0])
    table.append(dict(type=t,ids_LR=[int(selected.iloc[l].bodyId),int(selected.iloc[r].bodyId)],
      q_LR={a:[float(qsel[a][l]),float(qsel[a][r])] for a in ARMS},
      L_minus_R_delta_from_sham={a:float((qsel[a][l]-qsel[a][r])-(qsel['sham'][l]-qsel['sham'][r])) for a in ARMS if a!='sham'}))
np.savez_compressed(H/'steering_endpoints_projection.npz',body_ids=selected.bodyId.to_numpy(),rows=rows,types=selected['type'].to_numpy(str),sides=selected.somaSide.to_numpy(str),**qsel)
result=dict(scope='Exploratory four endpoints, no temporal/causal inference or material pass criterion; types fixed from Yang before inspection, not selected by response.',table=table,source_hashes=source_hashes,
 CPU_s=time.process_time()-start,new_neural_ms=0,GPU_calls=0,readout_changed=False)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,allow_nan=False))
