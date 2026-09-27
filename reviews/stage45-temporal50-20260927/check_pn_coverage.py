"""Static direct ALPN-to-DN interface, not an activity or causal-path estimate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,time,resource,csv
import numpy as np
import pandas as pd
from scipy import sparse
H=Path(__file__).resolve().parent
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(15,16))
need(not (H/'PN_COVERAGE.json').exists(),'Preserve result')
plan=json.loads((H/'COVERAGE_PLAN.json').read_text())
for p,s in plan['source_hashes'].items():need(sha(p)==s,'Source mismatch')
old=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10')
nodes=pd.read_parquet(old/'nodes.parquet').sort_values('node_index')
ids=nodes.bodyId.to_numpy();alpn=nodes['class'].fillna('').to_numpy(str)=='ALPN'
counts=sparse.load_npz(old/'counts_pre_post.npz')
need(counts.shape==(len(ids),len(ids)),'Anatomical counts shape')
source=H.parents[1]/'investigacion/atribucion_entradas48_20260927_07/endpoint_inputs.npz'
with np.load(source,allow_pickle=False) as z:
    target_ids=z['target_ids'];target_rows=z['target_rows'];ptr=z['ptr'];pre=z['pre_rows'];w=z['sham_cuda_weights'];nt=z['pre_nt'];types=z['pre_types']
    need(np.array_equal(ids[pre],z['pre_ids']) and np.array_equal(ids[target_rows],target_ids),'CSR identities')
    for arm in ['dm1','profile','permuted']:need(np.array_equal(z[arm+'_cuda_weights'],w),'Condition-dependent static weight')
rows=[];summary=[];alpn_rows=np.flatnonzero(alpn)
for k,(tid,tr) in enumerate(zip(target_ids,target_rows)):
    n=counts[alpn_rows,int(tr)].toarray().ravel()
    selected=alpn_rows[n>0];n=n[n>0].astype(np.int64)
    positions={int(pre[j]):j for j in range(int(ptr[k]),int(ptr[k+1]))}
    entries=[]
    for pr,count in zip(selected,n):
        j=positions.get(int(pr));weight=0. if j is None else float(w[j]);emitter=str(nodes.iloc[int(pr)].nt_consensus_nt).lower()
        sign={'acetylcholine':1,'ach':1,'gaba':-1,'glutamate':-1,'glu':-1,'histamine':-1,'his':-1}.get(emitter,0)
        entry=dict(PN_id=int(ids[pr]),PN_type=str(nodes.iloc[int(pr)]['type']),DN_id=int(tid),synapses=int(count),NT=emitter,
            anatomical_edge_in_runtime=j is not None,coefficient_nonzero=weight!=0,effective_weight=weight,
            weight_per_synapse=weight/int(count),expected_model_sign=sign,model_sign_consistent=int(np.sign(weight))==sign,
            route='PN10208_legacy_general' if ids[pr]==10208 else 'legacy_generic_ALPN')
        entries.append(entry);rows.append(entry)
    denom=int(n.sum());active=sum(r['synapses'] for r in entries if r['coefficient_nonzero'])
    summary.append(dict(DN_id=int(tid),ALPN_direct_edges=len(entries),synapses=denom,
        active_coefficient_coverage=None if not denom else active/denom,
        missing_edges=sum(not r['anatomical_edge_in_runtime'] for r in entries),
        sign_inconsistent=sum(not r['model_sign_consistent'] for r in entries),
        weight_per_synapse_range=None if not entries else [min(r['weight_per_synapse'] for r in entries),max(r['weight_per_synapse'] for r in entries)]))
with (H/'PN_COVERAGE.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['PN_id','DN_id']);writer.writeheader();writer.writerows(rows)
out=dict(author_annotated_ALPN=int(alpn.sum()),summary=summary,edges=rows,
    scope='Static direct connections only; nonzero coefficient is not neural activity. Unknown/unannotated classes and indirect paths excluded. NT sign is model convention, not receptor validation.',
    inputs_sha256=plan['source_hashes'],plan_sha256=sha(H/'COVERAGE_PLAN.json'),CPU_s=time.process_time()-start,new_neural_ms=0)
(H/'PN_COVERAGE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['author_annotated_ALPN','summary','CPU_s','new_neural_ms']}))
