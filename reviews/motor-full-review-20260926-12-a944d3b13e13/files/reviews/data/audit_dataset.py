"""Read-only CPU audit of the actual canonical graph and prepared model11 data.

No organism import, GPU allocation, simulation or historical write.
Each mode is capped at 120 s and 6 GiB virtual address space.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import hashlib, json, resource, signal, sys, time
resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120s CPU audit wall budget')))
signal.alarm(120)
START = time.perf_counter()
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[3]).resolve() if len(sys.argv) == 4 and sys.argv[2] == '--out' else HERE
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
ASTRA = Path('/home/daroch/AXIOMA_ASTRA')
CAN = OLD / 'data/male_v10'
BRAIN = OLD / 'work/stage234_settling_extension_20260915/settled_700ms/core_carrier/brain'
PREP = ASTRA / 'campanas/etapa45_navigation_wind_20260925_40/navigation_minus_filtered_wind_03/prepared_state'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

def stats(a):
    a = np.asarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype), size=int(a.size),
                finite=bool(np.isfinite(a).all()), zero=int(np.count_nonzero(a == 0)),
                negative=int(np.count_nonzero(a < 0)), positive=int(np.count_nonzero(a > 0)),
                quantiles=dict(zip(['min','p01','p25','median','p75','p99','max'],
                    map(float,np.quantile(a.astype(np.uint8) if a.dtype.kind == 'b' else a,[0,.01,.25,.5,.75,.99,1])))) if a.size else {})

def write(name, result):
    result['budget'] = dict(wall_limit_s=120, memory_limit_gib=6,
        wall_s=time.perf_counter()-START, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_loaded=False, historical_writes=False)
    result['script_sha256'] = sha(__file__)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('x') as f:
        f.write(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'output':str(OUT/name),'budget':result['budget']}))

def graph():
    prov=json.loads((CAN/'provenance.json').read_text())
    ids=np.load(CAN/'node_ids.npy',allow_pickle=False)
    nodes=pd.read_parquet(CAN/'nodes.parquet')
    C=sparse.load_npz(CAN/'counts_pre_post.npz')
    C.check_format(full_check=True)
    with np.load(BRAIN/'state.npz',allow_pickle=False) as z:
        brain_ids=z['node_ids']; sign=z['nt_sign']; nt=z['nt_labels']
    W=sparse.load_npz(BRAIN/'weights_post_pre.npz')
    T=C.T.tocsr()
    r=dict(scope='Static canonical C and loader brain W, plus campaign40 prepared effective main CSR; specialized operators are separate.',
        sources={str(p):sha(p) for p in [CAN/'provenance.json',CAN/'node_ids.npy',CAN/'nodes.parquet',CAN/'counts_pre_post.npz',BRAIN/'manifest.json',BRAIN/'state.npz',BRAIN/'weights_post_pre.npz',PREP/'effective_operator.json',PREP/'effective_operator.npz']},
        ids=dict(n=len(ids),strictly_increasing=bool(np.all(np.diff(ids)>0)),
            nodes_equal=bool(np.array_equal(ids,nodes.bodyId)),brain_equal=bool(np.array_equal(ids,brain_ids)),
            canonical_index_equal=bool(np.array_equal(nodes.node_index,np.arange(len(ids)))),
            examples=[dict(bodyId=int(ids[i]),row=int(i),type=str(nodes.iloc[i]['type'])) for i in [52,104,159,189]]),
        anatomy=dict(shape=list(C.shape),canonical=bool(C.has_canonical_format),nnz=C.nnz,
            synapses=int(C.data.sum()),weights=stats(C.data),self_pairs=int(np.count_nonzero(C.diagonal())),self_synapses=int(C.diagonal().sum()),
            source_csr_matches=bool(C.nnz==prov['csr']['stored_pairs'] and C.data.sum()==prov['csr']['count_sum'])),
        topology_post_pre_equal=bool(np.array_equal(T.indptr,W.indptr) and np.array_equal(T.indices,W.indices)),
        brain_weights=stats(W.data),nt_counts=dict(zip(*[a.tolist() for a in np.unique(nt,return_counts=True)])))
    for axis,label in [(1,'out'),(0,'in')]:
        degree=np.diff(C.indptr) if axis==1 else np.bincount(C.indices,minlength=len(ids))
        r['anatomy'][label+'_degree']=stats(degree)
        r['anatomy'][label+'_synapses']=stats(np.asarray(C.sum(axis=axis)).ravel())
    A=C.astype(bool)
    ncomp,labels=connected_components(A,directed=True,connection='strong')
    sz=np.bincount(labels);r['anatomy']['scc']=dict(components=int(ncomp),largest=int(sz.max()),largest_fraction=float(sz.max()/len(ids)),nontrivial_components=int(np.count_nonzero(sz>1)))
    reciprocal=A.multiply(A.T).tocsr()
    r['anatomy']['reciprocal_directed_edges_including_self']=int(reciprocal.nnz)
    r['anatomy']['reciprocal_directed_fraction']=float(reciprocal.nnz/C.nnz)
    del A,reciprocal
    classes,code=np.unique(nodes.superclass.astype(str).to_numpy(),return_inverse=True)
    sc=len(classes);pairs=np.zeros((sc,sc),dtype=np.int64);contacts=np.zeros_like(pairs)
    for lo in range(0,len(ids),4000):
        hi=min(lo+4000,len(ids));a,b=C.indptr[lo],C.indptr[hi]
        pre=np.repeat(code[lo:hi],np.diff(C.indptr[lo:hi+1]));keys=pre*sc+code[C.indices[a:b]]
        pairs += np.bincount(keys,minlength=sc*sc).reshape(sc,sc)
        contacts += np.bincount(keys,weights=C.data[a:b],minlength=sc*sc).reshape(sc,sc).astype(np.int64)
    r['superclass_connections']=[dict(pre=str(classes[i]),post=str(classes[j]),pairs=int(pairs[i,j]),synapses=int(contacts[i,j])) for i,j in zip(*np.nonzero(pairs))]
    r['within_superclass']=dict(pairs=int(np.trace(pairs)),synapses=int(np.trace(contacts)))
    initial=T.data.astype(np.float32)*np.float32(.03)*sign[W.indices]
    diff=W.data!=initial
    r['brain_vs_count_sign_rule']=dict(changed=int(np.count_nonzero(diff)),max_abs=float(np.max(np.abs(W.data-initial))),source_sign_zero_neurons=int(np.count_nonzero(sign==0)),initial_zero_edges=int(np.count_nonzero(initial==0)))
    del initial,diff,C,T
    manifest=json.loads((PREP/'effective_operator.json').read_text())
    with np.load(PREP/'effective_operator.npz',allow_pickle=False) as z:
        vals={k:z[v['__array__']] for k,v in manifest['values'].items()}
    cpu,gpu=vals['weights'],vals['cuda_weights']
    r['effective_parameters']={k:stats(v) for k,v in vals.items() if k not in ('weights','cuda_weights')}
    r['effective_weights']={k:stats(vals[k]) for k in ('weights','cuda_weights')}
    r['effective_cpu_gpu_difference']=dict(changed=int(np.count_nonzero(cpu!=gpu)),max_abs=float(np.max(np.abs(cpu-gpu))))
    r['brain_effective_gpu_difference']=dict(changed=int(np.count_nonzero(W.data.astype(float)!=gpu)),max_abs=float(np.max(np.abs(W.data.astype(float)-gpu))))
    r['effective_gpu_source_sign_disagreement_edges']=int(np.count_nonzero((gpu!=0)&(np.sign(gpu)!=sign[W.indices])))
    support=sparse.csr_matrix(((gpu!=0),W.indices,W.indptr),shape=W.shape);support.eliminate_zeros()
    ncomp,labels=connected_components(support,directed=True,connection='strong');sz=np.bincount(labels)
    r['effective_main_scc']=dict(components=int(ncomp),largest=int(sz.max()),largest_fraction=float(sz.max()/len(ids)),active_edges=int(support.nnz))
    write('GRAPH_AUDIT.json',r)

def state():
    s=json.loads((PREP/'session.json').read_text());h=s['hybrid']
    arrays=np.load(PREP/'session.npz',allow_pickle=False)
    def compact(x, depth=0):
        if isinstance(x,dict):
            if set(x)=={'__array__'}:
                a=arrays[x['__array__']]
                return dict(array=x['__array__'],**stats(a)) if a.dtype.kind in 'bifu' else dict(shape=list(a.shape),dtype=str(a.dtype))
            return {k:compact(v,depth+1) for k,v in x.items()}
        if isinstance(x,list):
            if len(x)>32:
                return dict(list_length=len(x),first=x[:3],last=x[-3:])
            return [compact(v,depth+1) for v in x]
        return x
    records={k:compact(v) for k,v in h.items()}
    r=dict(scope='Serialized campaign40 prepared state at the exact model11 starting point; inspection only, no constructor or simulation.',
        sources={str(p):sha(p) for p in [PREP/'MANIFEST.json',PREP/'session.json',PREP/'session.npz']},
        session=dict(time_ns=s['time_ns'],start_ns=s['start_ns'],config=compact(s['config']),
            plasticity=compact(s['plasticity']),hybrid=records),
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__))
    write('STATE_AUDIT.json',r)

if __name__=='__main__':
    expected=OUT/({'graph':'GRAPH_AUDIT.json','state':'STATE_AUDIT.json'}[sys.argv[1]])
    if expected.exists():
        raise FileExistsError('Preserve existing receipt; use --out with a new directory: '+str(expected))
    {'graph':graph,'state':state}[sys.argv[1]]()
