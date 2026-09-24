"""CPU-only complete capsule verification; needs no parent data, CuPy or GPU."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from event_sparse import array_digest, file_digest, need, prepare_ports, project_cpu


def verify(directory):
    start=time.perf_counter()
    expected=json.loads((directory/'OUTPUT_HASHES.json').read_text())
    for name,value in expected.items():
        need(file_digest(directory/name)==value,'Changed output: '+name)
    frozen=json.loads((directory/'FROZEN.json').read_text())
    for name,value in frozen['sources'].items():
        need(file_digest(Path(__file__).parent/name)==value,'Changed source/contract: '+name)
    result=json.loads((directory/'RESULT.json').read_text())
    contract=json.loads((Path(__file__).parent/'CONTRACT.json').read_text())
    with np.load(directory/'capsule.npz',allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    with np.load(directory/'currents.npz',allow_pickle=False) as z:
        r={k:z[k] for k in z.files}
    ptr,src,w=a['ptr'],a['source_index'],a['weights']
    rows=a['receiver_rows'];ns=len(a['source_rows']);nr=len(rows)
    need(ptr.shape==(nr+1,) and ptr[0]==0 and ptr[-1]==len(src)==len(w),'Capsule CSR layout')
    need(np.all(np.diff(ptr)>0) and np.all((src>=0)&(src<ns)),'Capsule CSR domain')
    erows=np.repeat(np.arange(nr),np.diff(ptr))
    receiver_visual=a['receiver_visual'][erows]
    source_visual=a['source_visual'][src]
    nonvisual_allowed=bool(a['connected'])|~source_visual
    data=dict(rows=a['source_rows'],q=a['q'],s=a['s'],tau=a['tau_q'],ts=float(a['tau_s']),
              event_rows=np.repeat(np.arange(ns),np.diff(a['event_ptr'])),times=a['event_times'],
              jumps=a['event_jumps'],setop=a['event_set'],posts=a['event_posts'],
              duration_ns=round(float(a['duration'])*1e9))
    ports=prepare_ports(data,result['full_sources'])
    need(ports.version==result['event_version'],'Port event version')
    atol,rtol=contract['gates']['current_absolute_tolerance'],contract['gates']['current_relative_tolerance']
    def normalized(x,y):
        return np.abs(x-y)/(atol+rtol*np.maximum(np.abs(x),np.abs(y)))
    m=len(a['query_times'])
    need(r['full'].shape==r['reduced'].shape==(m,nr,2),'Saved currents layout')
    need(all(np.isfinite(v).all() for v in r.values()),'Nonfinite raw result')
    cpu_current_max_norm=0.;port_max_abs=0.
    for k,t in enumerate(a['query_times']):
        q,s=project_cpu(ports,float(t))
        port_max_abs=max(port_max_abs,float(np.max(np.abs(q-r['q_gpu'][k]))),float(np.max(np.abs(s-r['s_gpu'][k]))))
        # Independent sequential scatter sums via numpy.bincount. The CUDA
        # implementation instead uses one fixed warp per receiving row.
        v=(w*float(a['scale']))*s[src]
        u=w*(s[src]*a['source_caps'][src])
        pos=np.where(receiver_visual,np.maximum(v,0.),np.where(nonvisual_allowed,u,0.))
        neg=np.where(receiver_visual,np.maximum(-v,0.),0.)
        expected_current=np.stack((np.bincount(erows,weights=pos,minlength=nr),
                                   np.bincount(erows,weights=neg,minlength=nr)),axis=1)
        for arm in ('full','reduced'):
            cpu_current_max_norm=max(cpu_current_max_norm,float(normalized(expected_current,r[arm][k]).max()))
    errors=normalized(r['full'],r['reduced']);absolute=np.abs(r['full']-r['reduced'])
    need(float(errors.max())==result['current_max_normalized'],'Reported current error differs from raw')
    need(np.array_equal(errors.max(axis=(1,2)),result['query_max_normalized']),'Per-query normalized errors differ')
    need(np.array_equal(absolute.max(axis=(1,2)),result['query_max_abs']),'Per-query absolute errors differ')
    # Reconstitute full receiver layout to validate hashes and recorded zero rows.
    dense=np.zeros((m,result['full_sources'],2),dtype=np.float64)
    for arm in ('full','reduced'):
        dense[:,rows,:]=r[arm]
        need(array_digest(dense)==result[arm+'_current_sha256'],'Full-layout current hash mismatch')
    del dense
    edge_ratio=m*result['full_edges']/(result['full_edges']+m*len(w))
    need(edge_ratio==result['edge_work_ratio_including_one_partition'],'Wrong setup-aware edge ratio')
    gates=dict(current=bool(errors.max()<=contract['gates']['current_normalized_error_max']),
               edge_work=bool(edge_ratio>=contract['gates']['edge_work_reduction_min']),
               independent_cpu_current=bool(cpu_current_max_norm<=1.),ports=bool(port_max_abs<=1e-12))
    need(all(gates.values()),'Recomputed scientific gate failed')
    return dict(schema='event_sparse_capsule_verification_v1',queries=m,active_receivers=nr,edges=len(w),
                gates_recomputed_from_arrays=gates,current_max_normalized=float(errors.max()),
                independent_cpu_current_max_normalized=cpu_current_max_norm,ports_cpu_gpu_max_abs=port_max_abs,
                edge_ratio_including_one_partition=edge_ratio,wall_s=time.perf_counter()-start,optimized=not __debug__,
                full_graph_reexecuted=False,scope='All 60 port currents independently recomputed on CPU from compact real CSR; full CSR GPU outputs retained and hashed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();result=verify(args.directory)
    if args.out:
        need(not args.out.exists(),'Verification output already exists')
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
