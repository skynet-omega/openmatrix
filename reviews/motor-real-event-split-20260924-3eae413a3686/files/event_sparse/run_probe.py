"""One bounded 60-query real probe. All query buffers are resident before timing."""
import argparse
import gc
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import numpy as np
from event_sparse import (array_digest, decode_npz, file_digest, need,
                          partition_csr, prepare_ports, project_cpu, current_cpu)
from gpu_operator import GpuPortOperator
from test_cpu import fixture_inputs, run_fixtures


HERE=Path(__file__).resolve().parent


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def unique_boolean_from_json_lines(path,key):
    """Bounded-memory read of one uniquely named captured configuration flag."""
    pattern=re.compile(r'^\s*"'+re.escape(key)+r'"\s*:\s*(true|false)\s*,?\s*$')
    matches=[]
    with open(path) as f:
        for line_number,line in enumerate(f,1):
            match=pattern.fullmatch(line.rstrip('\n'))
            if match:
                matches.append((line_number,match.group(1)=='true'))
    need(len(matches)==1,'Expected a unique boolean configuration key: '+key)
    return matches[0]


def run(args):
    start=time.perf_counter()
    args.out.mkdir(parents=True,exist_ok=False)
    contract=json.loads((HERE/'CONTRACT.json').read_text())
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen 120-second wall limit')))
    signal.alarm(contract['budget']['gpu_probe_wall_seconds'])
    source_names=['CONTRACT.json','event_sparse.py','sparse_kernels.cu','gpu_operator.py','test_cpu.py','run_probe.py','verify_capsule.py']
    inputs=['block_events.json','block_events.npz','block_inputs.json','trace_clock.npz','trace_state.npz',
            'EFFECTIVE_ARRAY_PATHS.json','effective_gpu_arrays.npz','EPOCHS.json','final_state/session.json']
    manifest=dict(schema='event_sparse_frozen_inputs_v1',
                  sources={name:file_digest(HERE/name) for name in source_names},
                  capture_inputs={name:file_digest(args.capture/name) for name in inputs})
    save_json(args.out/'FROZEN.json',manifest)
    run_fixtures()
    mapping=json.loads((args.capture/'EFFECTIVE_ARRAY_PATHS.json').read_text())
    with np.load(args.capture/'effective_gpu_arrays.npz',allow_pickle=False) as z:
        ptr,indices,weights,caps,visual,photo_rows=(z[mapping[k]] for k in
            ('cuda/indptr','cuda/indices','cuda/weights','cuda/caps','cuda/visual','cuda/pi'))
    n=len(ptr)-1
    ports=prepare_ports(decode_npz(args.capture/'block_events'),n)
    with np.load(args.capture/'trace_clock.npz',allow_pickle=False) as z:
        times=z['query_s'];epochs=z['epoch'];start_ns=int(z['global_start_ns'])
    need(len(times)==contract['budget']['real_query_times'] and len(np.unique(epochs))==1,'Wrong real query cohort')
    parameters=json.loads((args.capture/'block_inputs.json').read_text())['parameters']
    scale=parameters['conductance_per_stored_weight']
    connected_line,connected=unique_boolean_from_json_lines(args.capture/'final_state/session.json','visual_output_connected')
    partition_start=time.perf_counter()
    part=partition_csr(ptr,indices,weights,ports,n)
    partition_s=time.perf_counter()-partition_start
    continuous_digest=array_digest(part.continuous_ptr,part.continuous_positions)
    continuous_edges=len(part.continuous_positions)
    # Continuously driven sources remain represented by the complement; this
    # probe never pretends to evaluate their current from the port-only release.
    part.continuous_positions=np.empty(0,dtype=np.int64)
    q_cpu,s_cpu=zip(*(project_cpu(ports,float(t)) for t in times))
    q_cpu,s_cpu=np.array(q_cpu),np.array(s_cpu)
    transmission_start=n+2*len(photo_rows)
    with np.load(args.capture/'trace_state.npz',allow_pickle=False) as z:
        trace=z['values']
        trace_q_error=float(np.max(np.abs(trace[:,ports.rows]-q_cpu)))
        trace_s_error=float(np.max(np.abs(trace[:,transmission_start+ports.rows]-s_cpu)))
    del trace
    need(trace_q_error<=1e-12 and trace_s_error<=1e-12,'Reconstructed ports do not match captured state')
    cpu_preparation_s=time.perf_counter()-start
    # CuPy/NVRTC writes cache only inside the authorized output directory.
    os.environ['CUPY_CACHE_DIR']=str(args.out/'cuda_cache')
    import cupy as cp
    free_before,total_vram=cp.cuda.runtime.memGetInfo()
    gpu_start=time.perf_counter()
    # The same four-fixture synthetic cohort, with four CUDA query timestamps.
    data,sptr,sidx,sw,scaps,svisual=fixture_inputs()
    sp=prepare_ports(data,len(scaps));spl=partition_csr(sptr,sidx,sw,sp,len(scaps))
    st=np.array([.00069,.0007,.0008,.0004])
    synthetic=GpuPortOperator(cp,sptr,sidx,sw,scaps,svisual,sp,spl,st,1./30.,False)
    for k in range(len(st)):
        synthetic.project(k,spl.weight_version,sp.version)
        synthetic.current(k,False);synthetic.current(k,True)
    synthetic.stream.synchronize()
    sf,sr=cp.asnumpy(synthetic.full_out),cp.asnumpy(synthetic.reduced_out)
    sq,ss=cp.asnumpy(synthetic.q_trace),cp.asnumpy(synthetic.s_trace)
    synthetic_error=0.
    for k,t in enumerate(st):
        q,s=project_cpu(sp,float(t));release=np.zeros(len(scaps));release[sp.rows]=s
        reference=current_cpu(sptr,sidx,sw,release,scaps,svisual,1./30.,False)
        error=max(float(np.max(np.abs(reference-sf[k]))),float(np.max(np.abs(reference-sr[k]))),
                  float(np.max(np.abs(q-sq[k]))),float(np.max(np.abs(s-ss[k]))))
        synthetic_error=max(synthetic_error,error)
    need(synthetic_error<=1e-12,'Synthetic CUDA mismatch')
    synthetic.close();del synthetic;gc.collect()
    setup_start=time.perf_counter()
    op=GpuPortOperator(cp,ptr,indices,weights,caps,visual,ports,part,times,scale,connected)
    setup_s=time.perf_counter()-setup_start
    need(op.pool.total_bytes()<=contract['budget']['extra_vram_bytes'],'Private VRAM budget')
    # Allocate events before the hot loop. No host/device array copy or sync occurs
    # between queries; integer query indices select already resident timestamps.
    event_sets=[[cp.cuda.Event() for _ in range(6)] for _ in times]
    enqueue_start=time.perf_counter()
    for k,events in enumerate(event_sets):
        ps,pe,fs,fe,rs,re=events
        ps.record(op.stream);op.project(k,part.weight_version,ports.version);pe.record(op.stream)
        for reduced in ((False,True) if k%2==0 else (True,False)):
            begin,end=(rs,re) if reduced else (fs,fe)
            begin.record(op.stream);op.current(k,reduced);end.record(op.stream)
    op.stream.synchronize()
    query_loop_wall_s=time.perf_counter()-enqueue_start
    gpu_ms=np.array([[cp.cuda.get_elapsed_time(ev[0],ev[1]),cp.cuda.get_elapsed_time(ev[2],ev[3]),
                      cp.cuda.get_elapsed_time(ev[4],ev[5])] for ev in event_sets])
    full=cp.asnumpy(op.full_out);reduced=cp.asnumpy(op.reduced_out)
    q_gpu,s_gpu=cp.asnumpy(op.q_trace),cp.asnumpy(op.s_trace)
    free_after,_=cp.cuda.runtime.memGetInfo()
    active=part.active_rows
    inactive=np.ones(n,dtype=bool);inactive[active]=False
    outside_nonzeros=int(np.count_nonzero(full[:,inactive,:])+np.count_nonzero(reduced[:,inactive,:]))
    need(outside_nonzeros==0,'Unexpected contribution outside selected receptors')
    full_digest,reduced_digest=array_digest(full),array_digest(reduced)
    full,reduced=full[:,active,:].copy(),reduced[:,active,:].copy()
    need(all(np.isfinite(a).all() for a in (full,reduced,q_gpu,s_gpu,gpu_ms)),'Nonfinite measured values')
    delta=np.abs(full-reduced)
    denominator=contract['gates']['current_absolute_tolerance']+contract['gates']['current_relative_tolerance']*np.maximum(np.abs(full),np.abs(reduced))
    normalized=delta/denominator
    q_error=float(np.max(np.abs(q_gpu-q_cpu)));s_error=float(np.max(np.abs(s_gpu-s_cpu)))
    need(q_error<=1e-12 and s_error<=1e-12,'CUDA analytic port reconstruction mismatch')
    # Compact complete numerical capsule: 6342 receiver rows x 4062 port sources,
    # not a dense grouped matrix, and all 60 baseline/reduced currents are kept.
    lookup=np.full(n,-1,dtype=np.int32);lookup[ports.rows]=np.arange(len(ports.rows),dtype=np.int32)
    compact_ptr=np.r_[part.port_ptr[active],len(part.port_sources)].astype(np.int64)
    np.savez_compressed(args.out/'capsule.npz',ptr=compact_ptr,source_index=lookup[part.port_sources],weights=part.port_weights,
        source_rows=ports.rows,receiver_rows=active,source_caps=caps[ports.rows],source_visual=visual[ports.rows],
        receiver_visual=visual[active],q=ports.q,s=ports.s,tau_q=ports.tau,tau_s=np.array(ports.ts),
        event_ptr=ports.ptr,event_times=ports.times,event_jumps=ports.jumps,event_set=ports.sets,event_posts=ports.posts,
        duration=np.array(ports.duration),query_times=times,scale=np.array(scale),connected=np.array(connected))
    np.savez_compressed(args.out/'currents.npz',full=full,reduced=reduced,q_gpu=q_gpu,s_gpu=s_gpu,gpu_ms=gpu_ms)
    query_max_abs=delta.max(axis=(1,2));query_max_normalized=normalized.max(axis=(1,2))
    edges,port_edges=len(indices),len(part.port_sources)
    work_steady=edges/port_edges
    work_one_partition=len(times)*edges/(edges+len(times)*port_edges)
    vram_bytes=max(int(op.pool.total_bytes()),int(free_before-free_after))
    result=dict(schema='event_sparse_real_probe_v1',classification='PROMETEDOR_NO_CONFIRMADO',
        scope='Fixed captured base CSR snapshot only; PN/APL/other overrides require separate versioned extensions; no engine admission',
        full_sources=n,port_sources=len(ports.rows),unique_tau=len(np.unique(ports.tau)),active_receivers=len(active),
        full_edges=edges,port_edges=port_edges,continuous_edges=continuous_edges,queries=len(times),
        events=len(ports.times),time_backtracks=int(np.count_nonzero(np.diff(times)<0)),epoch=int(epochs[0]),start_ns=start_ns,
        weight_version=part.weight_version,event_version=ports.version,continuous_partition_sha256=continuous_digest,
        full_current_sha256=full_digest,reduced_current_sha256=reduced_digest,outside_receiver_nonzeros=outside_nonzeros,
        current_max_abs=float(delta.max()),current_max_normalized=float(normalized.max()),
        query_max_abs=query_max_abs.tolist(),query_max_normalized=query_max_normalized.tolist(),
        current_gate=bool(normalized.max()<=contract['gates']['current_normalized_error_max']),
        edge_gate=bool(work_one_partition>=contract['gates']['edge_work_reduction_min']),
        edge_work_ratio_steady=work_steady,edge_work_ratio_including_one_partition=work_one_partition,
        partition_s=partition_s,cpu_preparation_s=cpu_preparation_s,gpu_setup_s=setup_s,
        query_loop_wall_s=query_loop_wall_s,projection_gpu_ms=float(gpu_ms[:,0].sum()),
        full_csr_gpu_ms=float(gpu_ms[:,1].sum()),reduced_csr_gpu_ms=float(gpu_ms[:,2].sum()),
        steady_gpu_time_ratio=float(gpu_ms[:,1].sum()/gpu_ms[:,2].sum()),
        query_time_ratio_including_shared_projection=float((gpu_ms[:,0].sum()+gpu_ms[:,1].sum())/(gpu_ms[:,0].sum()+gpu_ms[:,2].sum())),
        gpu_probe_including_setup_s=time.perf_counter()-gpu_start,
        q_vs_cpu_max_abs=q_error,s_vs_cpu_max_abs=s_error,q_vs_trace_max_abs=trace_q_error,s_vs_trace_max_abs=trace_s_error,
        synthetic_gpu_queries=len(st),synthetic_gpu_max_abs=synthetic_error,private_gpu_bytes=int(op.pool.total_bytes()),
        extra_gpu_bytes_observed=vram_bytes,total_vram_bytes=int(total_vram),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        python=sys.version,cupy=cp.__version__,numpy=np.__version__,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        current_channels=['signed nonvisual or positive visual','negative visual magnitude'],
        source_csr_version_scope='Snapshot cuda/weights after captured coefficient return; not all transient override weights',
        connected_flag_provenance=dict(file='final_state/session.json',key='visual_output_connected',line=connected_line,value=connected),
        complete_engine=False,stage_admission=False,organism_interactions=0)
    result['baseline_csr_s']=result['full_csr_gpu_ms']/1000.
    result['candidate_partition_plus_csr_s']=partition_s+result['reduced_csr_gpu_ms']/1000.
    result['amortized_time_ratio_60_queries']=result['baseline_csr_s']/result['candidate_partition_plus_csr_s']
    result['wall_s']=time.perf_counter()-start
    need(vram_bytes<=contract['budget']['extra_vram_bytes'],'Extra VRAM budget exceeded')
    need(result['peak_rss_bytes']<=contract['budget']['ram_bytes'],'RSS budget exceeded')
    result['output_bytes']=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    need(result['output_bytes']<=contract['budget']['output_bytes'],'Disk budget exceeded')
    save_json(args.out/'RESULT.json',result)
    save_json(args.out/'OUTPUT_HASHES.json',{name:file_digest(args.out/name) for name in ('FROZEN.json','capsule.npz','currents.npz','RESULT.json')})
    op.close();signal.alarm(0)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as e:
        if args.out.is_dir():
            save_json(args.out/'ERROR.json',dict(type=type(e).__name__,message=str(e)))
        raise
