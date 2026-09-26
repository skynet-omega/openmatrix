"""Bounded read-only inspection of the restored initial motor pathway.

No integration, coefficient call, parameter sweep or biological perturbation.
The generic CSR current is reported as effective only when no relevant dynamic
writer or coefficient override touches the inspected postsynaptic row.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,sys,time,signal,hashlib
import numpy as np
ROOT=Path('/home/daroch/AXIOMA_ASTRA')
HERE=Path(__file__).resolve().parent
BASE=ROOT/'motor_nuevo/full_pipeline_review_20260925_12'
sys.path.insert(0,str(BASE))
import run_trial as ref
sys.path.insert(0,str(ROOT/'motor_nuevo/causal_runtime_20260922'))
from motor_runtime import load
from restore_prepared import restore

def stop(*_):raise TimeoutError('Prepared inspection exceeded180seconds; no retry')
signal.signal(signal.SIGALRM,stop);signal.alarm(180)
start=time.monotonic();obj=None
try:
    import cupy as cp
    import pandas as pd
    obj,*_=load(HERE/'prepared_probe_static_final')
    restored=restore(obj,ref.PREFIX/'prepared_state')
    h=obj.core.hybrid;b=h.brain
    ids=np.array([10045,10056,10118,10065],np.int64);rows=np.searchsorted(b.node_ids,ids)
    if not np.array_equal(b.node_ids[rows],ids):raise ValueError('Wrong canonical mapping')
    nodes=pd.read_parquet(ref.OLD/'data/male_v10/nodes.parquet').sort_values('node_index')
    if not np.array_equal(nodes.bodyId.to_numpy(np.int64),b.node_ids):raise ValueError('Wrong annotation order')
    prepared_clock=int(h.time_ns);before=h.state.copy();transmission=h.transmission_release()
    drive=np.asarray(obj.core.pending_proprioception['drive'],float).copy() if obj.core.proprioception_enabled else np.zeros(b.n_neurons)
    for side in ('L','R'):
        drive[obj.core.port_indices['ORN_DM1_'+side]]+=obj.core.config['odor_drive']*obj.core.pending_sensors[0 if side=='L' else 1]
    annotations={}
    with np.load(ROOT/'campanas/iniciacion_olfativa_20260926_45/odor/initial_observation.npz') as z:
        if not np.array_equal(h.release()[rows],z['DN_release']):raise ValueError('Published DN release mapping differs from trial45')
    records=[]
    for ident,row in zip(ids,rows):
        lo,hi=b.W.indptr[row:row+2];positions=np.arange(lo,hi);pre=b.W.indices[lo:hi]
        weights=cp.asnumpy(h.cuda['weights'][lo:hi]);terms=weights*transmission[pre]*h.caps[pre]
        overrides=[];writers=[]
        # List every cache whose explicit output row set contains this row.
        for key,val in vars(h).items():
            if isinstance(val,dict) and key.endswith(('_cache','_cuda')):
                for label in ('rows','target_rows','post_rows'):
                    if label in val:
                        a=val[label]
                        if isinstance(a,(np.ndarray,cp.ndarray)):
                            a=cp.asnumpy(a) if isinstance(a,cp.ndarray) else a
                            if np.any(a==row):overrides.append(key+'.'+label)
        if h.visual_mask[row]:overrides.append('visual_mask')
        for key in ('_apl_positions','_general_positions'):
            val=getattr(h,key,None)
            if val is not None:
                val=cp.asnumpy(val) if isinstance(val,cp.ndarray) else np.asarray(val)
                count=int(np.count_nonzero(np.isin(positions,val)))
                if count:writers.append(dict(owner=key,edge_count=count,enabled=bool(h.pn_online_manifest['general_outputs']['enabled']) if key=='_general_positions' else bool(h.kc_apl_dynamic_manifest['enabled'])))
        net=float(np.sum(terms));margin=net+float(drive[row])-float(h.rate_theta[row])
        # PN-general positions are inactive in45; active writers forbid an
        # effective-current claim without evaluating their proper context.
        generic_effective=not overrides and not any(w['enabled'] for w in writers)
        ranked=np.argsort(abs(terms))[::-1][:12]
        meta=nodes.iloc[row]
        records.append(dict(id=int(ident),row=int(row),type=str(meta['type']),
            state=float(h.state[row]),release=float(h.release()[row]),published=float(b.rates[row]),cap=float(h.caps[row]),
            motor_baseline=float(obj.dn_baseline[np.flatnonzero(np.asarray(obj.dn_ids)==ident)[0]]) if hasattr(obj,'dn_baseline') and hasattr(obj,'dn_ids') else None,
            tau_s=float(h.tau[row]),theta=float(h.rate_theta[row]),gain=float(h.rate_gain[row]),
            input_edges=len(pre),base_positive=float(np.sum(terms[terms>0])),base_negative=float(np.sum(terms[terms<0])),
            base_net=net,direct_drive=float(drive[row]),base_margin_over_theta=margin,
            base_target=max(0.,float(np.tanh(h.rate_gain[row]*margin))),generic_current_is_effective_at_prepared=generic_effective,
            applicable_row_caches=overrides,dynamic_writers=writers,
            largest_base_contributions=[dict(id=int(b.node_ids[pre[j]]),type=str(nodes.iloc[pre[j]]['type']),
                term=float(terms[j]),weight=float(weights[j]),transmission=float(transmission[pre[j]])) for j in ranked]))
    if not np.array_equal(before,h.state) or h.time_ns!=prepared_clock:raise ValueError('Read-only probe changed state/clock')
    result=dict(status='PREPARED_ONLY',restoration=restored,neural_steps=0,physical_steps=0,coefficient_calls=0,
        time_ns=prepared_clock,observations=records,wall_s=time.monotonic()-start,
        warning='Prepared generic CSR input is not the unrecorded input during the4s stimulus. Applicable overrides/writers preclude treating base current as effective.',
        reader_file=str(ROOT/'campanas/iniciacion_olfativa_20260926_45/protocol.py'))
    (HERE/'PREPARED_DN.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],rows=[{k:r[k] for k in ('id','type','base_margin_over_theta','generic_current_is_effective_at_prepared','applicable_row_caches','dynamic_writers')} for r in records],wall_s=result['wall_s'])),flush=True)
finally:
    if obj is not None:obj.close()
    signal.alarm(0)
