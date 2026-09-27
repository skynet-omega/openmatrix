"""Extract only the six preregistered rows. Never import a neural runtime."""
import hashlib, json, resource, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAM=ROOT/'campanas/etapa45_composicion_20260927_48'
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
STATIC=OLD/'work/stage234_settling_extension_20260915/settled_700ms/core_carrier/brain'
def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def ah(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()
def save(p,o): p.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    started=time.perf_counter(); cpu=time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    need(not (HERE/'endpoint_inputs.npz').exists(),'Preserve previous extraction')
    plan=json.loads((HERE/'PLAN.json').read_text())
    lock=json.loads((CAM/'SOURCES.json').read_text()); sources={}
    def verify(p,expected=None):
        p=Path(p); digest=sha(p); expected=expected or lock.get(str(p))
        need(expected is not None and digest==expected,'Changed or unpinned source '+str(p))
        sources[str(p)]={'sha256':digest,'bytes':p.stat().st_size}
    for p in [STATIC/'state.npz',STATIC/'weights_post_pre.npz',OLD/'data/male_v10/nodes.parquet',OLD/'config/matrix_diagnostico_olfativo_v1.json']:
        verify(p)
    with np.load(STATIC/'state.npz',allow_pickle=False) as z:
        ids=z['node_ids']; caps=z['r_max'].astype(np.float64); nt=z['nt_labels']
    nodes=pd.read_parquet(OLD/'data/male_v10/nodes.parquet').sort_values('node_index')
    need(np.array_equal(ids,nodes.bodyId.to_numpy()),'Node order mismatch')
    types=nodes.type.fillna('').to_numpy(str)
    target_ids=np.array([i for t in plan['targets'] for i in t['ids']],np.int64)
    rows=np.searchsorted(ids,target_ids)
    need(np.array_equal(ids[rows],target_ids),'Target IDs')
    need(types[rows].tolist()==[t['type'] for t in plan['targets'] for i in t['ids']],'Target types')
    with np.load(STATIC/'weights_post_pre.npz',allow_pickle=False) as z:
        ptr=z['indptr']; idx=z['indices']
    positions=np.concatenate([np.arange(ptr[r],ptr[r+1],dtype=np.int64) for r in rows])
    pre=idx[positions]; localptr=np.r_[0,np.cumsum([ptr[r+1]-ptr[r] for r in rows])]
    need(len(caps)==len(ids)==166700,'Graph size')
    out=dict(target_ids=target_ids,target_rows=rows,ptr=localptr,positions=positions,pre_rows=pre,
        pre_ids=ids[pre],pre_types=types[pre],pre_superclass=nodes.superclass.fillna('').to_numpy(str)[pre],
        pre_nt=nt[pre],caps=caps[pre],target_types=types[rows],target_sides=nodes.somaSide.fillna('').to_numpy(str)[rows])
    metadata={}; traces={}
    for arm in plan['arms']:
        folder=CAM/arm/'final_state'
        manifest=json.loads((folder/'MANIFEST.json').read_text())
        for name in ['session.json','session.npz','effective_operator.json','effective_operator.npz','published.json','published.npz']:
            verify(folder/name,manifest['files'][name])
        sources[str(folder/'MANIFEST.json')]={'sha256':sha(folder/'MANIFEST.json'),'bytes':(folder/'MANIFEST.json').stat().st_size}
        desc=json.loads((folder/'session.json').read_bytes()); h=desc['hybrid']
        meta=dict(time_ns=h['time_ns'],visual_output_connected=h['visual_output_connected'],restart_tested=manifest['restart_tested'])
        with np.load(folder/'session.npz',allow_pickle=False) as z:
            def a(x): return z[x['__array__']]
            state=a(h['state']); n=len(ids); start=n+2*len(a(h['photo_ids']))
            vm=np.isin(ids,a(h['visual_ids']))
            need(not vm[rows].any(),'Selected destinations are not rate cells')
            out[arm+'_transmission']=state[start+pre]
            release=state[:n].copy();release[vm]=np.clip((80*release[vm]-15)/40,0,1)
            out[arm+'_release']=release[pre]
            out[arm+'_target_q']=state[rows]
            out[arm+'_visual_pre']=vm[pre]
            meta['state_layout']={'size':len(state),'transmission_start':start,'n':n}
            meta['specialized_target_membership']={}
            for name,m in h.items():
                if not name.endswith('_manifest') or not isinstance(m,dict):continue
                hits={}
                for key in ['target_rows','rows','source_rows']:
                    if isinstance(m.get(key),dict) and '__array__' in m[key]:
                        hits[key]=target_ids[np.isin(rows,a(m[key]))].tolist()
                if hits: meta['specialized_target_membership'][name]={'enabled':m.get('enabled'),**hits}
            rm=h['receptor_manifest']
            need(ah(ptr)==rm['anatomical_indptr_sha256'] and ah(idx)==rm['anatomical_indices_sha256'],'CSR identity')
            need(ah(ids)==rm['node_ids_sha256'],'ID identity')
            meta['csr_verified_against_checkpoint']=True
            meta['general_PN_enabled']=h['pn_online_manifest']['general_outputs']['enabled']
            out[arm+'_APL_presynaptic']=np.char.startswith(types[pre],'APL')
            meta['APL_edges_per_target']=[int(out[arm+'_APL_presynaptic'][localptr[k]:localptr[k+1]].sum()) for k in range(len(rows))]
            # The whole publication is only a witness to static caps and release,
            # never the presynaptic filtered input used for row contributions.
            with np.load(folder/'published.npz',allow_pickle=False) as pub:
                d=json.loads((folder/'published.json').read_text())
                published=pub[d['rates']['__array__']]
            meta['publication_equal_release_times_caps']=bool(np.array_equal(published,(release*caps).astype(np.float32)))
            need(meta['publication_equal_release_times_caps'],'Publication/caps provenance mismatch')
        operator=json.loads((folder/'effective_operator.json').read_text())
        values=operator['values']
        with np.load(folder/'effective_operator.npz',allow_pickle=False) as z:
            for key in ['weights','cuda_weights']:
                full=z[values[key]['__array__']]
                out[arm+'_'+key]=full[positions]
                del full
            for key in ['tau','theta','gain','cuda_tau','cuda_theta','cuda_gain']:
                out[arm+'_'+key]=z[values[key]['__array__']][rows]
        observed=CAM/arm/'OBSERVED_IDS.npz'
        sources[str(observed)]={'sha256':sha(observed),'bytes':observed.stat().st_size,
            'authority':'captured at extraction; observational identity file, not independently frozen by this diagnostic'}
        with np.load(observed,allow_pickle=False) as z:
            out[arm+'_prescribed_ORN']=np.isin(ids[pre],z['prescribed_ORN_ids'])
        # Preserve a real last committed RHS witness separately: its hidden
        # midpoint/event inputs are not assumed equal to final checkpoint.
        block=CAM/arm/'blocks/02901_03000ms';obs=block/'dng100_observed.npz'
        bm=json.loads((block/'MANIFEST.json').read_text());verify(obs,bm['hashes'][obs.name])
        with np.load(obs,allow_pickle=False) as z:
            r=z['records'];ep=np.repeat(np.arange(len(z['trials'])),z['trials'])
            take=z['committed'][ep] & (r[:,138]==1)
            last=r[np.flatnonzero(take)[-1]]
            out[arm+'_last_RHS']=last[:128].reshape(4,2,16)[-1]
            traces[arm]={'fields':z['fields'].tolist(),'last_committed_epoch_start_ns':int(z['start_ns'][ep[np.flatnonzero(take)[-1]]]),'trial_end_s':float(last[130]),'source':str(obs)}
        metadata[arm]=meta
        del desc,h,state,release
    for k,v in out.items():
        if v.dtype.kind in 'fc':need(np.isfinite(v).all(),'Nonfinite '+k)
    np.savez_compressed(HERE/'endpoint_inputs.npz',**out)
    save(HERE/'EXTRACTION.json',dict(scope='Endpoint direct-row operands only; not historical RHS replay',
        metadata=metadata,last_RHS_witness=traces,sources=sources,plan_sha256=sha(HERE/'PLAN.json'),
        inputs_sha256=sha(HERE/'endpoint_inputs.npz'),cpu_s=time.process_time()-cpu,
        wall_s=time.perf_counter()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        new_neural_steps=0,GPU_runs=0))
    print(json.dumps({'edges':len(pre),'per_target':np.diff(localptr).tolist(),'metadata':metadata,'cpu_s':time.process_time()-cpu}))
if __name__=='__main__':main()
