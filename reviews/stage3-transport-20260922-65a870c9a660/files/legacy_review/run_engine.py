"""Complete neural/body trials with small current-state exports, no history copy."""
from pathlib import Path
import argparse,json,resource,sys,time,traceback,hashlib,shutil
import numpy as np
if not __debug__:raise RuntimeError('Whole-organism trials require normal Python: legacy loaders contain essential asserts')
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from motor_runtime import load,dump

def run(out,ms,graph=False,coupling=None,validate_graph=False,odor='sham',symmetric=None,waveform=None,fine=False,kc_midpoint=False,block_midpoint=None,kc_adaptive=False):
    import cupy as cp
    from threadpoolctl import threadpool_limits
    from session_io import write_state
    from kcgamma_regional_brain import _record_hash
    start=time.perf_counter();obj=None;restore=None;engine=None;rows=[];times=[];restore_split=None;split_reports=None;restore_kc=None
    try:
        obj,d,plan,Field,field,ports=load(out);out=Path(out);h=obj.core.hybrid
        frozen=out/'executed_sources';frozen.mkdir()
        for file in HERE.glob('*.py'):shutil.copy2(file,frozen/file.name)
        dump(out/'EXECUTED_SOURCES.json',{file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in frozen.glob('*.py')})
        if kc_midpoint or block_midpoint or kc_adaptive:
            if kc_adaptive:from kc_adaptive import install as install_kc
            else:from kc_midpoint import install as install_kc
            restore_kc=install_kc(h)
            dump(out/'KC_NUMERICAL_POLICY.json',{'scheme':'Exponential midpoint gates / CN voltage; unchanged continuous equations, new numerical trajectory','adaptive':kc_adaptive})
        if coupling is not None:
            if coupling not in (7812,15625,31250,62500,125000):raise ValueError('Unregistered coupling')
            previous=h.pn_online_manifest['coupling_ns']
            h.pn_online_manifest['coupling_ns']=coupling
            h.pn_online_manifest['record_sha256']=_record_hash(h.pn_online_manifest)
            dump(out/'NUMERICAL_CHANGE.json',{'old_coupling_ns':previous,'new_coupling_ns':coupling,
                 'scope':'Experimental partitioned integration step. Changes numerical trajectory, never claims byte identity or stage3.'})
        if graph:
            from graph_midpoint import install
            engine,restore=install(h,validate_graph)
        if symmetric:
            from symmetric_coupling import install as install_split
            split_reports,restore_split=install_split(h,symmetric)
            dump(out/'SYMMETRIC_POLICY.json',{'step_ns':symmetric,'composition':'PN(h/2, held initial CNS); CNS(h, held midpoint PN); PN(h/2, held final CNS)',
                 'second_order_whole_system':'Not presumed; unchanged internal coupling may limit order.'})
        if waveform:
            from waveform_coupling import install as install_waveform
            split_reports,restore_split=install_waveform(h,waveform)
            dump(out/'WAVEFORM_POLICY.json',{'step_ns':waveform,'composition':'Rollback CNS predictor; continuous initial-mid-final receptor drivers; explicit midpoint PN output for CNS',
                 'second_order_whole_system':'Not presumed; convergence required.'})
        if fine:
            from fine_schedule import install as install_fine
            restore_split=install_fine(h)
            dump(out/'FINE_POLICY.json',{'alternating_steps_ns':[7812,7813],'sum_equals_original_ns':15625})
        if block_midpoint:
            from block_midpoint import install as install_block
            split_reports,restore_split=install_block(h,block_midpoint)
            dump(out/'BLOCK_MIDPOINT_POLICY.json',{'macro_ns':block_midpoint,'KC_scheme':'Exponential midpoint / CN',
                 'interface':'Predicted midpoint KC/APL conductances and outputs; PN continuous initial-mid-final drivers',
                 'status':'Experimental; independent and full-state convergence required'})
        if odor!='sham':d.instalar_campo(obj,Field,field,odor,0.)
        yaw0=d.yaw_grados(obj.body.data.qpos);before=dict(h.statistics);initial=h.time_ns
        with threadpool_limits(limits=1,user_api='blas'):
            for k in range(ms):
                used=obj.core.pending_sensors.copy();cp.cuda.runtime.deviceSynchronize()
                t=time.perf_counter();cpu=time.process_time();obj.step();cp.cuda.runtime.deviceSynchronize()
                timing={'step':k+1,'clock_ns':int(h.time_ns),'wall_s':time.perf_counter()-t,'cpu_s':time.process_time()-cpu}
                times.append(timing)
                rows.append(d.captura(obj,ports,'ensayo',k+1,used,yaw0,cp))
                with (out/'TIMES.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(timing)+'\n')
                if k==0 or (k+1)%10==0:print(json.dumps(timing),flush=True)
            if h.time_ns!=initial+ms*1_000_000:raise ValueError('Simulation clock mismatch')
            np.savez_compressed(out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
            # Full declared brain state; inherited observation history remains in
            # the unchanged named checkpoint and is not operational neural state.
            write_state(out/'brain_final',h.state_dict())
            np.savez_compressed(out/'body_final.npz',qpos=obj.body.data.qpos,qvel=obj.body.data.qvel,
                qacc=obj.body.data.qacc,qacc_warmstart=obj.body.data.qacc_warmstart,act=obj.body.data.act,
                ctrl=obj.body.data.ctrl,pending_sensors=obj.core.pending_sensors,
                pending_excitation=obj.core.pending_excitation)
            dump(out/'PROVENANCE.json',{'checkpoint':plan['checkpoint'],'checkpoint_role':'Unmodified inherited history/anatomy; this result is not a full standalone reload checkpoint',
                 'graph':graph,'coupling_ns':h.pn_online_manifest['coupling_ns'],'odor':odor,'initial_clock_ns':initial})
        data={'ms':ms,'graph':graph,'wall_total_s':time.perf_counter()-start,'step_wall_s':sum(x['wall_s'] for x in times),
              'steady_step_wall_s':sum(x['wall_s'] for x in times[1:]),'steady_ms':max(0,ms-1),
              'cpu_process_s':time.process_time(),'rss_max_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
              'statistics_before':before,'statistics_after':h.statistics,
              'graph_trials':engine.comparisons if engine else [],
              'graph_build_s':sum(g.build_s for g in engine.graphs.values()) if engine else 0.,
              'graph_device_bytes':sum(g.pool.total_bytes() for g in engine.graphs.values()) if engine else 0,
              'gpu_pool_bytes':cp.get_default_memory_pool().total_bytes(),'coupling_ns':h.pn_online_manifest['coupling_ns'],
              'symmetric_policy':split_reports}
        data['KC_adaptive_work']=getattr(h._spatial_batch,'_motor_adaptive_stats',None)
        dump(out/'RESULT.json',data);print(json.dumps(data),flush=True)
    except BaseException as exc:
        Path(out).mkdir(parents=True,exist_ok=True)
        dump(Path(out)/'ERROR.json',{'error':str(exc),'traceback':traceback.format_exc(),'completed_ms':len(times)})
        raise
    finally:
        if restore_split:restore_split()
        if restore_kc:restore_kc()
        if restore:restore()
        if obj:obj.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--ms',type=int,required=True)
    p.add_argument('--graph',action='store_true');p.add_argument('--validate-graph',action='store_true');p.add_argument('--coupling',type=int)
    p.add_argument('--odor',default='sham',choices=['sham','odor_left','odor_right','odor_uniform'])
    p.add_argument('--symmetric',type=int)
    p.add_argument('--waveform',type=int);p.add_argument('--fine',action='store_true')
    p.add_argument('--kc-midpoint',action='store_true');p.add_argument('--block-midpoint',type=int)
    p.add_argument('--kc-adaptive',action='store_true')
    a=p.parse_args()
    if sum(bool(x) for x in (a.symmetric,a.waveform,a.fine,a.block_midpoint))>1:p.error('One integration policy required')
    run(a.out,a.ms,a.graph,a.coupling,a.validate_graph,a.odor,a.symmetric,a.waveform,a.fine,a.kc_midpoint,a.block_midpoint,a.kc_adaptive)
