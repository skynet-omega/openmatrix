"""One cold reference-profile continuation from100ms to400ms, without preparation reset."""
from pathlib import Path
import hashlib,json,resource,shutil,signal,sys,time,traceback
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'etapa3_pn629_intervention_20260923_15'
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
sys.path[:0]=[str(OLD/'work/motor14_20260922'),str(OLD/'work/motor13_20260922'),str(ROOT/'motor_nuevo/pipeline_review_20260922')]
from motor_runtime import load
from run_storage import RunStorage,atomic_json,verify_snapshot
from restore_checkpoint import restore
from checkpoint_compare import compare,sha

def main():
    if not __debug__:raise RuntimeError('Historical loader requires normal Python; -O forbidden')
    out=Path(sys.argv[1]).resolve();source=HERE/'reference_odor_right_01';checkpoint=source/'state_100ms'
    frozen=json.loads((HERE/'CONTINUATION_SOURCES.json').read_text())
    for name,digest in frozen.items():
        if sha(name)!=digest:raise ValueError('Continuation source changed: '+name)
    contract=json.loads((HERE/'PLAN.json').read_text());limit=contract['budget']['wall_each_s_max']
    storage=RunStorage(out);start=time.perf_counter();obj=session=tap=undo=None;rows=[];error=None;cleanup=[];status='STARTING';initial=None
    previous_signal=signal.getsignal(signal.SIGTERM)
    def terminated(*_):raise InterruptedError('Bounded continuation terminated')
    signal.signal(signal.SIGTERM,terminated)
    def snapshot(folder):
        from session_io import write_state
        from operator_state import OperatorState,LEGACY_BINDINGS
        h=obj.core.hybrid
        if obj.core.failed or getattr(h,'_native_rebuild_required',False):raise RuntimeError('Failed owner cannot publish state')
        write_state(folder/'session',obj.core.state_dict());write_state(folder/'prosthesis',obj.state())
        write_state(folder/'published',{'rates':obj.core.brain.rates,'time_ns':obj.core.brain.time_ns,'rng':obj.core.brain.rng.bit_generator.state})
        write_state(folder/'effective_operator',OperatorState(h,LEGACY_BINDINGS).state_dict())
        atomic_json(folder/'boundary.json',obj.core.world.boundary.metadata())
    try:
        import cupy as cp
        import pandas as pd
        from runtime_session import RuntimeSession
        verify_snapshot(source/'prepared_state')
        # A saved experiment-frame yaw is used; the resumed posture is not a new origin.
        obj,d,plan,Field,base,old_ports=load(out/'static_preparation_inputs')
        del base,old_ports
        receipt=restore(obj,checkpoint,Field)
        from pn_cns_ports import PnCnsPorts
        ports=PnCnsPorts(obj.core.hybrid,10208);h=obj.core.hybrid
        # qpos is also recorded at the same prepared boundary in the actual trace.
        with np.load(source/'traces.npz',allow_pickle=False) as z:yaw0=d.yaw_grados(z['qpos'][39])
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            if name in sys.modules:raise RuntimeError('Ambiguous event owner '+name)
        sys.path.insert(0,str(PARENT))
        import event_waveform,event_ports,event_coupling,native_cell,device_cell
        for module in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            if Path(module.__file__).resolve().parent!=PARENT:raise RuntimeError('Wrong event implementation')
        session=RuntimeSession(h,'reference_cuda')
        from native_tap import NativeFlowTap
        metadata=pd.read_parquet(OLD/'data/male_v10/nodes.parquet',columns=['bodyId','type','rootSide','somaSide','instance'])
        tap=NativeFlowTap(h,metadata,out/'flow');undo=tap.install()
        storage.snapshot('state_100ms',snapshot)
        initial=compare(checkpoint,out/'state_100ms')
        atomic_json(out/'RESTORATION.json',dict(construction=receipt,initial=initial))
        if not initial['exact']:raise ValueError('Restored initial state differs before any step')
        start_clock=int(h.time_ns)
        boundary=obj.core.world.boundary.metadata()
        if start_clock!=boundary['installed_ns']+100_000_000 or boundary['arm']!='odor_right':raise ValueError('Wrong continuation clock/odor')
        atomic_json(out/'RUN_CONTRACT.json',{'schema':'full_organism_continuation_v1','engine':'reference_cuda','odor':'odor_right','source':str(source),
            'first_step':101,'last_step':400,'field_reinstalled':False,'baseline_reset':False,'plan_sha256':sha(HERE/'PLAN.json'),
            'checkpoint_manifest_sha256':sha(checkpoint/'MANIFEST.json'),'sources':frozen,'wall_limit_s':limit,'stage3_admission':False})
        for step in range(101,401):
            if time.perf_counter()-start>limit-40:raise TimeoutError('Frozen continuation budget')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>18:raise MemoryError('Frozen RAM budget')
            used=obj.core.pending_sensors.copy();tap.set_interval('ensayo',step);tick=time.perf_counter()
            obj.step();tap.sample_last(session.adapter);cp.cuda.runtime.deviceSynchronize()
            row=d.captura(obj,ports,'ensayo',step,used,yaw0,cp);rows.append(row)
            record={'step':step,'clock_ns':int(h.time_ns),'step_wall_s':time.perf_counter()-tick,'elapsed_s':time.perf_counter()-start,'yaw_delta_deg':row['yaw_delta_deg']}
            with (out/'PROGRESS.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if step%10==0:print(json.dumps(record),flush=True)
        status='COMPLETE'
    except BaseException as exc:
        status='INCOMPLETE';error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    finally:
        try:
            if rows:np.savez_compressed(out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
            if session and session.events:atomic_json(out/'EVENT_AUDIT.json',{'profile':'reference_cuda','odor':'odor_right','blocks':session.events.audit})
            if tap:tap.save();tap.close()
        except BaseException as exc:cleanup.append('trace/observer: '+repr(exc))
        if obj is not None and obj.core is not None:
            try:storage.snapshot('final_state',snapshot)
            except BaseException as exc:cleanup.append('final state: '+repr(exc))
        if undo:
            try:undo()
            except BaseException as exc:cleanup.append('observer undo: '+repr(exc))
        runtime=None if session is None else session.report()
        for close in ([session.close] if session else [])+([obj.close] if obj is not None and obj.core is not None else []):
            try:close()
            except BaseException as exc:cleanup.append('close: '+repr(exc))
        signal.signal(signal.SIGTERM,previous_signal)
        if cleanup:status='INCOMPLETE'
        folder=out/'executed_sources';folder.mkdir()
        for i,name in enumerate(frozen):shutil.copyfile(name,folder/(str(i)+'__'+Path(name).name))
        atomic_json(out/'RESULT.json',{'status':status,'error':error,'cleanup_errors':cleanup,'completed_continuation_ms':len(rows),
            'initial_exact':None if initial is None else initial['exact'],'runtime':runtime,'wall_total_s':time.perf_counter()-start,'stage3_admission':False})
    return 0 if status=='COMPLETE' else 2
if __name__=='__main__':raise SystemExit(main())
