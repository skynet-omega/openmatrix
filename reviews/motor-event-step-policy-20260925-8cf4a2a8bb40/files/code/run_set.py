"""Bounded organism test of SET/ADD at every physical somatic event producer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
sys.path[:0] = [str(OLD/'work/motor14_20260922'), str(OLD/'work/motor13_20260922'),
                str(ROOT/'motor_nuevo/pipeline_review_20260922')]
from motor_runtime import load
from run_pipeline import RunStorage, atomic_json
from native_tap import NativeFlowTap


def capture_kc(h, dest: Path) -> None:
    """Post-commit readback only; never fed into the next model step."""
    import cupy as cp
    b=h._spatial_batch
    while hasattr(b,'base'):
        b=b.base
    # The per-epoch publisher callback is deliberately removed on return from
    # advance(). The committed axonal owner is the brain's release ledger.
    ax=h._axonal_release.state
    arrays={'time_ns':np.asarray(h.time_ns,dtype=np.int64)}
    for name in ('delta','gates','q','last_siz','previous_slope','trough','counts','clipped'):
        arrays['somatic_'+name]=cp.asnumpy(cp.asarray(getattr(b,name)))
    for name in ('q','s','last_voltage','previous_slope','trough','counts','clipped'):
        arrays['axonal_'+name]=np.asarray(ax[name]).copy()
    if not all(np.isfinite(x).all() for x in arrays.values() if x.dtype.kind in 'fc'):
        raise ValueError('Nonfinite KC diagnostic state')
    dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,**arrays)


def main() -> int:
    if not __debug__:
        raise RuntimeError('Historical loader requires normal Python; -O forbidden')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--odor', choices=('sham','uniform','odor_left','odor_right'), required=True)
    p.add_argument('--engine', choices=('causal_cuda','reference_cuda'), default='causal_cuda')
    p.add_argument('--ms', type=int, choices=(1,2,20,130,400), required=True)
    p.add_argument('--observe', choices=('on','off'), required=True)
    p.add_argument('--cuda-profile',choices=('on','off'),default='off')
    p.add_argument('--profile', choices=('on','off'), required=True)
    p.add_argument('--kc-capture', choices=('on','off'), required=True)
    a = p.parse_args()
    if a.ms in (1,2) and a.odor != 'sham':
        raise ValueError('One/two-ms boundary gate must be sham')
    if a.ms == 20 and a.odor != 'sham':
        raise ValueError('Twenty-ms numerical reference must be sham')
    if a.ms in (130,400) and a.observe != 'on' and a.odor != 'sham':
        raise ValueError('Only sham may run as a full observer-off control')
    prep = 0 if a.ms in (1,2,20) else 40
    max_wall = 240 if a.ms in (1,2) else (600 if a.ms == 20 else (700 if a.ms == 130 else 2050))
    storage = RunStorage(a.out)
    start = time.perf_counter()
    obj = session = observer = undo = None
    rows = []
    import cProfile
    profiler = cProfile.Profile()
    status = 'STARTING'
    error = None
    cleanup = []
    phase = 'loading'
    checkpoint = {'status':'NOT_ATTEMPTED'}
    old_sigterm = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise InterruptedError('Bounded process terminated')
    signal.signal(signal.SIGTERM, terminated)

    def snapshot(folder: Path) -> None:
        from session_io import write_state
        h = obj.core.hybrid
        if getattr(obj.core,'failed',False) or getattr(h,'_native_rebuild_required',False):
            raise RuntimeError('Failed owner cannot publish coherent state')
        write_state(folder/'session', obj.core.state_dict())
        write_state(folder/'prosthesis', obj.state())
        write_state(folder/'published', {'rates':obj.core.brain.rates,
                    'time_ns':obj.core.brain.time_ns,'rng':obj.core.brain.rng.bit_generator.state})
        atomic_json(folder/'boundary.json', obj.core.world.boundary.metadata())

    def save_trace() -> None:
        if rows:
            np.savez_compressed(a.out/'traces.npz',
                **{key:np.asarray([r[key] for r in rows]) for key in rows[0]})

    try:
        import cupy as cp
        import pandas as pd
        from runtime_session import RuntimeSession
        obj, d, plan, Field, field, ports = load(a.out/'preparation_inputs')
        h = obj.core.hybrid
        # RuntimeSession establishes the model source path but imports event
        # modules only during instantiation. Refuse an ambiguous preloaded
        # owner rather than silently mixing historical and local operators.
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            if name in sys.modules:
                raise RuntimeError('Event owner already imported before local candidate: '+name)
        sys.path.insert(0,str(HERE))
        import event_waveform,event_ports,event_coupling,native_cell,device_cell
        for module in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            if Path(module.__file__).resolve().parent != HERE:
                raise RuntimeError('Wrong SET/ADD module source: '+module.__name__)
        session = RuntimeSession(h,a.engine)
        if a.observe == 'on':
            metadata = pd.read_parquet(OLD/'data/male_v10/nodes.parquet',
                        columns=['bodyId','type','rootSide','somaSide','instance'])
            observer = NativeFlowTap(h,metadata,a.out/'flow')
            undo = observer.install()
        frozen = a.out/'executed_sources'
        frozen.mkdir()
        files = [HERE/'PLAN.json',HERE/'run_set.py',HERE/'native_tap.py',
                 HERE/'event_waveform.py',HERE/'event_ports.py',HERE/'event_coupling.py',
                 HERE/'native_cell.py',HERE/'device_cell.py',HERE/'physical_events.cu',
                 HERE/'device_cell.cu',HERE/'block_runtime.hpp',
                 ROOT/'motor_nuevo/pipeline_review_20260922/runtime_session.py',
                 ROOT/'campanas/etapa3_motor_nuevo_20260922/organism_adapter.py',
                 ROOT/'motor_nuevo/native_hybrid_20260922/legacy_sources/gpu_visual_brain.py',
                 ROOT/'motor_nuevo/native_hybrid_20260922/legacy_sources/prosthetic_olfactory_brain.py',
                 ROOT/'motor_nuevo/native_hybrid_20260922/legacy_sources/gpu_coefficient_layout.py',
                 OLD/'work/motor13_20260922/block_midpoint.py',
                 OLD/'work/motor14_20260922/event_coupling.py']
        manifest = {}
        for source in files:
            destination = frozen/(source.parent.name+'__'+source.name)
            shutil.copy2(source,destination)
            manifest[destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
        atomic_json(a.out/'FROZEN.json',manifest)
        atomic_json(a.out/'RUN_CONTRACT.json',{
            'schema':'stage3_flow_run_contract_v1','odor':a.odor,'engine':a.engine,
            'observer':a.observe,'preparation_ms':prep,'trial_ms':a.ms,
            'checkpoint':plan['checkpoint'],
            'checkpoint_manifest_sha256':hashlib.sha256((OLD/plan['checkpoint']/'manifest.json').read_bytes()).hexdigest(),
            'plan_sha256':hashlib.sha256((HERE/'PLAN.json').read_bytes()).hexdigest(),
            'wall_limit_s':max_wall,'event_boundaries':True,
            'biological_parameters_changed':False,'stage3_admission':False,
            'event_representation':'physical SET(post_q) from both LIF and gamma CUDA publishers; genuine ADD retained; no state clipping',
            'native_sideband':'FP64 raw signed net before tanh in general and PN CUDA kernels' if a.observe=='on' else 'off',
            'kc_post_commit_capture':a.kc_capture})
        yaw0 = d.yaw_grados(obj.body.data.qpos)
        for phase, duration in [('preparacion',prep),('ensayo',a.ms)]:
            if phase == 'ensayo':
                if prep:
                    storage.snapshot('prepared_state',snapshot)
                d.instalar_campo(obj,Field,field,a.odor,0.)
                yaw0 = d.yaw_grados(obj.body.data.qpos)
            for k in range(duration):
                if time.perf_counter()-start > max_wall-40:
                    raise TimeoutError('Frozen process budget; saving prefix evidence')
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 > 18:
                    raise MemoryError('Frozen 18-GiB RAM budget')
                used = obj.core.pending_sensors.copy()
                if observer:
                    observer.set_interval(phase,k+1)
                t = time.perf_counter()
                if a.cuda_profile == 'on' and k == 1: cp.cuda.profiler.start()
                if a.profile == 'on' and k > 0: profiler.enable()
                try: obj.step()
                finally: profiler.disable()
                if a.cuda_profile == 'on' and k == 1: cp.cuda.profiler.stop()
                if a.kc_capture == 'on':
                    capture_kc(h,a.out/'kc'/f'after_{k+1:03d}ms.npz')
                if observer:
                    observer.sample_last(session.adapter)
                cp.cuda.runtime.deviceSynchronize()
                row = d.captura(obj,ports,phase,k+1,used,yaw0,cp)
                rows.append(row)
                record = {'phase':phase,'step':k+1,'clock_ns':int(h.time_ns),
                          'step_wall_s':time.perf_counter()-t,
                          'elapsed_s':time.perf_counter()-start,'yaw_delta_deg':row['yaw_delta_deg']}
                with (a.out/'PROGRESS.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(record,allow_nan=False)+'\n')
                if phase == 'preparacion' and np.any(used != 0):
                    raise ValueError('Preparation consumed odor')
                if phase == 'ensayo' and k+1 == 100:
                    storage.snapshot('state_100ms',snapshot)
                if (k+1)%10 == 0:
                    print(json.dumps(record),flush=True)
        if a.profile == 'on':
            import pstats
            stats=pstats.Stats(profiler)
            top=sorted(stats.stats.items(),key=lambda x:x[1][3],reverse=True)
            atomic_json(a.out/'PROFILE.json', {'rows':[{'file':key[0],'line':key[1],'function':key[2],'primitive_calls':value[0],'calls':value[1],'self_s':value[2],'cumulative_s':value[3]} for key,value in top], 'scope':'obj.step after first warmup; nested cumulative times must not be summed', 'total_self_s':stats.total_tt})
        status = 'COMPLETE'
    except BaseException as exc:
        error = {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                 'numerical_diagnostic':getattr(exc,'numerical_diagnostic',None)}
        status = 'INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'INCOMPLETE'
    finally:
        try:
            save_trace()
        except BaseException as exc:
            cleanup.append('trace: '+repr(exc))
            status = 'INCOMPLETE'
        if session is not None and session.events is not None:
            try:
                atomic_json(a.out/'EVENT_AUDIT.json',{'schema':'stage3_event_owner_log_v1',
                              'profile':a.engine,'odor':a.odor,'blocks':session.events.audit})
            except BaseException as exc:
                cleanup.append('event audit: '+repr(exc))
                status = 'INCOMPLETE'
        if observer:
            try:
                observer.save()
                observer.close()
            except BaseException as exc:
                cleanup.append('observer: '+repr(exc))
                status = 'INCOMPLETE'
        if obj is not None:
            try:
                checkpoint = storage.snapshot('final_state',snapshot)
            except BaseException as exc:
                checkpoint = {'status':'FAILED_NOT_RESUMABLE','message':repr(exc)}
                cleanup.append('checkpoint: '+repr(exc))
                status = 'INCOMPLETE'
        if undo:
            try:undo()
            except BaseException as exc:cleanup.append('observer undo: '+repr(exc))
        runtime = None if session is None else session.report()
        for closer in ([session.close] if session else [])+([obj.close] if obj else []):
            try:closer()
            except BaseException as exc:cleanup.append('close: '+repr(exc))
        signal.signal(signal.SIGTERM,old_sigterm)
        trial = [r for r in rows if r['fase']=='ensayo']
        atomic_json(a.out/'RESULT.json',{'status':status,'error':error,'cleanup_errors':cleanup,
            'odor':a.odor,'engine':a.engine,'observer':a.observe,
            'requested_trial_ms':a.ms,'completed_trial_ms':len(trial),
            'completed_preparation_ms':len(rows)-len(trial),'wall_total_s':time.perf_counter()-start,
            'checkpoint':checkpoint,'runtime':runtime,'stage3_pass':None,
            'reason':'Flow diagnostic only; not a new biological or historical admission.'})
    return 0 if status == 'COMPLETE' and not cleanup else 2


if __name__ == '__main__':
    raise SystemExit(main())
