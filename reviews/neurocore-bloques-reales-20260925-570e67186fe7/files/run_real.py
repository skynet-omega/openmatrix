"""Run the actual coupled organism with a process-local neural executor.

No weights, physiological constants, event owners or geometry are normalized.
This is an integration comparison; the body is included in total timings but
no navigation/flight or biological equivalence claim follows from it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import signal
import sys
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
EPOCH=ROOT/'motor_nuevo/epoch_cost_20260923'
sys.path[:0]=[str(OLD/'work/motor14_20260922'),str(OLD/'work/motor13_20260922'),
             str(ROOT/'motor_nuevo/pipeline_review_20260922'),str(EPOCH)]


def sha_bytes(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('reference','candidate'),required=True)
    p.add_argument('--ms',type=int,choices=(1,100),required=True)
    p.add_argument('--field',choices=('sham','odor_left'),default='sham')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wall-limit',type=int,default=600)
    p.add_argument('--precision-factor',type=float,choices=(1.,),default=1.)
    p.add_argument('--compact-state',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();obj=session=undo=None
    result={'mode':a.mode,'requested_ms':a.ms,'completed_ms':0,'field':a.field,
            'status':'STARTED','parameters_changed':a.precision_factor!=1.,
            'biological_parameters_changed':False,'numerical_precision_factor':a.precision_factor,
            'compact_state':a.compact_state,'whole_neural_rewrite':False,
            'scope':'Real coupled organism; candidate replaces CNS and spatial executors. Physical equations and PN solver conserved.'}
    result['sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
         [HERE/name for name in ('run_real.py','real_model.py','graph_runtime.py','event_projection.py',
          'resident_controller.cu','libresident_controller.so','PLAN.md','device_cell.py',
          'block_executor.py','dense_warp.cuh','independent_blocks.cuh','membrane_model.cu')]}
    records=[];arrays={};sampled=[]
    def stop(signum,frame):raise TimeoutError('process evaluation wall budget')
    previous=signal.signal(signal.SIGALRM,stop);signal.alarm(a.wall_limit)
    def collect():
        b=obj.core.hybrid._spatial_batch
        while hasattr(b,'base'):b=b.base
        ax=obj.core.hybrid._axonal_release.state
        import cupy as cp
        used_gpu=int(cp.cuda.runtime.memGetInfo()[1]-cp.cuda.runtime.memGetInfo()[0])
        result['peak_sampled_gpu_used_bytes']=max(used_gpu,result.get('peak_sampled_gpu_used_bytes',0))
        if used_gpu-result['initial_gpu_used_bytes']>8*1024**3:
            raise MemoryError('8-GiB sampled VRAM increment budget')
        row={'time_ns':np.asarray(obj.core.hybrid.time_ns,dtype=np.int64),
             'cns':obj.core.hybrid.state.copy(),'body_qpos':obj.body.data.qpos.copy(),
             'body_qvel':obj.body.data.qvel.copy(),
             'pending_sensors':obj.core.pending_sensors.copy()}
        for name in ('delta','gates','q','counts','clipped','last_siz','previous_slope','trough'):
            row['cell_'+name]=cp.asnumpy(cp.asarray(getattr(b,name)))
        for name,value in ax.items():
            if isinstance(value,np.ndarray):row['axon_'+name]=value.copy()
        for name,value in row.items():
            if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError('nonfinite '+name)
        sampled.append(row)

    try:
        if not __debug__:raise RuntimeError('Historical preparation requires normal Python')
        import cupy as cp
        from motor_runtime import load
        from runtime_session import RuntimeSession
        from session_io import write_state
        obj,d,plan,Field,field,ports=load(a.out/'preparation_inputs')
        h=obj.core.hybrid;initial_ns=int(h.time_ns)
        # Numerical refinement of the CNS integrator, identical in both arms.
        # Physiological parameters and other owner solvers remain conserved.
        h.parameters['rtol']*=a.precision_factor
        h.parameters['atol']*=a.precision_factor
        if plan['checkpoint']!='work/stage234_settling_extension_20260915/settled_700ms':
            raise ValueError('unexpected preparation')
        result['checkpoint']=plan['checkpoint']
        result['parameters']=dict(h.parameters)
        result['initial_cns_sha256']=sha_bytes(h.state)
        result['weights_sha256']=sha_bytes(cp.asnumpy(h.cuda['weights']))
        result['weights_count']=int(h.cuda['weights'].size)
        result['device']=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        result['initial_gpu_used_bytes']=int(cp.cuda.runtime.memGetInfo()[1]-cp.cuda.runtime.memGetInfo()[0])
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            if name in sys.modules:raise RuntimeError('unexpected imported owner '+name)
        sys.path.insert(0,str(EPOCH))
        import event_waveform,event_ports,event_coupling,native_cell,device_cell
        for module in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            if Path(module.__file__).resolve().parent!=EPOCH:raise RuntimeError('wrong event owner')
        if a.mode=='candidate':
            sys.path.insert(0,str(HERE))
            from real_model import install
            undo=install()
        session=RuntimeSession(h,'causal_cuda')
        d.instalar_campo(obj,Field,field,a.field,0.)
        result['load_setup_s']=time.perf_counter()-start
        collect()
        for k in range(a.ms):
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>18*1024**2:
                raise MemoryError('18-GiB RSS process budget')
            if time.perf_counter()-start>a.wall_limit-(5 if a.compact_state else 25):
                raise TimeoutError('saving prefix before wall limit')
            used=obj.core.pending_sensors.copy();t=time.perf_counter()
            obj.step();cp.cuda.runtime.deviceSynchronize()
            elapsed=time.perf_counter()-t
            if int(h.time_ns)!=initial_ns+(k+1)*1000000:raise ValueError('clock mismatch')
            row={'step':k+1,'time_ns':int(h.time_ns),'advance_s':elapsed,
                 'elapsed_s':time.perf_counter()-start,'sensors_sha256':sha_bytes(used),
                 'accepted':int(h.statistics['accepted']),'rejected':int(h.statistics['rejected'])}
            records.append(row);result['completed_ms']=k+1
            if a.ms<=100 or (k+1)%10==0:collect()
            with (a.out/'PROGRESS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if k==0 or (k+1)%5==0:print(json.dumps(row),flush=True)
        result['status']='COMPLETE'
        final=a.out/'final_state';final.mkdir()
        if a.compact_state:
            write_state(final/'pn_state',obj.core.state_dict()['hybrid']['pn_online_state'])
        else:write_state(final/'session',obj.core.state_dict())
        write_state(final/'prosthesis',obj.state())
        write_state(final/'published',{'rates':obj.core.brain.rates,
                    'time_ns':obj.core.brain.time_ns,'rng':obj.core.brain.rng.bit_generator.state})
    except BaseException as exc:
        result.update(status='INCOMPLETE',error={'type':type(exc).__name__,'message':str(exc),
                      'traceback':traceback.format_exc(),'diagnostic':getattr(exc,'numerical_diagnostic',None)})
        print(json.dumps({'status':'INCOMPLETE','error':str(exc)}),flush=True)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if sampled:
            arrays={name:np.stack([r[name] for r in sampled]) for name in sampled[0]}
            np.savez_compressed(a.out/'trajectory.npz',**arrays)
        result['advance_total_s']=sum(r['advance_s'] for r in records)
        if session is not None:
            result['runtime']=session.report()
            save(a.out/'EVENTS.json',session.events.audit)
            if a.mode=='candidate' and session.adapter.core:
                core=session.adapter.core
                result['cns_resident_s']=core.resident_wall_s
                result['rhs_evaluations']=5*(core.steps+core.rejected)
                result['cns_build_s']=core.build_s
            try:session.close()
            except Exception as exc:result.setdefault('cleanup_errors',[]).append(repr(exc))
        if undo:undo()
        if obj:
            try:obj.close()
            except Exception as exc:result.setdefault('cleanup_errors',[]).append(repr(exc))
        result['wall_total_s']=time.perf_counter()-start
        result['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['files_bytes']=sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file())
        save(a.out/'RESULT.json',result)
        print(json.dumps({k:result[k] for k in ('mode','status','completed_ms','advance_total_s','wall_total_s')}),flush=True)
    return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
