"""Two independently restored, bounded whole-organism arms; frozen motor13 core."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'motor_nuevo/full_pipeline_review_20260925_12'
LONG=ROOT/'motor_nuevo/dynamics_12s_20260926_13'
sys.path[:0]=[str(BASE),str(LONG)]
from source_inventory import imported,verify,check_loaded
import run_trial as reference
from observations import save,sha,need
from protocol import UniformOdor,NeuralPropulsion,CausalIntervals


def stop(*_):
    raise TimeoutError('Bounded supervisor stop')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('sham','odor'),required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'blocks').mkdir()
    plan=json.loads((HERE/'PLAN.json').read_text())
    lock=json.loads((HERE/'SOURCES.json').read_text())
    started=time.monotonic()
    obj=session=undo=None
    window=[]
    completed=persisted=cursor=0
    result=dict(status='STARTED',arm=a.arm,requested_ms=plan['duration_ms'],completed_ms=0,
        plan_sha256=sha(HERE/'PLAN.json'),sources_sha256=sha(HERE/'SOURCES.json'),
        plasticity_enabled=False,prior_stable_comparison='Historical FAIL preserved; this is not a parity trial')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    save(a.out/'STATUS.json',result)
    try:
        verify(lock)
        import cupy as cp
        import pandas as pd
        from motor_runtime import load
        from restore_prepared import restore
        from state_compare import qualify_initial
        from runtime_session import RuntimeSession
        from pn_cns_ports import PnCnsPorts
        from hardware import sample
        result['hardware_initial']=sample()
        obj,d,*_=load(a.out/'static_inputs')
        result['restoration']=restore(obj,reference.PREFIX/'prepared_state')
        initial=qualify_initial(obj,reference.PREFIX/'prepared_state')
        need(initial['exact'],'Different prepared history')
        save(a.out/'INITIAL.json',initial)
        h=obj.core.hybrid
        need(not obj.core.plasticity.enabled and not h.pn_online_manifest['general_outputs']['enabled'],'Different neural intervention')
        ports=PnCnsPorts(h,10208)
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            need(name not in sys.modules,'Preloaded event owner '+name)
        sys.path.insert(0,str(reference.EVENTS))
        import event_waveform,event_ports,event_coupling,native_cell,device_cell
        for mod in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            need(Path(mod.__file__).resolve().parent==reference.EVENTS,'Different event owner')
        sys.path.insert(0,str(BASE/'pn'))
        from install_pn import install as install_pn
        result['pn_overlay']=install_pn()
        sys.path.insert(0,str(BASE/'engine'))
        from real_model import install
        undo=install('persistent_fp32')
        session=RuntimeSession(h,'causal_cuda')
        world=obj.core.world
        base=world.boundary.base
        field=UniformOdor(base,world,a.arm,plan)
        # Keep the prepared pending input; the new field affects only future
        # world samples. It cannot edit the motor, neural state or weights.
        world.boundary=field
        obj._validate_adapter()
        motor=NeuralPropulsion(obj)
        auditor=CausalIntervals(obj,field,motor,plan)
        initial_ns=int(h.time_ns)
        yaw0=d.yaw_grados(obj.body.data.qpos)
        nodes=pd.read_parquet(reference.OLD/'data/male_v10/nodes.parquet').sort_values('node_index')
        need(np.array_equal(nodes.bodyId.to_numpy(np.int64),obj.core.brain.node_ids),'Dataset row order')
        types=nodes['type'].fillna('').astype(str)
        mn_rows=np.flatnonzero(types.str.contains('MN',regex=False).to_numpy())
        need(len(mn_rows)>0,'No identified MN observations')
        np.savez_compressed(a.out/'OBSERVED_IDS.npz',DN_ids=obj.dn_ids,MN_ids=obj.core.brain.node_ids[mn_rows],
            ORN_L_ids=obj.core.brain.node_ids[obj.core.port_indices['ORN_DM1_L']],
            ORN_R_ids=obj.core.brain.node_ids[obj.core.port_indices['ORN_DM1_R']])
        initial_weight_digest=sha(reference.PREFIX/'prepared_state/effective_operator.npz')
        import hashlib
        stored_digest=hashlib.sha256(obj.core.brain.W.data.tobytes()).hexdigest()
        factors_digest=hashlib.sha256(obj.core.plasticity.factors.tobytes()).hexdigest()
        np.savez_compressed(a.out/'initial_observation.npz',published_output=obj.core.brain.rates,
            DN_release=h.release()[obj.dn_ix],qpos=obj.body.data.qpos,qvel=obj.body.data.qvel,
            pending_sensors=obj.core.pending_sensors,DN_baseline=obj.dn_baseline,time_ns=np.array(initial_ns))
        result.update(initial_ns=initial_ns,parameters=dict(h.parameters),stored_weight_digest=stored_digest,
            factors_digest=factors_digest,prepared_effective_operator_sha256=initial_weight_digest,
            MN_observation_count=len(mn_rows),load_setup_s=time.monotonic()-started,
            initial_pending_sensors=obj.core.pending_sensors.tolist(),sensory_order=['L','R','reward'])
        save(a.out/'EXECUTED_SOURCES.json',check_loaded(lock))
        print(json.dumps(dict(status='RUNNING',arm=a.arm,step=0,elapsed_s=time.monotonic()-started)),flush=True)
        with (a.out/'PROGRESS.jsonl').open('a',buffering=1) as log:
            for k in range(1,plan['duration_ms']+1):
                used=auditor.before(obj,k)
                tick=time.monotonic()
                obj.step()
                cp.cuda.runtime.deviceSynchronize()
                row=d.captura(obj,ports,'ensayo',k,used,yaw0,cp)
                row.update(forward_unclipped_mm_s=motor.raw_forward,
                    neural_yaw_unapplied_rad_s=motor.raw_yaw,MN_release=h.release()[mn_rows].copy())
                auditor.after(obj,row,k,used)
                need(not obj.core.plasticity.enabled,'Plasticity changed')
                need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<plan['RAM_GiB_max']*1024**3,'RAM budget')
                free,total=cp.cuda.runtime.memGetInfo()
                need(total-free<plan['VRAM_GiB_max']*1024**3,'VRAM budget')
                window.append(row)
                completed=k
                if k%plan['block_ms']==0:
                    first=k-len(window)+1
                    target=a.out/'blocks'/f'{first:05d}_{k:05d}ms'
                    temp=target.with_name(target.name+'.partial')
                    temp.mkdir()
                    arrays={key:np.asarray([r[key] for r in window]) for key in window[0]}
                    for key,value in arrays.items():
                        if value.dtype.kind in 'fc':need(np.isfinite(value).all(),'Nonfinite observation '+key)
                    np.savez_compressed(temp/'traces.npz',**arrays)
                    np.savez_compressed(temp/'published.npz',output=obj.core.brain.rates,time_ns=np.array(h.time_ns))
                    end=len(session.events.audit)
                    save(temp/'events.json',session.events.audit[cursor:end])
                    save(temp/'MANIFEST.json',dict(first_ms=first,last_ms=k,
                        hashes={f.name:sha(f) for f in temp.iterdir()}))
                    temp.replace(target)
                    window=[];persisted=k;cursor=end
                rec=dict(arm=a.arm,step=k,persisted_ms=persisted,elapsed_s=time.monotonic()-started,
                    interval_wall_s=time.monotonic()-tick)
                log.write(json.dumps(rec)+'\n')
                if k==1 or k%100==0:
                    save(a.out/'STATUS.json',{**result,**rec,'status':'RUNNING'})
                    print(json.dumps(rec),flush=True)
        need(hashlib.sha256(obj.core.brain.W.data.tobytes()).hexdigest()==stored_digest,'Stored weights changed')
        need(hashlib.sha256(obj.core.plasticity.factors.tobytes()).hexdigest()==factors_digest,'Plasticity factors changed')
        result['motor']=motor.audit()
        result['runtime']=session.report()
        result['status']='COMPLETE'
        verify(lock)
        check_loaded(lock)
    except BaseException as exc:
        result.update(status='INCOMPLETE',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        print(json.dumps(result['error']),flush=True)
    finally:
        for owner,action in (('session',None if session is None else session.close),('patch',undo),
                             ('organism',None if obj is None else obj.close)):
            if action:
                try:action()
                except BaseException as exc:
                    result.setdefault('cleanup_errors',[]).append(dict(owner=owner,error=repr(exc)))
                    result['status']='INCOMPLETE'
        result.update(completed_ms=completed,persisted_ms=persisted,wall_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save(a.out/'RESULT.json',result)
        save(a.out/'STATUS.json',result)
        print(json.dumps(dict(status=result['status'],completed_ms=completed,persisted_ms=persisted)),flush=True)
    return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
