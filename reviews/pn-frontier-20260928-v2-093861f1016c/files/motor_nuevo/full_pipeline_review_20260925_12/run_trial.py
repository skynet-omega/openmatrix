"""Serial, bounded stable/reviewed whole-organism trials under common physics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse, hashlib, json, signal, sys, time, traceback, resource, cProfile, pstats
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
STABLE=ROOT/'campanas/etapa45_navigation_wind_20260925_40'
PREFIX=STABLE/'navigation_minus_filtered_wind_03'
PARENT=ROOT/'motor_nuevo/persistent_fp32_20260925_07'
from state_compare import qualify_initial
EVENTS=ROOT/'campanas/etapa3_pn629_intervention_20260923_15'
sys.path[:0]=[str(OLD/'work/motor14_20260922'),str(OLD/'work/motor13_20260922'),
             str(ROOT/'motor_nuevo/pipeline_review_20260922'),str(STABLE),str(EVENTS)]

def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def collect(obj,cp):
    h=obj.core.hybrid;b=h._spatial_batch
    while hasattr(b,'base'):b=b.base
    row={'time_ns':np.asarray(h.time_ns,dtype=np.int64),'cns':h.state.copy(),
         'body_qpos':obj.body.data.qpos.copy(),'body_qvel':obj.body.data.qvel.copy(),
         'pending_sensors':obj.core.pending_sensors.copy()}
    for name in ('delta','gates','q','counts','clipped','last_siz','previous_slope','trough'):
        row['cell_'+name]=cp.asnumpy(cp.asarray(getattr(b,name)))
    for name,value in h._axonal_release.state.items():
        if isinstance(value,np.ndarray):row['axon_'+name]=value.copy()
    for name,value in row.items():
        if value.dtype.kind in 'fc':need(np.isfinite(value).all(),'Nonfinite '+name)
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=('stable','reviewed'),required=True)
    p.add_argument('--ms',type=int,choices=(20,100,2000),default=100)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wall-limit',type=int,default=400)
    p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    need((a.ms==20)==a.diagnostic,'20ms is diagnostic only')
    started=time.perf_counter();obj=session=undo=None;rows=[];states=[];progress=[]
    ENGINE=HERE/'engine'
    from source_inventory import imported, verify, check_loaded
    source_lock={} if a.diagnostic else json.loads((HERE/'SOURCES.json').read_text())
    verify(source_lock)
    from hardware import sample
    hardware=[{'step':0,**sample()}]
    result={'status':'STARTED','engine':a.engine,'requested_ms':a.ms,'completed_ms':0,
            'parameters_changed':False,'event_owner':str(EVENTS),
            'scope':'Whole organism from campaign40 preparation; continuous corrected world wind, common PN setup repair',
            'source_checkpoint':str(PREFIX/'prepared_state'),
            'source_trace_sha256':sha(PREFIX/'traces.npz'),
            'runner_sha256':sha(__file__),'restore_sha256':sha(HERE/'restore_prepared.py'),'plan_sha256':sha(HERE/'PLAN.json'),'sources_sha256':None if a.diagnostic else sha(HERE/'SOURCES.json')}
    def stop(*_):raise TimeoutError('Finite comparison wall budget')
    previous=signal.signal(signal.SIGALRM,stop);signal.alarm(a.wall_limit)
    try:
        need(__debug__,'Historical loader requires normal Python')
        import cupy as cp
        from motor_runtime import load
        from runtime_session import RuntimeSession
        from restore_prepared import restore
        from gaussiano_400 import install as install_gaussian
        from motor_wind_fixed import MotorWind
        from full_snapshot import snapshot
        from session_io import write_state
        from pn_cns_ports import PnCnsPorts
        lock=json.loads((STABLE/'SOURCE_LOCK.json').read_text())
        need(all(sha(k)==v for k,v in lock.items()),'Historical source identity changed')
        obj,d,plan,Field,base,old_ports=load(a.out/'static_inputs')
        result['restoration']=restore(obj,PREFIX/'prepared_state')
        result['initial']=qualify_initial(obj,PREFIX/'prepared_state')
        save(a.out/'INITIAL.json',result['initial'])
        need(result['initial']['exact'],'Initial state differs from recorded stable preparation')
        h=obj.core.hybrid;ports=PnCnsPorts(h,10208)
        need(h.pn_online_manifest['general_outputs']['enabled'] is False,'PN intervention differs')
        result['parameters']=dict(h.parameters)
        with np.load(PREFIX/'traces.npz',allow_pickle=False) as z:
            prepared={k:z[k][39].copy() for k in z.files}
        need(prepared['fase']=='preparacion' and prepared['paso']==40,'Wrong reference row')
        need(np.array_equal(obj.body.data.qpos,prepared['qpos']),'Prepared body differs')
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            need(name not in sys.modules,'Preloaded event owner '+name)
        sys.path.insert(0,str(EVENTS))
        import event_waveform,event_ports,event_coupling,native_cell,device_cell
        for module in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            need(Path(module.__file__).resolve().parent==EVENTS,'Wrong event owner')
        sys.path.insert(0,str(HERE/'pn'))
        from install_pn import install as install_pn_overlay
        result['pn_overlay']=install_pn_overlay()
        if a.engine=='reviewed':
            sys.path.insert(0,str(ENGINE))
            from real_model import install
            result['engine_path']=str(ENGINE)
            result['engine_library_sha256']=sha(ENGINE/'libresident_controller.so')
            undo=install('persistent_fp32')
        session=RuntimeSession(h,'causal_cuda')
        specs=json.loads((PREFIX/'GAUSSIAN_SPEC.json').read_text())
        auditor=install_gaussian(obj,obj.core.world.boundary.base,'minus',specs['minus'],prepared,
                                geometry_path=ROOT/'campanas/etapa4_diseno_20260923_17/GEOMETRY.json')
        motor=MotorWind(obj)
        yaw0=d.yaw_grados(obj.body.data.qpos);initial_ns=int(h.time_ns)
        result['load_setup_s']=time.perf_counter()-started
        result['runtime_versions']={'numpy':np.__version__,'cupy':cp.__version__,'python':sys.version}
        result['device']=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        states.append(collect(obj,cp))
        loaded=imported()
        verify(loaded)
        if not a.diagnostic:check_loaded(source_lock)
        save(a.out/'EXECUTED_SOURCES_SETUP.json',loaded)
        profile=cProfile.Profile() if a.diagnostic else None
        zero=a.out/'state_0ms';zero.mkdir()
        write_state(zero/'pn_state',obj.core.state_dict()['hybrid']['pn_online_state'])
        write_state(zero/'published',{'rates':obj.core.brain.rates,'time_ns':obj.core.brain.time_ns,
                                    'rng':obj.core.brain.rng.bit_generator.state})
        for k in range(1,a.ms+1):
            used=auditor.before(obj);tick=time.perf_counter()
            if profile:profile.enable()
            try:obj.step();cp.cuda.runtime.deviceSynchronize()
            finally:
                if profile:profile.disable()
            advance=time.perf_counter()-tick
            row=d.captura(obj,ports,'ensayo',k,used,yaw0,cp)
            row.update(neural_command_raw_rad_s=motor.raw,motor_filter_applied_rad_s=motor.applied,
                       motor_filter_state_rad_s=motor.filtered,wind_torque_native=(motor.WIND_TORQUE_NATIVE if motor.WIND_FIRST_STEP<=k<=motor.WIND_LAST_STEP else 0.))
            auditor.after(obj,row)
            need(motor.trial_step==k and motor.body_calls==40*k,'Motor/body clocks differ')
            need(h.time_ns==initial_ns+k*1_000_000,'CNS clock differs')
            expected_wind=max(0,min(k,motor.WIND_LAST_STEP)-motor.WIND_FIRST_STEP+1)*40
            need(motor.wind_substeps==expected_wind,'Wind physical substep count differs')
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 24*1024**3,'RAM budget')
            free,total=cp.cuda.runtime.memGetInfo()
            need(total-free < 14*1024**3,'VRAM budget')
            rows.append(row)
            if k in (1,20,100,1000,1020,1500,a.ms):states.append(collect(obj,cp))
            if k in (20,100,1000,1020,1500,a.ms):
                checkpoint=a.out/('state_'+str(k)+'ms');checkpoint.mkdir()
                write_state(checkpoint/'pn_state',obj.core.state_dict()['hybrid']['pn_online_state'])
                write_state(checkpoint/'published',{'rates':obj.core.brain.rates,'time_ns':obj.core.brain.time_ns,
                                                  'rng':obj.core.brain.rng.bit_generator.state})
            if k%100==0:
                np.savez_compressed(a.out/'traces_prefix.npz',**{key:np.asarray([r[key] for r in rows]) for key in rows[0]})
                np.savez_compressed(a.out/'neural_prefix.npz',**{key:np.stack([r[key] for r in states]) for key in states[0]})
            record={'step':k,'time_ns':int(h.time_ns),'advance_s':advance,
                    'elapsed_s':time.perf_counter()-started}
            progress.append(record);result['completed_ms']=k
            with (a.out/'PROGRESS.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if k==1 or k%10==0:print(json.dumps(record),flush=True)
            if k==1 or k%100==0:hardware.append({'step':k,**sample()})
        final=a.out/'final_state';final.mkdir()
        write_state(final/'pn_state',obj.core.state_dict()['hybrid']['pn_online_state'])
        write_state(final/'published',{'rates':obj.core.brain.rates,'time_ns':obj.core.brain.time_ns,
                                      'rng':obj.core.brain.rng.bit_generator.state})
        save(a.out/'MOTOR.json',motor.audit())
        if motor.wind_log:
            np.savez_compressed(a.out/'wind_substeps.npz',**{key:np.asarray([r[key] for r in motor.wind_log]) for key in motor.wind_log[0]})
        if profile:
            stats=pstats.Stats(profile)
            records=[dict(file=key[0],line=key[1],name=key[2],primitive_calls=value[0],calls=value[1],own_s=value[2],cumulative_s=value[3]) for key,value in stats.stats.items()]
            save(a.out/'PROFILE.json',sorted(records,key=lambda r:r['cumulative_s'],reverse=True))
        if a.ms==2000:
            snapshot(a.out/'scientific_final',obj,motor,auditor)
        save(a.out/'EXECUTED_SOURCES_FINAL.json',imported())
        verify(loaded)
        if not a.diagnostic:check_loaded(source_lock)
        operator=getattr(session.adapter,'_fp32_operator',None)
        if operator is not None:
            result['weight_mirror']=operator.mirror.report()
            result['weight_mirror']['writer_groups']=operator.writer_groups
        result['status']='COMPLETE'
    except BaseException as exc:
        result.update(status='INCOMPLETE',error={'type':type(exc).__name__,'message':str(exc),
                      'traceback':traceback.format_exc()})
        print(json.dumps({'error':str(exc)}),flush=True)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if rows:np.savez_compressed(a.out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        if states:np.savez_compressed(a.out/'neural_states.npz',**{k:np.stack([r[k] for r in states]) for k in states[0]})
        if session is not None:
            result['runtime']=session.report();save(a.out/'EVENTS.json',session.events.audit)
            if session.adapter.core and hasattr(session.adapter.core,'resident_wall_s'):
                result['cns_resident_s']=session.adapter.core.resident_wall_s
            try:session.close()
            except BaseException as exc:result.setdefault('cleanup_errors',[]).append(repr(exc))
        if undo:
            try:undo()
            except BaseException as exc:result.setdefault('cleanup_errors',[]).append(repr(exc))
        if obj:
            try:obj.close()
            except BaseException as exc:result.setdefault('cleanup_errors',[]).append(repr(exc))
        if result.get('cleanup_errors'):result['status']='INCOMPLETE'
        from threadpoolctl import threadpool_info
        result['thread_pools']=threadpool_info()
        result['advance_total_s']=sum(r['advance_s'] for r in progress)
        result['hardware_samples']=hardware
        result['wall_total_s']=time.perf_counter()-started
        result['resources_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if result['wall_total_s']>a.wall_limit:result['status']='INCOMPLETE_WALL_BUDGET'
        save(a.out/'RESULT.json',result)
        print(json.dumps({k:result[k] for k in ('status','engine','completed_ms','advance_total_s','wall_total_s')}),flush=True)
    return 0 if result['status']=='COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())
