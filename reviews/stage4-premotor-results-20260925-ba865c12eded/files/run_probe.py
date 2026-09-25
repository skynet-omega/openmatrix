"""One frozen 40+200ms causal recruitment arm, not a navigation admission."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'etapa4_contrast_ablation_20260925_36'
ROOT=HERE.parents[1]
sys.path.insert(0,str(PARENT))
import run_replay as parent
from run_pipeline import RunStorage,atomic_json
from replay_boundary import install as install_replay
from verify_pair import compare_preparation
sys.path.insert(0,str(HERE))
from native_probe import NativeProbe,need
from paired_inputs import PairedInputs


def main():
    need(__debug__,'Historical loader requires normal Python')
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan=json.loads((HERE/'DYNAMIC_PLAN.json').read_text())
    need(a.condition in plan['conditions'],'Unknown condition')
    parent.require_stage3_closed();parent.require_sources()
    lock=json.loads((HERE/'SOURCE_LOCK.json').read_text())
    need(all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in lock.items()),'Probe source changed')
    donor_path=(HERE/plan['donor']).resolve()
    need(hashlib.sha256(donor_path.read_bytes()).hexdigest()==plan['donor_sha256'],'Donor changed')
    with np.load(donor_path,allow_pickle=False) as z:donor={k:z[k] for k in z.files}
    control=HERE/'sham_01'
    if a.condition!='sham':
        r=json.loads((control/'RESULT.json').read_text())
        need(r['status']=='COMPLETE' and r['donor_prefix_exact'],'Sham reference did not reproduce')
    storage=RunStorage(a.out);start=time.perf_counter()
    obj=session=probe=inputs=auditor=None;rows=[];cleanup=[];status='STARTING';error=None;checkpoint=None
    initial=None;phase='loading';donor_exact=True
    old_term=signal.getsignal(signal.SIGTERM)
    def terminated(signum,frame):raise InterruptedError('Bounded process terminated')
    signal.signal(signal.SIGTERM,terminated)

    def snapshot(folder):
        from session_io import write_state
        from operator_state import OperatorState,LEGACY_BINDINGS
        h=obj.core.hybrid
        need(not getattr(obj.core,'failed',False) and not getattr(h,'_native_rebuild_required',False),'Failed state owner')
        write_state(folder/'session',obj.core.state_dict());write_state(folder/'prosthesis',obj.state())
        write_state(folder/'published',dict(rates=obj.core.brain.rates,time_ns=obj.core.brain.time_ns,rng=obj.core.brain.rng.bit_generator.state))
        write_state(folder/'effective_operator',OperatorState(h,LEGACY_BINDINGS).state_dict())
        atomic_json(folder/'boundary.json',obj.core.world.boundary.metadata())
    def save():
        if rows:np.savez_compressed(a.out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        if probe is not None:probe.save()
    try:
        import cupy as cp
        from runtime_session import RuntimeSession
        obj,d,legacy,Field,field,ports=parent.load(a.out/'preparation_inputs')
        h=obj.core.hybrid;initial=int(h.time_ns)
        from kcgamma_regional_brain import _record_hash
        old=copy.deepcopy(h.pn_online_manifest)
        need(old['general_outputs']['enabled'] is True,'Unexpected parent PN route')
        h.pn_online_manifest['general_outputs']['enabled']=False
        h.pn_online_manifest['record_sha256']=_record_hash(h.pn_online_manifest);h.validate_online()
        for name in ('event_waveform','event_ports','event_coupling','native_cell','device_cell'):
            need(name not in sys.modules,'Event owner already loaded: '+name)
        sys.path.insert(0,str(parent.EVENTS))
        import event_waveform,event_ports,event_coupling,native_cell,device_cell
        for module in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            need(Path(module.__file__).resolve().parent==parent.EVENTS,'Wrong event owner')
        session=RuntimeSession(h,'causal_cuda')
        panel=json.loads((HERE.parent/'etapa4_motor_interface_20260925_37/population_01/RESULT.json').read_text())['rows']
        probe=NativeProbe(h,[r['bodyId'] for r in panel],plan['sources'][a.condition],a.out/'probe')
        inputs=PairedInputs(obj,session,donor,a.out,None if a.condition=='sham' else control/'INPUTS.jsonl')
        atomic_json(a.out/'RUN_CONTRACT.json',dict(condition=a.condition,plan_sha256=hashlib.sha256((HERE/'DYNAMIC_PLAN.json').read_bytes()).hexdigest(),
            source_lock=lock,initial_clock_ns=initial,engine='causal_cuda',pulse='native generic current; fixed at first ON evaluation',
            checkpoint_scope='Serialized organism and pulse metadata; no resume validation',stage4_admission=False,stage5_admission=False))
        yaw0=d.yaw_grados(obj.body.data.qpos)
        specs=json.loads((parent.PRIOR/'CAMPOS.json').read_text())
        for phase,duration in [('preparacion',40),('ensayo',200)]:
            if phase=='ensayo':
                storage.snapshot('prepared_state',snapshot)
                equality=compare_preparation(a.out/'prepared_state',PARENT/'identity_01/prepared_state')
                atomic_json(a.out/'PREPARATION_EQUALITY.json',equality)
                need(equality['scientific_state_exact'],'Preparation differs')
                auditor=install_replay(obj,field,'minus',specs['minus'],rows[-1],
                    geometry_path=ROOT/'campanas/etapa4_diseno_20260923_17/GEOMETRY.json',mode='identity',
                    donor=donor_path,donor_sha=plan['donor_sha256'])
                yaw0=d.yaw_grados(obj.body.data.qpos)
                inputs.isolate_body()
            for step in range(1,duration+1):
                need(time.perf_counter()-start<plan['budget']['per_run_wall_s']-50,'Per-run wall budget exhausted')
                need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<=18,'RSS budget')
                used=obj.core.pending_sensors.copy()
                if phase=='ensayo':need(np.array_equal(used,auditor.field.tape.at(h.time_ns)),'Consumed sensory tape differs')
                else:need(np.all(used==0),'Preparation consumed odor')
                inputs.begin(phase,step);probe.begin(phase,step)
                t=time.perf_counter();obj.step();cp.cuda.runtime.deviceSynchronize()
                probe.sample();shadow=inputs.finish()
                row=d.captura(obj,ports,phase,step,used,yaw0,cp)
                row['concentracion_fisica']=(row['concentracion_campo'].copy() if phase=='preparacion' else
                    auditor.field.physical.sample(obj.body.data,None,None)['concentration'].copy())
                index=len(rows)
                exact=all(np.array_equal(row[k],donor[k][index]) for k in donor)
                donor_exact=donor_exact and exact
                if a.condition=='sham' or phase=='preparacion' or step<=20:
                    differing=[k for k in donor if not np.array_equal(row[k],donor[k][index])]
                    need(not differing,'Donor prefix differs: '+str(differing))
                else:
                    for k in ('qpos','qvel','command_forward_mm_s','command_yaw_rate_rad_s','sensores_usados','sensores_pendientes','antenas_mm'):
                        need(np.array_equal(row[k],donor[k][index]),'Paired body/tape differs: '+k)
                if phase=='ensayo':
                    need(np.array_equal(row['sensores_pendientes'],auditor.field.tape.at(h.time_ns)),'Pending sensory tape differs')
                    dq=row['DN_q_usada']-row['DN_baseline']
                    expected=np.array([float(np.clip(.2+np.mean(dq[:2]),0,.5)),float(np.tanh(250*(dq[2]-dq[3]))*np.deg2rad(5))])
                    need(np.array_equal(shadow,expected),'Observed original neural readout differs')
                else:shadow=np.array([row['command_forward_mm_s'],row['command_yaw_rate_rad_s']])
                row['neural_command_shadow']=shadow;rows.append(row)
                free,total=cp.cuda.runtime.memGetInfo()
                need((total-free)/1024**3<=12,'GPU budget')
                progress=dict(condition=a.condition,phase=phase,step=step,clock_ns=int(h.time_ns),
                    step_wall_s=time.perf_counter()-t,elapsed_s=time.perf_counter()-start,
                    rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,gpu_used_gib=(total-free)/1024**3)
                with (a.out/'PROGRESS.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
                if step%10==0:
                    save();print(json.dumps(progress),flush=True)
        status='COMPLETE'
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),numerical_diagnostic=getattr(exc,'numerical_diagnostic',None))
        status='INCOMPLETE'
    finally:
        try:save()
        except BaseException as exc:cleanup.append('save: '+repr(exc))
        if obj is not None:
            try:checkpoint=storage.snapshot('final_state',snapshot)
            except BaseException as exc:cleanup.append('checkpoint: '+repr(exc))
        runtime=None
        if session is not None:
            try:runtime=session.report()
            except BaseException as exc:cleanup.append('report: '+repr(exc))
        for close in ([inputs.close] if inputs else [])+([probe.close] if probe else [])+([session.close] if session else [])+([obj.close] if obj else []):
            try:close()
            except BaseException as exc:cleanup.append('close: '+repr(exc))
        signal.signal(signal.SIGTERM,old_term)
        if cleanup:status='INCOMPLETE_CLEANUP'
        atomic_json(a.out/'RESULT.json',dict(status=status,condition=a.condition,error=error,cleanup_errors=cleanup,
            completed_preparation_ms=sum(r['fase']=='preparacion' for r in rows),completed_trial_ms=sum(r['fase']=='ensayo' for r in rows),
            wall_total_s=time.perf_counter()-start,donor_prefix_exact=donor_exact,checkpoint=checkpoint,runtime=runtime,
            stage4_admission=False,stage5_admission=False,scope='Fixed-input causal recruitment; body follows common recorded commands.'))
    return 0 if status=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
