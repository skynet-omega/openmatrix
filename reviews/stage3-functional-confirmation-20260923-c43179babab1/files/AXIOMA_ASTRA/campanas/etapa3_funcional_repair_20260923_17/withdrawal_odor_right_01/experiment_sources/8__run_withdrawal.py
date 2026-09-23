"""Coupled angular-reader withdrawal after the unchanged40ms preparation."""
from pathlib import Path
import importlib.util,json,hashlib,shutil,sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=HERE.parent/'etapa3_pn629_intervention_20260923_15'
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('pn629_preserved_runner',PARENT/'run_set.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
spec=importlib.util.spec_from_file_location('support_observer_preserved',HERE/'mechanical_observer_original.py')
mechanics=importlib.util.module_from_spec(spec);spec.loader.exec_module(mechanics)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    arm=sys.argv[1];out=Path(sys.argv[2]).resolve()
    if arm not in ('odor_right','odor_left'):raise ValueError('Unknown withdrawal arm')
    if out.exists():raise FileExistsError(out)
    frozen=json.loads((HERE/'WITHDRAWAL_SOURCES.json').read_text())
    for name,digest in frozen.items():
        if sha(Path(name))!=digest:raise ValueError('Prospective source changed: '+name)
    original_load=runner.load;original_snapshot=runner.RunStorage.snapshot
    holder={};samples=[];withdrawal={'active':False,'clamped_physical_calls':0,'prepared_physical_calls':0,'max_requested_yaw_rad_s':0.}
    def load(path):
        values=original_load(path);obj,d,*_=values
        holder['object']=obj
        observer=mechanics.MechanicalObserver(obj.body).start();holder['observer']=observer
        initial=observer.read();holder['initial_support']=initial
        install=d.instalar_campo;command=obj.controller.set_command;controller=obj.controller
        def install_trial(*args,**kwargs):
            value=install(*args,**kwargs)
            if obj.core.hybrid.time_ns!=holder['initial_clock_ns']+40_000_000:
                raise ValueError('Withdrawal must start after exactly40ms of unchanged preparation')
            withdrawal['active']=True
            return value
        def angular_clamp(*,forward_mm_s,yaw_rate_rad_s):
            if withdrawal['active']:
                withdrawal['clamped_physical_calls']+=1
                withdrawal['max_requested_yaw_rad_s']=max(withdrawal['max_requested_yaw_rad_s'],abs(float(yaw_rate_rad_s)))
                return command(forward_mm_s=forward_mm_s,yaw_rate_rad_s=0.)
            withdrawal['prepared_physical_calls']+=1
            return command(forward_mm_s=forward_mm_s,yaw_rate_rad_s=yaw_rate_rad_s)
        holder['initial_clock_ns']=int(obj.core.hybrid.time_ns)
        holder['install_original']=install;holder['controller']=controller;holder['command_original']=command
        d.instalar_campo=install_trial;controller.set_command=angular_clamp
        capture=d.captura
        def captured(*args,**kwargs):
            if obj.controller is not controller:raise RuntimeError('Controller owner changed')
            result=capture(*args,**kwargs)
            if withdrawal['active'] and float(result['command_yaw_rate_rad_s'])!=0.:
                raise ValueError('Angular withdrawal was not consumed')
            record=observer.read();record.update(phase=result['fase'],step=int(result['paso']),clock_ns=int(result['CNS_time_ns']))
            samples.append(record)
            return result
        d.captura=captured
        holder['capture_owner']=d;holder['capture_original']=capture
        return values
    def snapshot(storage,name,writer):
        def extended(folder):
            writer(folder)
            from operator_state import OperatorState,LEGACY_BINDINGS
            from session_io import write_state
            obj=holder['object'];h=obj.core.hybrid
            write_state(folder/'effective_operator',OperatorState(h,LEGACY_BINDINGS).state_dict())
            runner.atomic_json(folder/'external_support_observer.json',holder['observer'].read())
        return original_snapshot(storage,name,extended)
    runner.load=load;runner.RunStorage.snapshot=snapshot
    sys.argv=[str(PARENT/'run_set.py'),'--out',str(out),'--odor',arm,'--engine','causal_cuda','--ms','400','--observe','on']
    try:code=runner.main()
    finally:
        if 'observer' in holder:holder['observer'].close()
        if 'capture_owner' in holder:
            holder['capture_owner'].captura=holder['capture_original']
            holder['capture_owner'].instalar_campo=holder['install_original']
            holder['controller'].set_command=holder['command_original']
        runner.load=original_load;runner.RunStorage.snapshot=original_snapshot
        if out.is_dir():
            folder=out/'experiment_sources';folder.mkdir(exist_ok=False)
            for i,(name,digest) in enumerate(frozen.items()):
                source=Path(name);destination=folder/(str(i)+'__'+source.name);shutil.copyfile(source,destination)
            runner.atomic_json(out/'EXPERIMENT_CONTRACT.json',{'plan_sha256':sha(HERE/'PLAN.json'),'sources':frozen,
               'changes':'Only angular command is clamped to zero after40ms; forward command, CNS, proprioception and spatial odor remain coupled. Read-only support and operator capture.',
               'stage3_admission':False})
            runner.atomic_json(out/'WITHDRAWAL.json',withdrawal)
            if samples:
                np.savez_compressed(out/'SUPPORT.npz',phase=np.asarray([s['phase'] for s in samples]),
                    step=np.asarray([s['step'] for s in samples]),clock_ns=np.asarray([s['clock_ns'] for s in samples],dtype=np.int64),
                    duration_s=np.asarray([0.]+[s['duration_s'] for s in samples]),
                    vertical_impulse_Ns=np.asarray([[0.,0.,0.]]+[s['vertical_impulse_Ns'] for s in samples]),
                    sample_steps=np.asarray([0]+[s['sample_steps'] for s in samples]),mg_N=np.float64(holder['initial_support']['mgN']))
    return code
if __name__=='__main__':raise SystemExit(main())
