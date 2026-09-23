"""Unchanged PN629 reference runner, with external support and complete operator capture."""
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
    if arm not in ('odor_right','sham','odor_left','uniform'):raise ValueError('Unknown reference arm')
    if out.exists():raise FileExistsError(out)
    frozen=json.loads((HERE/'EXECUTION_SOURCES.json').read_text())
    for name,digest in frozen.items():
        if sha(Path(name))!=digest:raise ValueError('Prospective source changed: '+name)
    original_load=runner.load;original_snapshot=runner.RunStorage.snapshot
    holder={};samples=[]
    def load(path):
        values=original_load(path);obj,d,*_=values
        holder['object']=obj
        observer=mechanics.MechanicalObserver(obj.body).start();holder['observer']=observer
        initial=observer.read();holder['initial_support']=initial
        capture=d.captura
        def captured(*args,**kwargs):
            result=capture(*args,**kwargs)
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
    sys.argv=[str(PARENT/'run_set.py'),'--out',str(out),'--odor',arm,'--engine','reference_cuda','--ms','400','--observe','on']
    try:code=runner.main()
    finally:
        if 'observer' in holder:holder['observer'].close()
        if 'capture_owner' in holder:holder['capture_owner'].captura=holder['capture_original']
        runner.load=original_load;runner.RunStorage.snapshot=original_snapshot
        if out.is_dir():
            folder=out/'experiment_sources';folder.mkdir(exist_ok=False)
            for i,(name,digest) in enumerate(frozen.items()):
                source=Path(name);destination=folder/(str(i)+'__'+source.name);shutil.copyfile(source,destination)
            runner.atomic_json(out/'EXPERIMENT_CONTRACT.json',{'plan_sha256':sha(HERE/'PLAN.json'),'sources':frozen,
               'changes':'Read-only mechanical observer and effective-operator serialization; neural engine/equations and PN intervention unchanged',
               'stage3_admission':False})
            if samples:
                np.savez_compressed(out/'SUPPORT.npz',phase=np.asarray([s['phase'] for s in samples]),
                    step=np.asarray([s['step'] for s in samples]),clock_ns=np.asarray([s['clock_ns'] for s in samples],dtype=np.int64),
                    duration_s=np.asarray([0.]+[s['duration_s'] for s in samples]),
                    vertical_impulse_Ns=np.asarray([[0.,0.,0.]]+[s['vertical_impulse_Ns'] for s in samples]),
                    sample_steps=np.asarray([0]+[s['sample_steps'] for s in samples]),mg_N=np.float64(holder['initial_support']['mgN']))
    return code
if __name__=='__main__':raise SystemExit(main())
