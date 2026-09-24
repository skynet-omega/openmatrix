"""One CPU postclose recomputation; never launches or advances an organism."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
HISTORICAL=Path('/home/daroch/AXIOMA_FLYWIRE')
CONFIRMED='CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _pairs(items):
    out={}
    for key,value in items:
        require(key not in out,'Duplicate JSON key: '+key)
        out[key]=value
    return out


def read(path):
    def invalid(value):raise ValueError('Nonfinite JSON token: '+value)
    return json.loads(Path(path).read_text(),object_pairs_hook=_pairs,parse_constant=invalid)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def closed_queue(repair):
    queue=read(Path(repair)/'QUEUE.json')
    require(queue.get('state')=='COMPLETE','Repair19 QUEUE must be COMPLETE before recomputation')
    return queue


def rebuild(repair,load_verifier):
    """load_verifier is delayed until COMPLETE; injectable only for CPU fixtures."""
    repair=Path(repair)
    before=closed_queue(repair)
    stored=read(repair/'FINAL_RAW_VERIFIED.json')
    fresh=load_verifier()()
    require(isinstance(fresh,dict),'Raw verifier returned a non-object')
    require(fresh.get('classification')==CONFIRMED and fresh.get('functional_stage3_pass') is True,
            'Recomputed functional Stage3 did not pass')
    original=fresh.get('original_raw_result',{})
    require(original.get('functional_stage3_pass') is True and original.get('classification')==CONFIRMED,
            'Recomputed original scientific gate did not pass')
    require(fresh.get('strict_original_contract_fulfilled') is False,
            'Historical strict-contract limitation changed')
    require(canonical(fresh)==canonical(stored),'Complete reconstructed result differs from FINAL_RAW_VERIFIED')
    require(canonical(before)==canonical(closed_queue(repair)),'Repair19 QUEUE changed while checking')
    return fresh


class ReadOnlyInputs:
    """Audit local reads; block source writes and changed inputs during checking."""
    def __init__(self,output_root,roots):
        self.output_root=Path(output_root).resolve()
        self.roots=tuple(Path(p).resolve() for p in roots)
        self.records={};self.active=False;self.busy=False

    def path(self,value):
        if isinstance(value,(str,bytes,os.PathLike)):return Path(os.fsdecode(value)).resolve()
        return None

    def writable(self,path):
        require(path is not None and path.is_relative_to(self.output_root),
                'Read-only verification attempted an outside write: '+str(path))

    def __call__(self,event,args):
        if not self.active or self.busy:return
        if event=='import':
            name=str(args[0]).split('.')[0]
            require(name not in {'cupy','mujoco','torch','pycuda','motor_runtime','runtime_session','run_continuation'},
                    'GPU/organism module forbidden in read-only verification: '+name)
        elif event=='ctypes.dlopen':
            name=str(args[0]).lower()
            require(not any(s in name for s in ('libcuda','libcupti','libcudart')),'GPU library forbidden')
        elif event=='open':
            path=self.path(args[0]);mode=args[1];flags=args[2]
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if write:self.writable(path)
            elif path is not None and any(path.is_relative_to(root) for root in self.roots) and path.is_file():
                key=str(path)
                if key not in self.records:
                    self.busy=True
                    try:self.records[key]={'sha256':sha(path),'bytes':path.stat().st_size}
                    finally:self.busy=False
        elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate'):
            self.writable(self.path(args[0]))
        elif event in ('os.rename','os.link'):
            self.writable(self.path(args[0]));self.writable(self.path(args[1]))
        elif event=='os.symlink':self.writable(self.path(args[1]))

    def finish(self):
        self.busy=True
        try:
            for path,record in self.records.items():
                require(Path(path).is_file() and Path(path).stat().st_size==record['bytes'] and sha(path)==record['sha256'],
                        'An input changed during recomputation: '+path)
        finally:self.busy=False
        return self.records


def check_sources():
    lock=read(HERE/'SOURCE_LOCK.json')
    for path,digest in lock.items():require(sha(path)==digest,'Frozen postclose source changed: '+path)
    return lock


def load_actual(repair):
    expected={
        'verify_repair19':Path(repair)/'verify_repair19.py',
        'verify_repair':ROOT/'campanas/etapa3_funcional_repair_20260923_17/verify_repair.py',
        'verify_all':ROOT/'campanas/etapa3_funcional_20260923_16/verify_all.py',
        'check_remaining':ROOT/'campanas/etapa3_funcional_20260923_16/check_remaining.py',
        'checkpoint_compare':ROOT/'campanas/etapa3_funcional_20260923_16/checkpoint_compare.py',
        'check_support':ROOT/'campanas/etapa3_funcional_20260923_16/check_support.py'}
    require(not any(name in sys.modules for name in expected),'Verifier requires a fresh unambiguous process')
    path=expected['verify_repair19']
    # This import is CPU-only; verify_repair19 imports original verifier helpers.
    spec=importlib.util.spec_from_file_location('verify_repair19',path)
    module=importlib.util.module_from_spec(spec);sys.modules['verify_repair19']=module
    spec.loader.exec_module(module)
    for name,source in expected.items():
        require(name in sys.modules and Path(sys.modules[name].__file__).resolve()==source.resolve(),
                'Verifier dependency loaded from an unexpected source: '+name)
    return module.verify


def main():
    require(len(sys.argv)==1,'No alternate roots, thresholds or output paths accepted')
    sys.dont_write_bytecode=True
    plan=read(HERE/'PLAN.json');repair=Path(plan['repair_directory'])
    # Reject premature launch before creating a receipt or importing the verifier.
    closed_queue(repair)
    check_sources()
    budget=plan['budget']
    resource.setrlimit(resource.RLIMIT_CPU,(budget['future_cpu_seconds_max'],budget['future_cpu_seconds_max']+1))
    memory=budget['future_address_space_gib_max']*1024**3
    resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
    def deadline(*_):raise TimeoutError('Frozen postclose wall/CPU budget exhausted')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline)
    signal.setitimer(signal.ITIMER_REAL,budget['future_wall_seconds_max'])
    start=time.perf_counter();cpu=time.process_time()
    receipt_path=HERE/plan['receipt']
    # Exclusive creation reserves the sole real attempt; failures are never overwritten.
    with receipt_path.open('x') as output:
        tracker=ReadOnlyInputs(HERE,(ROOT,HISTORICAL));sys.addaudithook(tracker);tracker.active=True
        value={'schema':'independent_stage3_postclose_receipt_v1','classification':'BLOQUEADO',
               'functional_stage3_pass':False,'gpu_calls':0,'organism_loads':0,
               'new_neural_or_physical_steps':0,'attempts':1}
        try:
            sources=check_sources()
            fresh=rebuild(repair,lambda:load_actual(repair))
            inputs=tracker.finish()
            value.update(classification=CONFIRMED,functional_stage3_pass=True,
                strict_original_contract_fulfilled=False,full_result_exact=True,
                rebuilt_raw_result=fresh,source_hashes=sources,read_input_hashes=inputs,
                final_raw_sha256=sha(repair/'FINAL_RAW_VERIFIED.json'),
                verifier_sha256=sha(repair/'verify_repair19.py'),
                contract_sha256=sha(repair/'REPAIR19_CONTRACT.json'),
                queue_sha256=sha(repair/'QUEUE.json'))
            code=0
        except BaseException as exc:
            value.update(error={'type':type(exc).__name__,'message':str(exc)},
                         read_input_hashes=tracker.records,full_result_exact=False)
            code=2
        finally:
            tracker.active=False;signal.setitimer(signal.ITIMER_REAL,0)
        value.update(cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-start,
                     peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                     python_version=platform.python_version(),numpy_version=getattr(sys.modules.get('numpy'),'__version__',None),
                     scope='Independent postclose recomputation using the unchanged scientific verifier; functional scope only')
        output.write(json.dumps(value,indent=2,allow_nan=False)+'\n');output.flush();os.fsync(output.fileno())
    print(json.dumps({'classification':value['classification'],'functional_stage3_pass':value['functional_stage3_pass'],'receipt':str(receipt_path)}))
    return code


if __name__=='__main__':raise SystemExit(main())
