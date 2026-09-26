"""Package only pertinent code and saved evidence; never credentials or caches."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NAME='MOTOR_EVENT_MEMORY_RK3_RESULTADOS.zip'

def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    target=HERE/NAME
    if target.exists():raise ValueError('Archive output must be new')
    files=set()
    def add(p):
        if p.is_file():files.add(p.resolve())
    def run_evidence(base,initial_zero=True):
        for name in ('RESULT.json','INITIAL.json','MOTOR.json','EVENTS.json','traces.npz','neural_states.npz','PROGRESS.jsonl'):add(base/name)
        for stage in (0,100,1000):
            if stage==0 and not initial_zero:continue
            folder=base/f'state_{stage}ms'
            if folder.is_dir():
                for f in folder.iterdir():
                    if f.suffix in ('.json','.npz'):add(f)
    for f in HERE.iterdir():
        if f.is_file() and f.suffix in ('.py','.md','.json','.diff','.log') and f.name!='LIVE_COMPARISON.json':add(f)
    for f in (HERE/'engine').iterdir():
        if f.is_file() and f.suffix in ('.py','.cu','.so'):add(f)
    for f in (HERE/'jev_01').iterdir():
        if f.suffix=='.json':add(f)
    for name in ('control_100ms_01','candidate_100ms_01','candidate_1000ms_01'):run_evidence(HERE/name)
    older=HERE.parent/'equivalence_1s_20260925_09'
    run_evidence(older/'optimized_1000ms_01',False)
    ref=older/'reference';add(ref/'MANIFEST.json')
    for name in json.loads((ref/'MANIFEST.json').read_text()):add(ref/name)
    parent=HERE.parent/'persistent_fp32_20260925_07'
    add(parent/'SOURCE_MANIFEST.json');add(parent/'build.py')
    for name in json.loads((parent/'SOURCE_MANIFEST.json').read_text()):add(parent/name)
    add(HERE.parent/'neurocore_real_20260925_01/check_runtime.py')
    metadata={}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in sorted(files):
            name=str(f.relative_to(ROOT));data=f.read_bytes()
            metadata[name]={'bytes':len(data),'sha256':digest(data)}
            z.writestr(name,data)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(metadata,indent=2)+'\n')
        z.writestr('REPRODUCIR.txt','Desde motor_nuevo/event_memory_rk3_20260925_11, ejecutar python -O verify_saved.py --long.\nReproduce métricas guardadas en CPU; requiere NumPy. No reinicia ni simula el organismo.\nLa simulación requiere el proyecto/checkpoint original, no incluido en este paquete de comparación.\n')
    with tempfile.TemporaryDirectory(prefix='event-memory-archive-') as directory:
        root=Path(directory)
        with zipfile.ZipFile(target) as z:
            names=z.namelist()
            if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('Unsafe archive')
            z.extractall(root)
        for name,item in metadata.items():
            data=(root/name).read_bytes()
            if len(data)!=item['bytes'] or digest(data)!=item['sha256']:raise ValueError('Archive payload differs '+name)
        report=root/'motor_nuevo/event_memory_rk3_20260925_11'
        run=subprocess.run([sys.executable,'-O','verify_saved.py','--long'],cwd=report,capture_output=True,text=True,timeout=60)
        if run.returncode:raise RuntimeError('Cold verification failed: '+run.stderr)
        verification=json.loads(run.stdout)
    result={'archive':str(target),'bytes':target.stat().st_size,'sha256':digest(target.read_bytes()),
            'files':len(files),'cold_verification':verification,
            'scope':'Complete saved-comparison reproduction; core sources and artifacts included. Full organism restart is not packaged.'}
    (HERE/'REPRODUCTION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
