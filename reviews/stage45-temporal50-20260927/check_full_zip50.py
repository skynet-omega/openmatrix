"""Independent all-file ZIP recipe verification and fresh recorded-data extraction."""
import argparse,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('destination',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    start=time.process_time();wall=time.monotonic()
    if a.destination.exists() or a.output.exists():raise ValueError('Preserve previous extraction/check')
    a.destination.mkdir(parents=True)
    prefix='AXIOMA_ASTRA/campanas/etapa45_temporal_factorial_20260927_50/'
    checked=0;extracted=0
    with zipfile.ZipFile(a.archive) as z:
        m=json.loads(z.read('MANIFEST.json'))
        for entry in m['files']:
            path=Path(entry['path'])
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe file path')
            selected=entry['path'].startswith(prefix) and '/final_state/' not in entry['path'] and '/static_inputs/' not in entry['path']
            target=a.destination/path
            stream=None
            if selected:target.parent.mkdir(parents=True,exist_ok=True);stream=target.open('xb')
            d=hashlib.sha256();size=0
            try:
                for piece in entry.get('segments',[dict(storage=entry.get('storage'))]):
                    with z.open(piece['storage']) as f:
                        for data in iter(lambda:f.read(4*1024**2),b''):
                            d.update(data);size+=len(data)
                            if stream is not None:stream.write(data)
            finally:
                if stream is not None:stream.close()
            if size!=entry['bytes'] or d.hexdigest()!=entry['sha256']:raise ValueError('File recipe mismatch '+entry['path'])
            checked+=1;extracted+=selected
    root=a.destination/prefix;checks=[]
    for args in [['-O','verify_complete50.py','--without-manifest'],['-O','test_analysis50.py','--real'],['-O','test_law_supplement50.py'],['-O','verify_supplements50.py']]:
        r=subprocess.run([sys.executable,*args],cwd=root,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=90)
        if r.returncode:raise ValueError(r.stderr)
        checks.append(json.loads(r.stdout))
    out=dict(all_file_recipes_verified=checked,recorded_evidence_files_extracted=extracted,checks=checks,
        scope='Every archived file hash verified; only recorded50 subset materialized/executed. Full CNS not rerun.',
        CPU_s=time.process_time()-start,wall_s=time.monotonic()-wall,new_neural_ms=0)
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['all_file_recipes_verified','recorded_evidence_files_extracted','CPU_s','wall_s','scope']}))
if __name__=='__main__':main()
