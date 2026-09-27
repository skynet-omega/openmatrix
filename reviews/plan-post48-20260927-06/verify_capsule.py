"""Verify the portable diagnostic, including real corruption cases, without CNS."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
FILES=['check_recorded_law.py','OFFLINE_PROTOCOL.json','OFFLINE_SOURCES.json','OFFLINE_RESULT.json','recorded_law_inputs.npz']
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(folder,optimized=True):
    folder=folder.resolve()
    args=[sys.executable]+(['-O'] if optimized else [])+[str(folder/'check_recorded_law.py'),'--verify']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='')
    p=subprocess.run(args,cwd=folder,env=env,capture_output=True,text=True,timeout=45)
    return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corruption',action='store_true');ap.add_argument('--checks-dir',type=Path);args=ap.parse_args()
    start=time.perf_counter();result={}
    manifest=HERE/'MANIFEST.json'
    if manifest.exists():
        for name,d in json.loads(manifest.read_text())['files'].items():
            require(sha(HERE/name)==d['sha256'],'Changed capsule file '+name)
    for optimized in (False,True):
        r=run(HERE,optimized);require(r['returncode']==0,r['stderr']);result['optimized' if optimized else 'normal']=r
    if args.corruption:
        cases=args.checks_dir or HERE/'verification_runs'/str(time.time_ns());cases.mkdir(parents=True,exist_ok=False)
        for case,expected in [('input_hash','Changed input'),('context_criterion','Contract interpretation'),('verdict_flag','Result mismatch'),('operand_with_updated_hash','Recorded margin identity')]:
            folder=cases/case;folder.mkdir()
            for name in FILES:shutil.copy2(HERE/name,folder/name)
            if case in ('input_hash','operand_with_updated_hash'):
                f=folder/'recorded_law_inputs.npz'
                with np.load(f,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
                arrays['sham_00901_01000ms_fields'][0,0,0,4]+=1
                np.savez_compressed(f,**arrays)
                if case=='operand_with_updated_hash':
                    p=folder/'OFFLINE_SOURCES.json';d=json.loads(p.read_text());d['input_sha256']=sha(f);p.write_text(json.dumps(d))
            elif case=='context_criterion':
                p=folder/'OFFLINE_PROTOCOL.json';d=json.loads(p.read_text());d['no_admission_of_stage4_or5']=False;p.write_text(json.dumps(d))
            else:
                p=folder/'OFFLINE_RESULT.json';d=json.loads(p.read_text());d['stage4_admission']=True;p.write_text(json.dumps(d))
            r=run(folder);require(r['returncode']!=0 and expected in r['stderr'],'Corruption not detected: '+case)
            result[case]=dict(detected=True,expected_error=expected,returncode=r['returncode'],files={n:sha(folder/n) for n in FILES})
    result.update(scope='Local extracted diagnostic only; no original source tree, CNS or historical MAT data loaded',
        wall_s=time.perf_counter()-start,neural_steps=0,body_steps=0)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
