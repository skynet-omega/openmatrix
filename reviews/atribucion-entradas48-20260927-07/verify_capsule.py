"""Reconstruct the diagnostic, rather than trusting its recorded verdict flags."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from analyze_endpoint import calculate,report,need,sha,write
HERE=Path(__file__).resolve().parent
CORE=['analyze_endpoint.py','verify_capsule.py','PLAN.json','BOUND_PROTOCOL.json','EXTRACTION.json','endpoint_inputs.npz','RESULTADOS.json','RESULTADOS.md']
def verify():
    if (HERE/'MANIFEST.json').exists():
        m=json.loads((HERE/'MANIFEST.json').read_text())
        for name,d in m['files'].items():need(sha(HERE/name)==d['sha256'],'Capsule hash '+name)
    computed=calculate();saved=json.loads((HERE/'RESULTADOS.json').read_text())
    need(computed==saved,'Recomputed result differs')
    need(report(computed)==(HERE/'RESULTADOS.md').read_text(),'Report differs')
    for a,rows in computed['rows'].items():
        for r in rows[:2]:need(r['endpoint_minus_last_recorded_RHS']==0,'Final DNg net differs')
    return dict(recomputed=True,exact_DNg_net_matches=8,admission_stage4=False,admission_stage5=False)
def corruption_tests(folder):
    folder=folder.resolve();folder.mkdir(parents=True,exist_ok=False);results={}
    for case,expected in [('input_hash','Changed inputs'),('changed_context','Boundary/context changed'),('verdict_flag','Recomputed result differs'),('altered_operand_with_hash','Recomputed result differs'),('changed_budget','Budget/scope changed')]:
        d=folder/case;d.mkdir()
        for f in CORE:shutil.copy2(HERE/f,d/f)
        if case in ['input_hash','altered_operand_with_hash']:
            f=d/'endpoint_inputs.npz'
            with np.load(f,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            # An actual high-weight source, safely inside finite numerical range.
            k=int(np.argmax(np.abs(v['sham_cuda_weights']*v['caps']*v['sham_transmission'])))
            old=float(v['sham_transmission'][k]);v['sham_transmission'][k]*=.5
            need(float(v['sham_transmission'][k])!=old,'Corruption must actually change the operand')
            np.savez_compressed(f,**v)
            if case=='altered_operand_with_hash':
                p=d/'EXTRACTION.json';e=json.loads(p.read_text());e['inputs_sha256']=sha(f);write(p,e)
        elif case=='changed_context':
            p=d/'EXTRACTION.json';e=json.loads(p.read_text());e['metadata']['sham']['time_ns']+=1;write(p,e)
        elif case=='verdict_flag':
            p=d/'RESULTADOS.json';e=json.loads(p.read_text());e['stage4_pass']=True;write(p,e)
        else:
            p=d/'PLAN.json';e=json.loads(p.read_text());e['new_neural_steps_max']=1;write(p,e)
            p=d/'EXTRACTION.json';e=json.loads(p.read_text());e['plan_sha256']=sha(d/'PLAN.json');write(p,e)
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
        proc=subprocess.run([sys.executable,'-O',str(d/'verify_capsule.py')],cwd=d,env=env,capture_output=True,text=True,timeout=20)
        need(proc.returncode!=0 and expected in proc.stderr,'Corruption test failed: '+case+' '+proc.stderr)
        results[case]={'detected':True,'expected_error':expected,'returncode':proc.returncode}
    return results
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corruption-dir',type=Path);args=ap.parse_args()
    t=time.process_time();w=time.monotonic();r=verify()
    if args.corruption_dir:r['deliberate_corruptions']=corruption_tests(args.corruption_dir)
    r.update(CPU_s=time.process_time()-t,wall_s=time.monotonic()-w,optimized=not __debug__,scope='Extracted endpoint diagnostic, not CNS replay')
    print(json.dumps(r,indent=2))
