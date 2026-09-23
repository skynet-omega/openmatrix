"""Finish the frozen sequential PN629 experiment, stopping after an unfavorable right arm."""
from pathlib import Path
import os,sys,json,time,subprocess
import numpy as np
from close import stop_rule,require

HERE=Path(__file__).resolve().parent
PY='/home/daroch/miniconda3/envs/GPU/bin/python'
PARENT=HERE.parent/'etapa3_largo_diagnostico_20260923_10'
def read(p):return json.loads(p.read_text())
def valid(name):
    folder=HERE/name;r=read(folder/'RESULT.json')
    require(r['status']=='COMPLETE' and r['error'] is None and not r['cleanup_errors'],'Failed '+name)
    require(r['completed_trial_ms']==400 and r['completed_preparation_ms']==40,'Incomplete '+name)
    f=read(folder/'flow/RESULT.json')
    require(f['samples']==440 and f['max_target_error']<=1e-9 and f['max_rate_error']<=1e-12,'Flow failed '+name)
    require(read(folder/'FROZEN.json')==read(HERE/'full_sham_01/FROZEN.json'),'Frozen sources differ')
    return r
def yaw(folder):
    with np.load(folder/'traces.npz',allow_pickle=False) as z:
        i=np.flatnonzero((z['fase']=='ensayo')&(z['paso']==400))
        require(len(i)==1,'Missing terminal state')
        return float(z['yaw_delta_deg'][i[0]])

def main():
    receipt=HERE/'QUEUE.json';require(not receipt.exists(),'Queue already exists')
    plan=read(HERE/'PLAN.json');start=time.monotonic();runs=[]
    receipt.write_text(json.dumps({'state':'WAITING_FOR_EXISTING_SHAM','runs':runs})+'\n')
    while not (HERE/'full_sham_01/RESULT.json').is_file():
        if time.monotonic()-start>600:raise TimeoutError('Existing sham did not finish')
        time.sleep(5)
    valid('full_sham_01')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for odor in ('odor_right','odor_left','uniform'):
        name='full_'+odor+'_01';folder=HERE/name
        require(not folder.exists(),'Do not overwrite '+name)
        elapsed=sum(read(HERE/n/'RESULT.json')['wall_total_s'] for n in ('smoke_on_01','smoke_off_01','full_sham_01',*runs))
        require(elapsed+plan['budget']['wall_single_long_s_max']<=plan['budget']['aggregate_wall_s_max'],'Aggregate allocation exhausted')
        receipt.write_text(json.dumps({'state':'RUNNING','current':name,'completed':runs})+'\n')
        with (HERE/(name+'.log')).open('x') as log:
            process=subprocess.Popen([PY,'-B',str(HERE/'run_set.py'),'--out',str(folder),'--odor',odor,'--engine','causal_cuda','--ms','400','--observe','on'],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=process.wait(timeout=plan['budget']['wall_single_long_s_max'])
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
        require(code==0,'Runner failed '+name);valid(name);runs.append(name)
        # Validate exposure, command, native clocks, and shared prepared state
        # before deciding whether the two remaining allocated arms are useful.
        subprocess.run(['python3','-B',str(HERE/'analyze_long.py')],check=True,env=env)
        subprocess.run(['python3','-B',str(HERE/'compare_preparation.py')],check=True,env=env)
        d=read(HERE/'LONG_DIAGNOSTIC.json')
        require(all(v['equal'] for v in d['preparation_exact'].values()),'Prepared traces differ')
        if odor=='odor_right':
            parent=yaw(PARENT/'full_odor_right_01');child=yaw(folder)
            decision={'parent_right_deg':parent,'child_right_deg':child,'change_deg':child-parent,
                      'stop':stop_rule(parent,child,plan['material_effect_deg']),
                      'material_effect_deg':plan['material_effect_deg']}
            (HERE/'RIGHT_DECISION.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n')
            print(json.dumps(decision),flush=True)
            if decision['stop']:break
    subprocess.run(['python3','-B',str(HERE/'close.py')],check=True,env=env)
    receipt.write_text(json.dumps({'state':'COMPLETE','new_runs':runs,'wall_queue_s':time.monotonic()-start},indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        (HERE/'QUEUE_ERROR.json').write_text(json.dumps({'error':type(e).__name__,'message':str(e)},indent=2)+'\n')
        raise
