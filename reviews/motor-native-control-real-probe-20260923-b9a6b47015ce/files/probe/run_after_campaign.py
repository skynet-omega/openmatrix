"""Run the finite native probe only after the current long campaign releases GPU."""
from pathlib import Path
import json,os,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PN=ROOT/'campanas/etapa3_pn629_intervention_20260923_15'
PY='/home/daroch/miniconda3/envs/GPU/bin/python'
def main():
    status=HERE/'QUEUE.json'
    if status.exists():raise ValueError('Probe queue already exists')
    status.write_text(json.dumps({'state':'WAITING_FOR_PN_GPU_RELEASE'})+'\n')
    start=time.monotonic()
    while True:
        if (PN/'QUEUE_ERROR.json').exists():raise RuntimeError('PN queue failed; inspect before using GPU')
        if json.loads((PN/'QUEUE.json').read_text())['state']=='COMPLETE':break
        if time.monotonic()-start>7200:raise TimeoutError('Existing campaign wait budget')
        time.sleep(10)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    elapsed=0.
    for mode,name in ((-1,'parent_01'),(0,'mode0_01'),(1,'mode1_01')):
        if elapsed+240>720:raise TimeoutError('Unallocated probe budget')
        status.write_text(json.dumps({'state':'RUNNING','current':name})+'\n')
        with (HERE/(name+'.log')).open('x') as log:
            child=subprocess.Popen([PY,'-B',str(HERE/'probe_runner.py'),str(mode),str(HERE/name)],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=240)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=25)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        if code:raise RuntimeError('Native probe run failed '+name)
        r=json.loads((HERE/name/'RESULT.json').read_text());elapsed+=r['wall_total_s']
        if r['status']!='COMPLETE':raise RuntimeError('Incomplete '+name)
        if mode>=0:
            p=json.loads((HERE/name/'SONDA.json').read_text())
            if p['overflow'] or p['timing_query_failed'] or not all(x['done'] for x in p['rows']):raise RuntimeError('Invalid native probe '+name)
    subprocess.run(['python3','-B',str(HERE/'analyze.py')],check=True,env=env)
    status.write_text(json.dumps({'state':'COMPLETE','wall_runner_s':elapsed},indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        (HERE/'QUEUE_ERROR.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)},indent=2)+'\n')
        raise
