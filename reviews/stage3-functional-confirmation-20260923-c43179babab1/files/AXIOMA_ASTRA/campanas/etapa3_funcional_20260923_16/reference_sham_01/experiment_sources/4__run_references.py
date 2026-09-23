"""Four prospective references, stopping at the first failed observable/support gate."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NATIVE=HERE.parent/'etapa3_pn629_intervention_20260923_15'
PROBE=ROOT/'motor_nuevo/native_stream_cost_20260923'
PY='/home/daroch/miniconda3/envs/GPU/bin/python'
def dump(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
    plan=json.loads((HERE/'PLAN.json').read_text());status=HERE/'QUEUE.json'
    if status.exists():raise FileExistsError(status)
    dump(status,{'state':'WAITING_FOR_NATIVE_PROBE'})
    start=time.monotonic()
    while True:
        if (PROBE/'QUEUE_ERROR.json').exists():raise RuntimeError('Inspect failed native probe before releasing GPU')
        if (PROBE/'QUEUE.json').is_file() and json.loads((PROBE/'QUEUE.json').read_text())['state']=='COMPLETE':break
        if time.monotonic()-start>3600:raise TimeoutError('Previous campaign wait budget')
        time.sleep(10)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    pairs={};used=0.;completed=[]
    for arm in ('odor_right','sham','odor_left','uniform'):
        name='reference_'+arm+'_01';out=HERE/name
        dump(status,{'state':'RUNNING','current':name,'completed':completed,'wall_consumed_s':used})
        with (HERE/(name+'.log')).open('x') as log:
            child=subprocess.Popen([PY,'-B',str(HERE/'run_reference.py'),arm,str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=plan['budget']['wall_each_s_max'])
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=25)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        if code:raise RuntimeError('Reference execution failed: '+name)
        r=json.loads((out/'RESULT.json').read_text());used+=r['wall_total_s']
        if r['status']!='COMPLETE' or used>plan['budget']['organism_wall_total_s_max']:raise RuntimeError('Reference completion/budget')
        pairs[arm]={'causal':str(NATIVE/('full_'+arm+'_01')),'reference':str(out)}
        index=len(pairs);input_plan={'schema':'orientation_functional_v1','registered_utc':plan['registered_utc'],'pairs':dict(pairs)}
        pairfile=HERE/('pairs_'+str(index)+'.json');dump(pairfile,input_plan)
        with (HERE/('comparison_'+str(index)+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,'-B',str(NATIVE/'chatgpt_verificador_original.py'),'--plan',str(pairfile),'--out',str(HERE/('comparison_'+str(index)+'_01'))],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Prospective observable comparison failed: '+arm)
        subprocess.run([sys.executable,'-B',str(HERE/'check_support.py'),str(out)],check=True,env=env)
        comparison=json.loads((HERE/('comparison_'+str(index)+'_01')/'RESULTADO.json').read_text())
        if index==4 and not comparison['orientation']['both_directional']:raise RuntimeError('Four-arm directional margins failed')
        completed.append(name)
    dump(status,{'state':'REFERENCES_COMPLETE','completed':completed,'wall_consumed_s':used,'stage3_admission':False,
                 'remaining':['coupled reader withdrawals','same-profile continuation','native mechanical support replay']})
if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(HERE/'QUEUE_ERROR.json',{'type':type(e).__name__,'message':str(e),'stage3_admission':False})
        raise
