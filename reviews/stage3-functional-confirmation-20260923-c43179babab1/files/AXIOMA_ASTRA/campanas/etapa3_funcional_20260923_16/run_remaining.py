"""Finish the existing finite contract after all four references pass; fail closed."""
from pathlib import Path
import json,os,subprocess,sys,time
from checkpoint_compare import sha

HERE=Path(__file__).resolve().parent;PY='/home/daroch/miniconda3/envs/GPU/bin/python'
def js(p):return json.loads(p.read_text())
def save(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def require(v,m):
    if not v:raise ValueError(m)
def main():
    status=HERE/'REMAINING_QUEUE.json';require(not status.exists(),'Continuation queue already exists')
    plan=js(HERE/'PLAN.json');frozen=js(HERE/'REMAINING_SOURCES.json')
    for name,digest in frozen.items():require(sha(name)==digest,'Execution source changed: '+name)
    save(status,{'state':'WAITING_FOR_FOUR_REFERENCES','stage3_admission':False})
    start=time.monotonic()
    while True:
        require(not (HERE/'QUEUE_ERROR.json').exists(),'Reference failed; confirmation stopped')
        require(not (HERE/'BODY_SUPPORT_FAILURE.json').exists(),'Native mechanical replay failed')
        q=js(HERE/'QUEUE.json')
        if q['state']=='REFERENCES_COMPLETE':break
        require(time.monotonic()-start<=7200,'Reference wait ceiling; do not start further runs')
        time.sleep(10)
    # Raw-data checks and a frozen external comparator are used by the final
    # verifier; here these fresh execution receipts control only dispatch.
    require(set(q['completed'])=={'reference_'+x+'_01' for x in ('sham','uniform','odor_left','odor_right')},'Missing reference')
    comparison=js(HERE/'comparison_4_01/RESULTADO.json')
    require(comparison['observations_concordant'] and comparison['orientation']['both_directional'],'Reference/functional gate failed')
    body=js(HERE/'body_support_01/RESULT.json')
    require(body['complete'] and len(body['cases'])==4 and all(x['passed'] for x in body['cases'].values()),'Native mechanical support incomplete')
    require(body['wall_s']<=plan['budget']['mechanical_wall_total_s_max'],'Mechanical replay budget')
    used=q['wall_consumed_s'];completed=[]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    jobs=[('withdrawal_odor_left_01','run_withdrawal.py',['odor_left']),
          ('withdrawal_odor_right_01','run_withdrawal.py',['odor_right']),
          ('continuation_right_01','run_continuation.py',[])]
    for name,script,args in jobs:
        for source,digest in frozen.items():require(sha(source)==digest,'Source changed while waiting: '+source)
        require(used<plan['budget']['organism_wall_total_s_max'],'Aggregate organism budget exhausted')
        out=HERE/name;save(status,{'state':'RUNNING','current':name,'completed':completed,'organism_wall_consumed_s':used,'stage3_admission':False})
        with (HERE/(name+'.log')).open('x') as log:
            child=subprocess.Popen([PY,'-B',str(HERE/script),*args,str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=plan['budget']['wall_each_s_max'])
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=25)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        if (out/'RESULT.json').exists():used+=js(out/'RESULT.json')['wall_total_s']
        require(code==0,'Organism run failed: '+name)
        require(used<=plan['budget']['organism_wall_total_s_max'],'Aggregate organism budget')
        check=[sys.executable,'-B',str(HERE/'check_remaining.py')]
        if script=='run_withdrawal.py':check+=['withdrawal',str(out),args[0]]
        else:check+=['continuation',str(out)]
        subprocess.run(check,check=True,env=env)
        completed.append(name)
        if name=='withdrawal_odor_right_01':
            subprocess.run([sys.executable,'-B',str(HERE/'check_remaining.py'),'pair',str(HERE/'withdrawal_odor_left_01'),str(HERE/'withdrawal_odor_right_01')],check=True,env=env)
    save(status,{'state':'EXPERIMENTS_COMPLETE_REQUIRES_RAW_DATA_VERIFICATION','completed':completed,
        'organism_wall_consumed_s':used,'stage3_admission':False})
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(HERE/'REMAINING_QUEUE_ERROR.json',{'type':type(e).__name__,'message':str(e),'stage3_admission':False})
        raise
