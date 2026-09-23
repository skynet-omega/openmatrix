"""Finite repair of one externally interrupted reference; all original criteria retained."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'etapa3_funcional_20260923_16'
NATIVE=HERE.parent/'etapa3_pn629_intervention_20260923_15'
PY='/home/daroch/miniconda3/envs/GPU/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def need(ok,message):
 if not ok:raise ValueError(message)
def guard():
 for path,digest in js(HERE/'REPAIR_SOURCES.json').items():need(sha(path)==digest,'Repair source changed: '+path)
def main():
 status=HERE/'REPAIR_QUEUE.json';need(not status.exists(),'Repair queue already exists')
 guard();contract=js(HERE/'REPAIR_CONTRACT.json');budget=contract['budget'];original=js(OLD/'INTERRUPTION.json')
 need(sha(HERE/'PLAN.json')==contract['original_contract_sha256'],'Original frozen criteria differ')
 need(sha(OLD/'QUEUE_AT_INTERRUPTION.json')==original['queue_sha256_before_edit'],'Interrupted queue receipt differs')
 need(sha(OLD/'reference_uniform_01/PROGRESS.jsonl')==original['uniform_progress_sha256'],'Interrupted uniform trace differs')
 need(sha(OLD/'reference_uniform_01/prepared_state/MANIFEST.json')==original['uniform_prepared_manifest_sha256'],'Interrupted preparation differs')
 need(js(OLD/'QUEUE.json')['state']=='INTERRUPTED_EXTERNAL_PROCESS','Old campaign not closed')
 prior={};used=float(original['partial_last_elapsed_s']);attempts=1
 for arm in ('odor_right','sham','odor_left'):
  name='reference_'+arm+'_01';folder=HERE/name;need(folder.is_symlink() and folder.resolve()==(OLD/name).resolve(),'Prior reference link changed')
  r=js(folder/'RESULT.json');need(r['status']=='COMPLETE' and r['error'] is None and not r['cleanup_errors'] and r['odor']==arm and r['engine']=='reference_cuda' and r['completed_trial_ms']==400 and r['completed_preparation_ms']==40,'Prior reference incomplete')
  need(r['wall_total_s']<=budget['wall_each_complete_s_max'],'Prior reference wall limit')
  used+=r['wall_total_s'];prior[name]={'wall_s':r['wall_total_s'],'result_sha256':sha(folder/'RESULT.json')};attempts+=1
 need(abs(sum(v['wall_s'] for v in prior.values())-original['completed_wall_s'])<1e-6,'Prior consumed budget mismatch')
 need(used<budget['aggregate_wall_including_partial_s_max'],'No aggregate time remains')
 save(status,{'state':'READY','prior':prior,'interrupted_partial_wall_s':original['partial_last_elapsed_s'],'wall_consumed_s':used,'attempts_started':attempts,'stage3_admission':False})
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 jobs=[('reference_uniform_01',OLD/'run_reference.py',['uniform',str(HERE/'reference_uniform_01')]),
       ('withdrawal_odor_left_01',OLD/'run_withdrawal.py',['odor_left',str(HERE/'withdrawal_odor_left_01')]),
       ('withdrawal_odor_right_01',OLD/'run_withdrawal.py',['odor_right',str(HERE/'withdrawal_odor_right_01')]),
       ('continuation_right_01',OLD/'run_continuation.py',[str(HERE/'continuation_right_01')])]
 completed=[]
 for name,script,args in jobs:
  guard();need(used<budget['aggregate_wall_including_partial_s_max'],'Aggregate budget exhausted')
  out=HERE/name;need(not out.exists(),'Attempt already exists: '+name)
  attempts+=1;need(attempts<=budget['total_attempts_max'],'Attempt budget exhausted')
  save(status,{'state':'RUNNING','current':name,'prior':prior,'completed_repair':completed,'interrupted_partial_wall_s':original['partial_last_elapsed_s'],'wall_consumed_s':used,'attempts_started':attempts,'stage3_admission':False})
  with (HERE/(name+'.log')).open('x') as log:
   child=subprocess.Popen([PY,'-B',str(script),*args],env=env,stdout=log,stderr=subprocess.STDOUT)
   try:code=child.wait(timeout=budget['wall_each_complete_s_max'])
   except subprocess.TimeoutExpired:
    child.terminate()
    try:child.wait(timeout=25)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    raise
  need(code==0,'Organism run failed: '+name)
  result=js(out/'RESULT.json');need(result['status']=='COMPLETE' and result['error'] is None and not result['cleanup_errors'],'Incomplete organism result: '+name)
  need(result['wall_total_s']<=budget['wall_each_complete_s_max'],'Per-run wall ceiling: '+name)
  used+=result['wall_total_s'];need(used<=budget['aggregate_wall_including_partial_s_max'],'Aggregate wall ceiling')
  if name=='reference_uniform_01':
   input_plan={'schema':'orientation_functional_v1','registered_utc':js(HERE/'PLAN.json')['registered_utc'],'pairs':{arm:{'causal':str(NATIVE/('full_'+arm+'_01')),'reference':str(HERE/('reference_'+arm+'_01'))} for arm in ('sham','odor_left','odor_right','uniform')}}
   save(HERE/'pairs_4_repair.json',input_plan)
   with (HERE/'comparison_4_repair.log').open('x') as log:
    x=subprocess.run([sys.executable,'-B',str(NATIVE/'chatgpt_verificador_original.py'),'--plan',str(HERE/'pairs_4_repair.json'),'--out',str(HERE/'comparison_4_01')],stdout=log,stderr=subprocess.STDOUT,env=env)
   need(x.returncode==0,'Four-arm observable comparison failed')
   x=subprocess.run([sys.executable,'-B',str(OLD/'check_support.py'),str(out)],env=env)
   need(x.returncode==0,'Uniform support failed')
   report=js(HERE/'comparison_4_01/RESULTADO.json')
   need(report['observations_concordant'] and report['orientation']['both_directional'],'Directional requirement failed')
  else:
   # The preserved checker writes to its own HERE when run as a CLI. Call it
   # with only its destination rebound; all scientific formulas are unchanged.
   sys.path.insert(0,str(OLD));import check_remaining as check
   before=(check.HERE,check.NATIVE);check.HERE,check.NATIVE=HERE,NATIVE
   try:
    if name.startswith('withdrawal_'):check.withdrawal(out,name.removeprefix('withdrawal_').removesuffix('_01'))
    else:check.continuation(out)
    if name=='withdrawal_odor_right_01':check.withdrawal_pair(HERE/'withdrawal_odor_left_01',out)
   finally:check.HERE,check.NATIVE=before
  completed.append(name)
  save(status,{'state':'REPAIR_PROGRESS','completed_repair':completed,'wall_consumed_s':used,'attempts_started':attempts,'stage3_admission':False})
 guard();need(len(completed)==4 and attempts==8,'Repair sequence incomplete')
 save(status,{'state':'EXPERIMENTS_COMPLETE_REQUIRES_RAW_VERIFICATION','completed_repair':completed,'prior':prior,'wall_consumed_s':used,'attempts_started':attempts,'stage3_admission':False})
 from verify_repair import verify
 final=verify();save(HERE/'FINAL_REPAIR_VERIFIED.json',final)
 save(status,{'state':'COMPLETE' if final['functional_stage3_pass'] else 'VERIFICATION_FAILED','completed_repair':completed,'wall_consumed_s':used,'attempts_started':attempts,'stage3_admission':final['functional_stage3_pass'],'receipt':'FINAL_REPAIR_VERIFIED.json'})
 return 0 if final['functional_stage3_pass'] else 2
if __name__=='__main__':
 try:code=main()
 except BaseException as e:
  save(HERE/'REPAIR_ERROR.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'stage3_admission':False})
  raise
 raise SystemExit(code)
