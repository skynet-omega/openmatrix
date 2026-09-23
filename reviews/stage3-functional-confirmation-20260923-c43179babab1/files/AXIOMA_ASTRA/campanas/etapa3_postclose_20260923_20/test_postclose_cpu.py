"""Synthetic inputs only: never import verify_repair19 or read organism outcomes."""
import copy,hashlib,importlib.util,json,resource,sys,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
START=time.process_time();HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('postclose_under_test',HERE/'verify_postclose.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
TAG=sys.argv[1] if len(sys.argv)==2 else '02'
if not TAG.replace('_','').isalnum():raise ValueError('Invalid fixture tag')
BASE=HERE/(('fixtures_optimized_' if not __debug__ else 'fixtures_normal_')+TAG);BASE.mkdir(exist_ok=False)
def write(path,value):Path(path).write_text(json.dumps(value,allow_nan=False)+'\n')
raw={'schema':'SYNTHETIC_FIXTURE_DO_NOT_USE','classification':v.CONFIRMED,'functional_stage3_pass':True,
     'strict_original_contract_fulfilled':False,'attempts_total':9,
     'original_raw_result':{'classification':v.CONFIRMED,'functional_stage3_pass':True,'value':.123,'nested':[True,2,{'x':-0.0}]}}
tests=[]
def case(name,queue='COMPLETE',stored=None):
    p=BASE/name;p.mkdir();write(p/'QUEUE.json',{'state':queue});write(p/'FINAL_RAW_VERIFIED.json',raw if stored is None else stored);write(p/'REPAIR19_CONTRACT.json',{'synthetic':True});(p/'verify_repair19.py').write_text('# SYNTHETIC FIXTURE ONLY\n');return p

def reject(name,fn,contains=None):
    try:fn()
    except Exception as exc:
        if contains is not None and contains not in str(exc):raise RuntimeError((name,str(exc)))
        tests.append({'name':name,'rejected':True,'type':type(exc).__name__})
    else:raise RuntimeError('Corruption accepted: '+name)

calls=[]
p=case('running','RUNNING')
reject('premature_no_verifier_import',lambda:v.rebuild(p,lambda:calls.append('import')),'COMPLETE')
if calls:raise RuntimeError('Premature loader called')
p=case('failed','FAILED');reject('failed_queue',lambda:v.rebuild(p,lambda:lambda:raw),'COMPLETE')
p=case('success');fresh=v.rebuild(p,lambda:lambda:copy.deepcopy(raw))
if v.canonical(fresh)!=v.canonical(raw):raise RuntimeError('Exact result rejected')
tests.append({'name':'exact_full_result','passed':True})
p=case('forged_stored_pass')
def failure():raise ValueError('Synthetic raw measurement failed')
reject('stored_true_cannot_override_raw_exception',lambda:v.rebuild(p,lambda:failure),'measurement failed')
for name,path,value in [('fresh_false',('functional_stage3_pass',),False),('nested_false',('original_raw_result','functional_stage3_pass'),False),('float_change',('original_raw_result','value'),.124),('bool_to_int',('original_raw_result','nested'),[1,2,{'x':-0.0}]),('missing_key',('original_raw_result',),{'functional_stage3_pass':True,'classification':v.CONFIRMED}),('nonfinite',('original_raw_result','value'),float('nan'))]:
    p=case(name);changed=copy.deepcopy(raw);target=changed
    for key in path[:-1]:target=target[key]
    target[path[-1]]=value
    reject(name,lambda p=p,changed=changed:v.rebuild(p,lambda:lambda:changed))
p=case('queue_changed')
def changing():write(p/'QUEUE.json',{'state':'FAILED'});return copy.deepcopy(raw)
reject('queue_changes_during_check',lambda:v.rebuild(p,lambda:changing),'COMPLETE')
p=case('duplicate_json');(p/'FINAL_RAW_VERIFIED.json').write_text('{"functional_stage3_pass":false,"functional_stage3_pass":true}')
reject('duplicate_json_key',lambda:v.rebuild(p,lambda:lambda:raw),'Duplicate')
p=case('nan_json');(p/'FINAL_RAW_VERIFIED.json').write_text('{"x":NaN}')
reject('nonfinite_stored_json',lambda:v.rebuild(p,lambda:lambda:raw),'Nonfinite')
# Real audit hooks with only tiny local files and no imported model.
p=BASE/'audit';p.mkdir();allowed=p/'allowed';allowed.mkdir();data=p/'data.txt';data.write_text('before')
audit=v.ReadOnlyInputs(allowed,(BASE,));sys.addaudithook(audit);audit.active=True
if data.read_text()!='before':raise RuntimeError('Fixture read changed')
reject('source_write_blocked',lambda:data.write_text('illegal'),'outside write')
reject('GPU_import_blocked',lambda:audit('import',('cupy',None)),'forbidden')
reject('GPU_library_blocked',lambda:audit('ctypes.dlopen',('libcuda.so',)),'forbidden')
audit.active=False;data.write_text('after');reject('input_changed_after_read',audit.finish,'input changed')
if str(data.resolve()) not in audit.records:raise RuntimeError('Read provenance missing')
# Actual main transaction with fake verifier and synthetic roots; preserves real receipt path unused.
real_here,real_loader,argv=v.HERE,v.load_actual,sys.argv
for name,good in [('transaction_success',True),('transaction_failure',False)]:
    p=case(name);out=p/'output';out.mkdir()
    plan={'repair_directory':str(p),'receipt':'SYNTHETIC_RECEIPT.json','budget':{'future_cpu_seconds_max':10,'future_wall_seconds_max':20,'future_address_space_gib_max':1}}
    write(out/'PLAN.json',plan);write(out/'SOURCE_LOCK.json',{str(real_here/'verify_postclose.py'):v.sha(real_here/'verify_postclose.py')})
    v.HERE=out;sys.argv=['verify_postclose.py']
    v.load_actual=lambda ignored,good=good:(lambda:copy.deepcopy(raw)) if good else failure
    code=v.main();receipt=json.loads((out/'SYNTHETIC_RECEIPT.json').read_text())
    if (code==0)!=good or receipt['functional_stage3_pass'] is not good:raise RuntimeError('Wrong synthetic transaction verdict')
    if good and v.canonical(receipt['rebuilt_raw_result'])!=v.canonical(raw):raise RuntimeError('Full reconstructed result omitted')
    if good:
        for key,name in [('queue_sha256','QUEUE.json'),('final_raw_sha256','FINAL_RAW_VERIFIED.json'),('verifier_sha256','verify_repair19.py'),('contract_sha256','REPAIR19_CONTRACT.json')]:
            if receipt[key]!=v.sha(p/name):raise RuntimeError('Gate binding hash missing/wrong '+key)
    reject(name+'_cannot_overwrite',v.main)
    tests.append({'name':name,'passed':True})
v.HERE=real_here;v.load_actual=real_loader;sys.argv=argv
if (real_here/'POSTCLOSE_VERIFIED_01.json').exists():raise RuntimeError('A real postclose receipt was created by synthetic tests')
result={'status':'PASS_SYNTHETIC_ONLY','optimized':not __debug__,'tests':tests,'count':len(tests),'gpu_calls':0,'organism_loads':0,'real_recomputations':0,'real_receipt_exists':False,'cpu_s':time.process_time()-START,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'source_sha256':v.sha(real_here/'verify_postclose.py')}
path=HERE/(('TEST_OPTIMIZED_' if not __debug__ else 'TEST_NORMAL_')+TAG+'.json')
with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'tests':len(tests),'cpu_s':result['cpu_s'],'real_recomputations':0}))
