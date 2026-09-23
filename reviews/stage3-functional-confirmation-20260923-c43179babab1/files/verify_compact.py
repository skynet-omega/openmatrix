"""Portable CPU readback of compact evidence, expressly not the full Stage3 verifier."""
from pathlib import Path
import hashlib, importlib.util, json, sys, time, resource
import numpy as np

HERE=Path(__file__).resolve().parent
NAMES={15:'etapa3_pn629_intervention_20260923_15',16:'etapa3_funcional_20260923_16',
       17:'etapa3_funcional_repair_20260923_17',19:'etapa3_continuation_repair_20260923_19',
       20:'etapa3_postclose_20260923_20'}
def need(ok,message):
    if not ok:raise ValueError(message)
def js(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))
def exact(a,b,label):need(canonical(a)==canonical(b),'Rebuilt value differs: '+label)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify(base=HERE):
    start=time.process_time();base=Path(base).resolve();inventory=js(base/'INVENTORY.json')
    seen=set()
    for row in inventory['files']:
        dest=Path(row['destination']);need(not dest.is_absolute() and '..' not in dest.parts,'Unsafe destination')
        need(str(dest) not in seen,'Duplicate inventory');seen.add(str(dest))
        p=base/dest
        need(not any(x.is_symlink() for x in (p,*p.parents)),'Symlink evidence')
        need(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Inventory identity: '+str(dest))
    if (base/'MANIFEST.json').is_file():
        for row in js(base/'MANIFEST.json')['files']:
            p=base/row['path'];need(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'OpenMatrix manifest identity')
    d={k:base/'AXIOMA_ASTRA/campanas'/name for k,name in NAMES.items()}
    runs={k:base/v for k,v in inventory['runs'].items()}
    post=js(d[20]/'POSTCLOSE_VERIFIED_01.json');final=js(d[19]/'FINAL_RAW_VERIFIED.json');raw=final['original_raw_result']
    for key,path in (('queue_sha256',d[19]/'QUEUE.json'),('final_raw_sha256',d[19]/'FINAL_RAW_VERIFIED.json'),
                     ('verifier_sha256',d[19]/'verify_repair19.py'),('contract_sha256',d[19]/'REPAIR19_CONTRACT.json')):
        need(post[key]==sha(path),'Postclose hash link: '+key)
    exact(post['rebuilt_raw_result'],final,'complete postclose result')
    need(js(d[19]/'QUEUE.json')['state']=='COMPLETE','Campaign not closed')
    need(post['functional_stage3_pass'] is True and post['full_result_exact'] is True,'Postclose failed')
    need(final['strict_original_contract_fulfilled'] is False and final['incomplete_attempts']==2,'Failures hidden')
    need(raw['historical_hidden_state_gate']=='FAIL_PRESERVADO' and raw['biological_equivalence'] is False
         and raw['source_navigation_demonstrated'] is False and raw['stage4_completed'] is False,'Scope broadened')
    for original,digest in post['source_hashes'].items():
        p=base/Path(original).relative_to('/home/daroch')
        need(p.is_file() and sha(p)==digest,'Postclose source closure missing or changed: '+str(p))
    sys.path.insert(0,str(d[16]))
    v=module('compact_original_scientific',d[16]/'verify_all.py')
    ext=module('compact_original_external',d[15]/'chatgpt_verificador_original.py')
    need(sha(d[16]/'PLAN.json')==v.PLAN_SHA and sha(d[15]/'chatgpt_verificador_original.py')==v.EXTERNAL_SHA,'Original contract/comparator')
    plan=js(d[16]/'PLAN.json');criteria=plan['criteria'];contract=js(d[19]/'REPAIR19_CONTRACT.json')
    pairs={};observations={};flows={};support={};hashes={};elapsed=0.
    op=base/'shared_operator';common=sha(op/'effective_operator.npz')
    need(len(inventory['common_operator_source_aliases'])==8,'Missing operator origins')
    for original,row in inventory['common_operator_source_aliases'].items():
        need(row['sha256']==common and post['read_input_hashes'][original]==row,'Common operator identity')
    for arm in ext.ARMS:
        a,b=runs['native_'+arm],runs['reference_'+arm]
        v.verify_sources(b)
        pair=[ext.load(a,arm,'causal_cuda',hashes),ext.load(b,arm,'reference_cuda',hashes)]
        observations[arm]=ext.compare(*pair);need(observations[arm]['observation_pair_ok'],'Observable pair failed')
        exact(observations[arm],raw['observable_pairs'][arm],'observable '+arm);pairs[arm]=pair
        flows[arm]={'native':v.raw_flow(a,op),'reference':v.raw_flow(b,op)}
        exact(flows[arm],raw['native_flow_reconstruction'][arm],'flow '+arm)
        support[arm]=v.support(b,criteria);exact(support[arm],raw['reference_support'][arm],'support '+arm)
        need(support[arm]['passed'],'Support failed')
        wall=js(b/'RESULT.json')['wall_total_s'];need(0<wall<=plan['budget']['wall_each_s_max'],'Reference budget');elapsed+=wall
    orientation=ext.orientation(pairs);exact(orientation,raw['orientation'],'orientation');need(orientation['both_directional'],'Orientation failed')
    mechanical=v.native_support(d[16],d[15],criteria);exact(mechanical,raw['native_support'],'mechanical replay')
    need(js(d[16]/'body_support_01/RESULT.json')['wall_s']<=plan['budget']['mechanical_wall_total_s_max'],'Replay budget')
    rem=v.remaining;withdrawals={}
    for arm in ('odor_left','odor_right'):
        folder=runs['withdrawal_'+arm];native=runs['native_'+arm];v.verify_sources(folder)
        a=rem.arrays(folder/'traces.npz');b=rem.arrays(native/'traces.npz')
        need(set(a)==set(b) and all(len(x)==440 for x in a.values()),'Withdrawal schema')
        need(all(np.array_equal(a[k][:40],b[k][:40]) for k in a),'Withdrawal observed preparation differs')
        r=js(folder/'RESULT.json');w=js(folder/'WITHDRAWAL.json')
        need(r['status']=='COMPLETE' and r['error'] is None and not r['cleanup_errors'],'Withdrawal incomplete')
        need(r['completed_trial_ms']==400 and r['completed_preparation_ms']==40 and r['engine']=='causal_cuda' and r['odor']==arm,'Withdrawal context')
        need(w['active'] and w['clamped_physical_calls']==16000 and w['prepared_physical_calls']==1600 and w['max_requested_yaw_rad_s']>0,'Withdrawal timing')
        need(0<r['wall_total_s']<=plan['budget']['wall_each_s_max'],'Withdrawal budget');elapsed+=r['wall_total_s']
        exact(js(folder/'FROZEN.json'),js(native/'FROZEN.json'),'withdrawal frozen sources')
        exact(js(folder/'INTERVENTION.json'),js(native/'INTERVENTION.json'),'withdrawal operator')
        need(js(folder/'EXPERIMENT_CONTRACT.json')['plan_sha256']==sha(d[16]/'PLAN.json'),'Withdrawal contract')
        t={k:x[40:] for k,x in a.items()};rem.clock(t,1,400)
        need(np.all(t['command_yaw_rate_rad_s']==0),'Angular withdrawal failed')
        forward=np.clip(.2+np.mean((t['DN_q_usada']-t['DN_baseline'])[:,:2],axis=1),0.,.5)
        need(np.array_equal(forward,t['command_forward_mm_s']),'Forward law changed')
        field=js(folder/'final_state/boundary.json');exact(field,js(native/'final_state/boundary.json'),'withdrawal field');rem.exposure(t,field)
        yaw=float(rem.pose_yaw(t,a['qpos'][39,3:7])[-1]);s=v.support(folder,criteria);need(s['passed'],'Withdrawal support')
        receipt=next(x for x in raw['coupled_withdrawals']['arms'] if x['arm']==arm)
        exact(yaw,receipt['yaw_400_deg'],'withdrawal yaw');exact(s,receipt['support'],'withdrawal support')
        need(sha(folder/'traces.npz')==receipt['traces_sha256'],'Withdrawal trace hash')
        flow=v.raw_flow(folder,op);exact(flow,raw['native_flow_reconstruction']['withdrawal_'+arm+'_01'],'withdrawal flow')
        withdrawals[arm]=yaw
    odd=(withdrawals['odor_left']-withdrawals['odor_right'])/2
    exact(odd,raw['coupled_withdrawals']['odd_yaw_deg'],'withdrawal odd')
    need(abs(odd)<=criteria['angular_reader_withdrawal_odd_yaw_deg_max'],'Withdrawal criterion')
    resume=runs['continuation'];source=runs['reference_odor_right'];a=rem.arrays(resume/'traces.npz');allref=rem.arrays(source/'traces.npz');b={k:x[140:] for k,x in allref.items()}
    need(set(a)==set(b) and all(len(x)==300 for x in a.values()),'Continuation schema');rem.clock(a,101,400)
    need(np.array_equal(a['CNS_time_ns'],b['CNS_time_ns']) and np.array_equal(a['DN_baseline'],b['DN_baseline']),'Continuation clock/baseline')
    need(np.array_equal(a['sensores_usados'],b['sensores_usados']),'Continuation exposure')
    field=js(resume/'final_state/boundary.json');exact(field,js(source/'final_state/boundary.json'),'Continuation field');rem.exposure(a,field)
    need(np.array_equal(ext.reader(a['DN_q_usada'],a['DN_baseline']),a['command_yaw_rate_rad_s']),'Continuation reader')
    rem.pose_yaw(a,allref['qpos'][39,3:7])
    yaw=float(np.max(np.abs(a['yaw_delta_deg']-b['yaw_delta_deg'])));cmd=float(np.rad2deg(np.sum(np.abs(a['command_yaw_rate_rad_s']-b['command_yaw_rate_rad_s']))*.001))
    exact(yaw,raw['continuation']['yaw_sup_deg'],'Continuation yaw');exact(cmd,raw['continuation']['command_L1_deg'],'Continuation command')
    need(yaw<=criteria['resume_yaw_sup_deg_max'] and cmd<=criteria['resume_command_L1_deg_max'],'Continuation criterion')
    exact({k:float(np.max(np.abs(a[k]-b[k]))) for k in a if a[k].dtype.kind=='f'},raw['continuation']['errors_by_trace'],'Continuation full observed float errors')
    exact({k:int(np.count_nonzero(a[k]!=b[k])) for k in a if a[k].dtype.kind in 'biu'},raw['continuation']['discrete_mismatches'],'Continuation discrete errors')
    exact(v.raw_flow(resume,op),raw['native_flow_reconstruction']['continuation'],'Continuation flow')
    current=js(resume/'RESULT.json');need(current['status']=='COMPLETE' and current['completed_continuation_ms']==300 and current['error'] is None and not current['cleanup_errors'],'Continuation incomplete')
    need(0<current['wall_total_s']<=contract['budget']['new_run_wall_s_max'],'Continuation budget');elapsed+=current['wall_total_s']
    failed=js(runs['incomplete_continuation']/'RESULT.json')
    need(failed['status']=='INCOMPLETE' and failed['completed_continuation_ms']==0 and failed['initial_exact'] is None,'Incomplete attempt hidden')
    need(failed['error']['message']=='World is not bound to this continuing body','Failure changed')
    need(not (runs['incomplete_uniform']/'RESULT.json').exists() and not (runs['incomplete_uniform']/'traces.npz').exists(),'Interrupted attempt falsely completed')
    exact(elapsed,raw['organism_wall_s'],'Complete-run budget')
    aggregate=elapsed+contract['prior_incomplete']['interrupted_uniform_wall_s']+failed['wall_total_s']
    exact(aggregate,final['aggregate_all_attempts_wall_s'],'All-attempt budget')
    need(elapsed<=plan['budget']['organism_wall_total_s_max'] and aggregate<=contract['budget']['aggregate_all_attempts_wall_s_max'],'Aggregate budget exceeded')
    posture=module('compact_repair_upright',d[17]/'verify_repair.py').upright
    rebuilt={}
    for arm in ext.ARMS:
        for kind in ('reference','native'):
            t=rem.arrays(runs[kind+'_'+arm]/'traces.npz');rebuilt[kind+'_'+arm]=posture(t['qpos'],t['upright'])
    for arm in ('odor_left','odor_right'):
        t=rem.arrays(runs['withdrawal_'+arm]/'traces.npz');rebuilt['withdrawal_'+arm]=posture(t['qpos'],t['upright'])
    rebuilt['continuation_right']=posture(a['qpos'],a['upright']);exact(rebuilt,final['upright_reconstructed_from_pose'],'upright from pose')
    return {'classification':'OBSERVABLES_REBUILT_AND_FULL_RESULT_RECEIPT_LINKED',
            'compact_observable_checks_pass':True,'new_stage3_admission_issued':False,
            'full_checkpoint_equivalence_recomputed':False,'full_organism_reproduced':False,
            'linked_campaign_classification':final['classification'],'historical_KC_gate':'FAIL_PRESERVADO',
            'complete_runs':7,'incomplete_attempts':2,'aggregate_wall_s':aggregate,
            'orientation':orientation,'withdrawal_odd_deg':odd,'continuation_yaw_sup_deg':yaw,'continuation_command_L1_deg':cmd,
            'cpu_s':time.process_time()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'limits':['Full session checkpoints exceed 100 MB and are absent; semantic preparation/restore equality is linked to the postclose receipt, not repeated.',
                      'This uses the archived comparator and numerical formulas; it is not an independent biological validation.',
                      'No organism, GPU, navigation, learning or Stage4 execution.']}

if __name__=='__main__':
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else HERE),indent=2,allow_nan=False))
