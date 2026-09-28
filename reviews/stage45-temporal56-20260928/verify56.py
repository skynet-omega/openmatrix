"""Reconstruct terminal-identity/duration effects from retained observations."""
from pathlib import Path
import json,hashlib,argparse,copy,time,math
import numpy as np
from trace_contract56 import need,finite,trace_check
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def equal(a,b,label):
 need(set(a)==set(b),'keys '+label)
 for k in a:need(np.array_equal(a[k],b[k]),'array '+label+'/'+k)
def load(root):
 root=Path(root);repair=root/'repair01';d={'arms':{}}
 for name,folder in [('original',root),('repair',repair)]:
  f=json.loads((folder/'A_FREEZE.json').read_text());need(sha(folder/'A_PLAN.json')==f['contract_sha256'],'plan hash '+name);need(sha(folder/'A_SOURCES.json')==f['sources_sha256'],'source inventory '+name)
  d[name+'_plan']=json.loads((folder/'A_PLAN.json').read_text());d[name+'_queue']=json.loads((folder/'QUEUE_RESULT.json').read_text())
 for n,digest in json.loads((root/'reference/REFERENCE_HASHES.json').read_text()).items():need(sha(root/'reference'/n)==digest,'reference '+n)
 d['terminals']=arrays(root/'reference/TERMINALS.npz');d['initial']=arrays(root/'reference/INITIAL_STATES.npz');d['failure']=json.loads((root/'FAILURE_ACCOUNTING.json').read_text());d['original_failure_log']=(root/'qual_disabled.log').read_text();d['partial_result']=(root/'qual_disabled/RESULT.json.tmp').read_text()
 d['reference']={part:arrays(root/'reference/qualification55'/filename) for part,filename in [('trace','traces.npz'),('neural','neural_and_inputs.npz'),('consumed','PN_consumed.npz'),('panel','wide_observation.npz'),('observer','dng100_observed.npz')]}
 for part,filename in [('owners','SCIENTIFIC_WITNESS.json'),('events','EVENTS.json')]:d['reference'][part]=json.loads((root/'reference/qualification55'/filename).read_text())
 for name in d['repair_plan']['arms']:
  folder=repair/name;a={part:arrays(folder/filename) for part,filename in [('trace','traces.npz'),('neural','neural_and_inputs.npz'),('consumed','PN_consumed.npz'),('native','PN_native.npz'),('panel','wide_observation.npz'),('observer','dng100_observed.npz'),('intervention','INTERVENTION.npz'),('ORN','ORN_AUDIT.npz')]}
  a.update({part:json.loads((folder/filename).read_text()) for part,filename in [('result','RESULT.json'),('owners','SCIENTIFIC_WITNESS.json'),('events','EVENTS.json'),('metadata','INTERVENTION.json')]});d['arms'][name]=a
 return d
def last_targets(observer,n):
 z=observer;r=z['records'];accepted=r[:,138].astype(bool);latest=np.maximum.accumulate(np.where(accepted,np.arange(len(r)),-1));end=z['start_ns']+z['duration_ns'];epochs=np.flatnonzero(z['committed'] & ((end-47486000000)%1000000==0));need(len(epochs)==n,'endpoint target count');indices=latest[z['offsets'][epochs+1]-1];need(np.all(indices>=z['offsets'][epochs]),'endpoint accepted')
 return r[indices,:128].reshape(-1,4,2,16)[:,3,:,11]
def check_arm(a,spec,terminals,initial,name):
 n=spec['duration_ms'];out=trace_check(a,n,name)
 expected_mode=np.full((n,1),2 if spec['mode']=='identity' else 0,dtype=np.int32)
 if spec['mode']=='hold':expected_mode[:]=1
 if spec['mode']=='pulse':expected_mode[0]=1
 c,v=a['consumed'],a['native'];finite(c,name+' consumed');finite(v,name+' native');patch=a['intervention'];finite(patch,name+' patch')
 need(np.array_equal(v['mode'],expected_mode) and not np.any(v['errors']),'writer mode/errors '+name)
 need(np.array_equal(patch['pn_rows'],terminals['rows']) and np.array_equal(patch['pn_ids'],terminals['ids']),'terminal support')
 need(np.array_equal(patch['fixed_FP32'],terminals[spec['donor']]),'fixed terminal identity')
 need(np.array_equal(patch['initial_state'],initial[spec['receiver']]),'initial neural state changed')
 need(a['metadata']['spec']==spec,'spec metadata')
 for k in ['first','last','lo','hi','counts']:
  need(c[k].shape==v[k].shape==(n,686),'witness shape '+k)
 need(np.array_equal(c['counts'],v['counts']) and np.all(c['counts']>0) and np.all(c['counts']==c['counts'][:,:1]),'every RHS coverage')
 for witness in [c,v]:
  need(np.all(witness['lo']<=witness['first']) and np.all(witness['first']<=witness['hi']) and np.all(witness['lo']<=witness['last']) and np.all(witness['last']<=witness['hi']),'witness bounds')
  need(np.all(witness['lo']>=0) and np.all(witness['hi']<=1),'release bounds')
 active=expected_mode[:,0]==1
 for k in ['first','last','lo','hi']:
  need(np.array_equal(c[k][active],np.broadcast_to(patch['fixed_FP32'],(active.sum(),686))),'constant source consumed '+name+'/'+k)
  need(np.array_equal(c[k][~active],v[k][~active]),'transparent source consumed '+name+'/'+k)
 audit=a['ORN']['counts_and_errors'];need(audit.shape==(n,2) and np.all(audit[:,0]>0) and not np.any(audit[:,1]),'ORN every RHS audit')
 need(np.array_equal(a['neural']['PN_ids'],terminals['ids']) and a['neural']['PN_q'].shape==(n,686),'PN observation support')
 need(np.array_equal(a['neural']['PN_q'][-1],a['neural']['final_q'][terminals['rows']]),'final PN observation cross-check')
 need(np.array_equal(a['panel']['q'][-1],a['neural']['final_q'][a['panel']['rows']]) and np.array_equal(a['panel']['time_ns'],a['trace']['CNS_time_ns']),'final panel/time identity')
 need(np.array_equal(a['neural']['initial_q'][terminals['rows']],initial[spec['receiver']][terminals['rows']]),'initial PN q')
 o=a['observer']['records'][:,:128].reshape(-1,4,2,16);net=o[:,:,:,1].astype(np.float32);drive=o[:,:,:,4].astype(np.float32);theta=o[:,:,:,5].astype(np.float32);margin=(net+drive)-theta
 flags=a['observer']['records'][:,138];need(np.array_equal(flags,flags.astype(bool).astype(flags.dtype)),'accepted flag must be exactly binary')
 need(np.array_equal(margin.astype(np.float64),o[:,:,:,10]),'DNg margin arithmetic')
 need(np.array_equal(o[:,:,:,6],o[:,:,:,6].astype(np.float32).astype(np.float64)) and np.all(o[:,:,:,6]>0),'DNg positive FP32 gain')
 need(np.array_equal(o[:,:,:,7],o[:,:,:,11]) and np.array_equal(o[:,:,:,8],o[:,:,:,12]),'DNg unexpected target/rate override')
 negative=margin<=0;need(np.all(o[:,:,:,11][negative]==0),'DNg rectifier for nonpositive margin')
 target=o[:,:,:,11].astype(np.float32);need(np.array_equal(target.astype(np.float64),o[:,:,:,11]),'DNg target FP32')
 argument=o[:,:,:,6].astype(np.float32)*margin
 desired=np.array([max(0.,math.tanh(float(x))) for x in argument.flat],np.float32).reshape(argument.shape)
 ulp=np.abs(target.view(np.uint32).astype(np.int64)-desired.view(np.uint32).astype(np.int64))
 need(np.all(ulp<=2),'DNg positive tanhf beyond inherited 49 documented 2ULP')
 targets=last_targets(a['observer'],n);out.update({'DNg_target_mean_endpoints_q':float(targets.mean()),'DNg_margin_max':np.max(o[:,:,:,10],axis=(0,1)).tolist(),'writer_active_intervals':int(active.sum()),'witness_RHS_calls':int(c['counts'][:,0].sum())})
 diff=c['first'].astype(float)-v['first'].astype(float);native_norm=np.linalg.norm(v['first'],axis=1);need(np.all(native_norm>0),'native input norm')
 out['input_first_RHS']={'mean_L2_writer_change':float(np.linalg.norm(diff,axis=1).mean()),'maximum_L2_writer_change':float(np.linalg.norm(diff,axis=1).max()),'mean_relative_L2_writer_change':float((np.linalg.norm(diff,axis=1)/native_norm).mean()),'scope':'First RHS in each millisecond only; every RHS identity is checked separately by the device writer.'}
 out['maximum_positive_target_ULP']=int(ulp.max())
 return out,targets
def compute(d):
 p=d['original_plan'];q=d['repair_plan'];c=p['criteria'];need(q['criteria']==c and q['arms']==p['arms'] and q['analysis_window_ms']==p['analysis_window_ms'],'repair altered science')
 need(p['analysis_window_ms']==[65,128] and c['DNb_difference_change_q_min']==1.6e-5 and c['yaw_mean_change_deg_s_min']==.02 and c['forward_mean_change_mm_s_min']==.001 and c['DNg_mean_target_change_min']==.001 and c['persistence_amplification_min']==2.,'scientific thresholds')
 need(c['stage4'] is False and c['stage5'] is False,'stage scope')
 need(len(p['arms'])==10 and sum(x['duration_ms'] for x in p['arms'].values())==1028,'factorial exposure')
 allmetrics={};means={};win=slice(64,128)
 for name,spec in q['arms'].items():
  a=d['arms'][name]
  need(a['result']['arm']==name,'worker arm identity')
  for k in ['CPU_s','wall_s']:need(np.isfinite(a['result'][k]) and a['result'][k]>=0,'invalid worker cost '+k)
  for key in ['ids','rows','ORN_ids','ORN_rows']:need(np.array_equal(a['panel'][key],d['reference']['panel'][key]),'panel support '+key)
  v,targets=check_arm(a,spec,d['terminals'],d['initial'],name);allmetrics[name]=v
  if spec['duration_ms']==2:
   for part in ['trace','neural','consumed','panel','observer']:equal(a[part],d['reference'][part],'qualification '+name+'/'+part)
   for part in ['owners','events']:need(a[part]==d['reference'][part],'qualification '+name+'/'+part)
  else:
   t=a['trace'];means[name]={'yaw_deg_s':float(np.rad2deg(t['command_yaw_rate_rad_s'][win]).mean()),'DNb_difference_q':float((t['DN_q_actual'][win,2]-t['DN_q_actual'][win,3]).mean()),'forward_mm_s':float(t['command_forward_mm_s'][win].mean()),'DNg_target_q':float(targets[win].mean())}
 effects={}
 for receiver in ['sham','profile']:
  simple={mode:{key:means[mode+'_'+receiver+'_from_profile'][key]-means[mode+'_'+receiver+'_from_sham'][key] for key in means['pulse_sham_from_sham']} for mode in ['pulse','hold']}
  interaction={k:simple['hold'][k]-simple['pulse'][k] for k in simple['pulse']};checks={}
  for field,minimum in [('yaw_deg_s',.02),('DNb_difference_q',1.6e-5)]:
   pulse,held=simple['pulse'][field],simple['hold'][field];same=(pulse==0 and abs(held)>=minimum) or (np.sign(pulse)==np.sign(held) and pulse!=0);amplified=abs(held)>=2*abs(pulse) if pulse!=0 else abs(held)>=minimum
   checks[field]={'interaction_material':bool(abs(interaction[field])>=minimum),'same_sign_or_exact_zero_branch':bool(same),'amplified':bool(amplified),'hold_to_pulse_ratio':None if pulse==0 else held/pulse}
  supported=all(all(v[k] for k in ['interaction_material','same_sign_or_exact_zero_branch','amplified']) for v in checks.values());forward=abs(simple['hold']['forward_mm_s'])>=.001 and abs(simple['hold']['DNg_target_q'])>=.001
  effects[receiver]={'terminal_identity_effect':simple,'duration_by_identity_interaction':interaction,'checks':checks,'persistence_supported_at_this_frontier':bool(supported),'held_forward_effect_material':bool(forward)}
 interaction_receiver={mode:{k:effects['profile']['terminal_identity_effect'][mode][k]-effects['sham']['terminal_identity_effect'][mode][k] for k in means['pulse_sham_from_sham']} for mode in ['pulse','hold']}
 f=d['failure'];oq=d['original_queue'];rq=d['repair_queue'];need(oq['status']=='STOPPED' and oq['arms']==[] and oq['active']=='qual_disabled','original failure accounting')
 need('UnicodeEncodeError' in d['original_failure_log'] and 'INTERVENTION.json' in d['original_failure_log'] and '"attempted_ms": 0' in d['partial_result'] and '"committed_ms": 0' in d['partial_result'],'original encoding failure evidence')
 need(f['CPU_s_budget_charge']==170 and f['CNS_ms_budget_charge']==2 and f['attempted_CNS_ms_observed']==0 and f['committed_CNS_ms_observed']==0 and not f['CPU_measurement_available'],'failure budget preservation')
 records=[d['arms'][n]['result'] for n in q['arms']];cpu=sum(r['CPU_s'] for r in records);ms=sum(r['attempted_ms'] for r in records);committed=sum(r['committed_ms'] for r in records)
 fields=['arm','status','attempted_ms','committed_ms','CPU_s','wall_s'];need(rq['arms']==[{k:r[k] for k in fields} for r in records],'queue rows differ from workers')
 need(rq['status']=='COMPLETE' and rq['CPU_s']==cpu and rq['attempted_ms']==ms and rq['committed_ms']==committed and len(rq['arms'])==10,'queue reconstruction')
 need(cpu<=q['queue_CPU_s'] and ms<=q['CNS_ms'] and rq['queue_wall_s']<=q['queue_wall_s'],'repair subbudget')
 budget={'attempted_CNS_ms_measured':ms,'committed_CNS_ms_measured':committed,'CNS_ms_budget_charge':ms+2,'worker_CPU_s_measured_repaired':cpu,'worker_CPU_s_budget_charge':cpu+170,'queue_wall_s':oq['queue_wall_s']+rq['queue_wall_s'],'unmeasured_original_CPU_charge_s':170}
 need(budget['CNS_ms_budget_charge']<=1200 and budget['worker_CPU_s_budget_charge']<=6000 and budget['queue_wall_s']<=5000,'round ceiling')
 supported=all(x['persistence_supported_at_this_frontier'] for x in effects.values())
 s=d['terminals']['sham'].astype(float);t=d['terminals']['profile'].astype(float);dn=float(np.linalg.norm(t-s));need(np.linalg.norm(s)>0 and np.linalg.norm(t)>0,'source norms')
 sizes={'difference_L2':dn,'relative_to_sham_L2':float(dn/np.linalg.norm(s)),'relative_to_profile_L2':float(dn/np.linalg.norm(t)),'mean_absolute_difference':float(np.abs(t-s).mean()),'maximum_absolute_difference':float(np.abs(t-s).max()),'changed_terminals':int(np.count_nonzero(t!=s)),'total_terminals':int(len(s)),'scope':'Fixed observed initial FP32 release; descriptive, no new input-size promotion gate.'}
 return {'schema':'factorial56_recomputed_v1','fixed_input_contrast':sizes,'means_primary_window':means,'effects_by_receiver':effects,'receiver_by_terminal_interactions':interaction_receiver,'arms':allmetrics,'budget':budget,'joint_persistence_support':supported,'classification':'PROMETEDOR_NO_CONFIRMADO' if supported else 'DESCARTADO','classification_scope':'Sufficiency of increased duration on this partial fixed generic-CSR source under the frozen diagnostic, not existence of all PN routes or absence of a smaller measured causal effect.','stage4_admitted':False,'stage5_admitted':False,'limits':'Single exposed preparation and two histories; fixed initial source values are not natural traces. Specialized consumers may override the generic CSR. PN first/last/min/max plus every-RHS counters are retained, not complete RHS input tapes. No GPU resume from portable paths claimed.'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--corruptions',action='store_true');ap.add_argument('--root',type=Path,default=H);args=ap.parse_args();start=time.process_time();d=load(args.root);result=compute(d);p=args.root/'A_RESULTADOS.json'
 if args.write:need(not p.exists() and not (args.root/'A_VERIFY_COST.json').exists(),'immutable results')
 else:need(result==json.loads(p.read_text()),'published effects differ')
 tests=[]
 if args.corruptions:
  name='pulse_sham_from_profile'
  mutations=[('criterion',lambda x:x['original_plan']['criteria'].__setitem__('yaw_mean_change_deg_s_min',.0001)),('stage_flag',lambda x:x['repair_plan']['criteria'].__setitem__('stage4',True)),('source_identity',lambda x:x['terminals']['ids'].__setitem__(0,999)),('initial_state',lambda x:x['arms'][name]['intervention']['initial_state'].__setitem__(0,.9)),('writer_schedule',lambda x:x['arms'][name]['native']['mode'].__setitem__((1,0),1)),('held_source',lambda x:x['arms']['hold_sham_from_profile']['consumed']['first'].__setitem__((40,0),.123)),('pulse_release',lambda x:x['arms'][name]['native']['first'].__setitem__((40,0),.123)),('RHS_count',lambda x:x['arms'][name]['native']['counts'].__setitem__((10,0),0)),('RHS_writer_error',lambda x:x['arms'][name]['native']['errors'].__setitem__((10,0),1)),('ORN_error',lambda x:x['arms'][name]['ORN']['counts_and_errors'].__setitem__((10,1),1)),('motor',lambda x:x['arms'][name]['trace']['command_yaw_rate_rad_s'].__setitem__(20,123)),('clock',lambda x:x['arms'][name]['trace']['CNS_time_ns'].__setitem__(20,0)),('RHS_context',lambda x:x['arms'][name]['observer']['committed'].__setitem__(0,True)),('nonfinite',lambda x:x['arms'][name]['neural']['final_q'].__setitem__(0,np.nan)),('failure_budget',lambda x:x['failure'].__setitem__('CPU_s_budget_charge',0)),('accepted_flag_value',lambda x:x['arms'][name]['observer']['records'].__setitem__((int(np.flatnonzero(x['arms'][name]['observer']['records'][:,138]==1)[0]),138),2))]
  for label,fn in mutations:
   changed=copy.deepcopy(d);fn(changed)
   try:compute(changed)
   except (ValueError,KeyError) as e:tests.append({'name':label,'rejected':True,'reason':str(e)})
   else:raise ValueError('corruption accepted '+label)
 report={'CPU_s':time.process_time()-start,'optimized':not __debug__,'tests':tests}
 if args.write:
  p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(args.root/'A_VERIFY_COST.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'joint_persistence_support':result['joint_persistence_support'],'effects':result['effects_by_receiver'],'budget':result['budget'],'corruptions_rejected':len(tests),'CPU_s':report['CPU_s']}))
if __name__=='__main__':main()
