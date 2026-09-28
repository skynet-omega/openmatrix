"""Portable reconstruction from recorded arrays and prospectively frozen plans."""
from pathlib import Path
import json,hashlib,argparse,math,copy,time
import numpy as np
H=Path(__file__).resolve().parent
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def equal(x,y,label):
 need(set(x)==set(y),'keys '+label)
 for k in x:need(np.array_equal(x[k],y[k]),'array '+label+'/'+k)
def finite(d,label):
 for k,v in d.items():
  if v.dtype.kind in 'fc':need(np.isfinite(v).all(),'nonfinite '+label+'/'+k)
def load(root):
 root=Path(root);d={'plans':{},'arms':{},'queues':{},'reference':{},'qualification':{},'repair_qual':{}}
 for stem in ['FEEDBACK','PN']:
  f=json.loads((root/(stem+'_FREEZE.json')).read_text());need(sha(root/(stem+'_PLAN.json'))==f['contract_sha256'],'contract hash '+stem)
  d['plans'][stem]=json.loads((root/(stem+'_PLAN.json')).read_text());d['queues'][stem]=json.loads((root/(stem+'_RESULT.json')).read_text())
 f=json.loads((root/'repair02/PN_FREEZE.json').read_text());need(sha(root/'repair02/PN_PLAN.json')==f['contract_sha256'],'repair contract hash')
 d['plans']['PN_REPAIR']=json.loads((root/'repair02/PN_PLAN.json').read_text());d['queues']['PN_REPAIR']=json.loads((root/'repair02/PN_RESULT.json').read_text())
 d['failed_attempt']=json.loads((root/'pn_sham_from_profile/RESULT.json').read_text())
 for name in ['54I_none_traces','54I_none_neural','54parent_none_traces','JO52_nominal','PN_donors','PANEL52']:d['reference'][name]=arrays(root/'reference'/(name+'.npz'))
 for name,meta in json.loads((root/'reference/ANALYTIC_REFERENCES.json').read_text()).items():need(sha(root/'reference'/name)==meta['sha256'],'reference identity')
 d['reference']['PN_meta']=json.loads((root/'reference/PN_donors.json').read_text())
 d['reference']['jump_identities']=json.loads((root/'reference/JUMP_OWNER_IDENTITIES.json').read_text())
 for arm in ['feedback_online','feedback_replay']+list(d['plans']['PN']['arms']):
  path=(root/'repair02' if arm in ['pn_sham_from_profile','pn_profile_from_sham'] else root)/arm;entry={'trace':arrays(path/'traces.npz'),'neural':arrays(path/'neural_and_inputs.npz'),'observer':arrays(path/'dng100_observed.npz'),'result':json.loads((path/'RESULT.json').read_text()),'events':json.loads((path/'EVENTS.json').read_text())}
  if arm.startswith('pn_') or arm.startswith('qual_'):
   entry.update(patch=arrays(path/'INTERVENTION.npz'),patch_meta=json.loads((path/'INTERVENTION.json').read_text()),consumed=arrays(path/'PN_consumed.npz'),panel=arrays(path/'wide_observation.npz'),owners=json.loads((path/'SCIENTIFIC_WITNESS.json').read_text()))
  else:entry['candidate']=arrays(path/'candidate_owner.npz')
  if path.parent.name=='repair02':entry['jumps']=json.loads((path/'JUMP_EVENTS.json').read_text())
  d['arms'][arm]=entry
 for receiver in ['sham','profile']:
  path=root/'repair02'/('qual_'+receiver+'_self');entry={k:arrays(path/name) for k,name in [('trace','traces.npz'),('neural','neural_and_inputs.npz'),('observer','dng100_observed.npz'),('consumed','PN_consumed.npz'),('panel','wide_observation.npz')]}
  entry.update({k:json.loads((path/name).read_text()) for k,name in [('owners','SCIENTIFIC_WITNESS.json'),('events','EVENTS.json'),('result','RESULT.json'),('jumps','JUMP_EVENTS.json')]});d['repair_qual'][receiver]=entry
 return d

def trace_check(a,n,name):
 t=a['trace'];v=a['neural'];finite(t,name);finite(v,name)
 need(t['qpos'].shape==(n,109) and t['qvel'].shape==(n,108),'body shape '+name)
 for k in ['CNS_time_ns','PN_time_ns','body_time_ns']:need(np.array_equal(t[k],47486000000+1000000*np.arange(1,n+1)),'clock '+name+'/'+k)
 need(np.array_equal(t['paso'],np.arange(3001,3001+n)),'step sequence')
 need(np.array_equal(t['spatial_sample_qpos'][1:],t['qpos'][:-1]),'sensor body clock')
 need(not np.any(t['sensores_usados']) and not np.any(t['sensores_pendientes']),'legacy double input')
 need(not np.any(t['spatial_concentration_used']),'baseline condition contaminated')
 need(np.array_equal(v['nominal_ORN_Hz'],np.broadcast_to(v['baseline_Hz'],(n,694))),'ORN baseline changed')
 need(np.array_equal(t['DN_q_usada'][1:],t['DN_q_actual'][:-1]),'DN motor latency')
 need(np.array_equal(t['DN_baseline'],np.broadcast_to(t['DN_baseline'][0],(n,4))),'reader recentered')
 delta=t['DN_q_usada']-t['DN_baseline'];raw=np.tanh(250*(delta[:,2]-delta[:,3]))*math.radians(5.)
 need(np.array_equal(raw,t['command_yaw_rate_rad_s']) and np.array_equal(raw,t['neural_yaw_raw_rad_s']),'yaw reader/unit')
 fwd=np.clip(delta[:,:2].mean(axis=1),0.,.5);need(np.array_equal(fwd,t['command_forward_mm_s']),'forward reader')
 z=a['observer'];finite(z,name+' observer');need(np.array_equal(z['ids'],[10045,10056]) and str(z['fields'][11])=='final_target','DNg observer identity')
 need(np.array_equal(z['rows'],[36,46]) and np.array_equal(z['epoch'],np.arange(16*n)),'DNg epoch identity')
 need(np.array_equal(z['committed'],np.tile([False,True],8*n)) and np.array_equal(z['duration_ns'],np.tile([62500,125000],8*n)),'predictor/committed contexts')
 need(np.array_equal(z['ms'],np.repeat(np.arange(3001,3001+n),16)) and np.array_equal(z['start_ns'],47486000000+np.repeat(np.arange(8*n)*125000,2)),'context clock')
 rec=z['records'];need(rec.shape[1]==140 and z['offsets'][0]==0 and z['offsets'][-1]==len(rec),'RHS record layout')
 need(np.all(rec[:,132:134]==0) and np.array_equal(rec[:,138].astype(bool),rec[:,131]<=1),'scheduler flags/predicate')
 need(np.array_equal(np.diff(z['offsets']),z['trials']) and np.array_equal(z['accepted']+z['rejected'],z['trials']),'RHS epoch accounting')
 idx=np.arange(len(rec));starts=z['offsets'][:-1];ends=z['offsets'][1:];epoch_start=np.repeat(starts,z['trials']);accepted=rec[:,138].astype(bool)
 need(np.array_equal(rec[:,139],idx-epoch_start) and np.array_equal(np.add.reduceat(accepted.astype(np.int64),starts),z['accepted']),'RHS trial sequence')
 latest=np.maximum.accumulate(np.where(accepted,idx,-1));previous=np.r_[-1,latest[:-1]];no_previous=previous<epoch_start
 previous_time=rec[np.maximum(previous,0),130].copy();previous_time[no_previous]=0.
 previous_state=rec[np.maximum(previous,0),136:138].copy();previous_state[no_previous]=rec[epoch_start[no_previous],134:136]
 need(np.array_equal(rec[:,128],previous_time),'RHS trial clock')
 need(np.array_equal(rec[:,134:136],previous_state),'RHS state continuity')
 final=latest[ends-1];need(np.all(final>=starts) and np.array_equal(rec[final,130],z['duration_ns']*1e-9),'RHS final clock')
 r=a['result'];need(r['status']=='COMPLETE' and r['attempted_ms']==n and r['committed_ms']==n,'worker completeness')
 return dict(DNg_target_max_all_RHS=float(rec[:,:128].reshape(-1,4,2,16)[:,:,:,11].max()),mean_forward_full_mm_s=float(fwd.mean()),final_forward_mm_s=float(fwd[-1]))

def compute(d):
 fp,pp=d['plans']['FEEDBACK'],d['plans']['PN'];a=d['arms'];ref=d['reference']
 need((fp['duration_ms'],fp['analysis_window_ms'],pp['analysis_window_ms'])==(89,[51,89],[65,128]),'prospective windows')
 for p in [fp,pp]:
  c=p['criteria'];need(c['stage4'] is False and c['stage5'] is False,'stage scope')
  need(c.get('yaw_window_mean_change_deg_s_min',c.get('yaw_mean_change_deg_s_min'))==.02 and c['DNb_difference_change_q_min']==1.6e-5,'material thresholds')
 need(fp['criteria']['JO_relative_L2_separation_min']==.01 and pp['criteria']['consumed_PN_relative_change_min']==.01,'input thresholds')
 out={}
 for name,arm in a.items():out[name]=trace_check(arm,89 if name.startswith('feedback') else pp['arms'][name]['duration_ms'],name)
 online,replay=a['feedback_online'],a['feedback_replay'];win=slice(50,89)
 for k in ['q0','S','gE0']:need(np.array_equal(online['candidate'][k],replay['candidate'][k]),'I initialization changed '+k)
 equal(online['trace'],ref['54I_none_traces'],'online exact54')
 need(np.array_equal(online['neural']['final_q'],ref['54I_none_neural']['final_q']),'final online exact54')
 for arm in [online,replay]:
  v=arm['neural'];need(np.array_equal(v['JO_ids'],ref['JO52_nominal']['JO_ids']) and np.array_equal(v['JO_rows'],ref['JO52_nominal']['JO_rows']),'JO identities')
  need(np.array_equal(v['JO_consumed_FP32'],(v['JO_native']+v['JO_injected']).astype(np.float32)),'JO actual postFP32')
  audit=v['JO_kernel_audit'];need(audit.shape==(89,3) and np.all(audit[:,0]>0) and not np.any(audit[:,1]),'JO allRHS witness')
  need(not np.any(v['JO_injected'][:10]),'JO common zero prefix')
  need(np.array_equal(v['initial_q'],online['neural']['initial_q']),'common initial state')
 need(np.array_equal(online['neural']['JO_online'],online['neural']['JO_injected']),'online JO writer')
 need(np.array_equal(replay['neural']['JO_injected'],ref['JO52_nominal']['JO_drive']),'tape exact index')
 need(np.array_equal(replay['trace']['CNS_time_ns'],ref['JO52_nominal']['CNS_time_ns']),'tape clock')
 for k in ['DN_q_actual','qpos','qvel']:need(np.array_equal(online['trace'][k][:10],replay['trace'][k][:10]),'feedback common prefix')
 y=lambda x,w:float(np.rad2deg(x['trace']['command_yaw_rate_rad_s'][w]).mean())
 q=lambda x,w:float((x['trace']['DN_q_actual'][:,2]-x['trace']['DN_q_actual'][:,3])[w].mean())
 yo,yr=y(online,win),y(replay,win);y52=float(np.rad2deg(ref['JO52_nominal']['yaw_raw_rad_s'][win]).mean());sep=float(np.linalg.norm(online['neural']['JO_injected'][win]-replay['neural']['JO_injected'][win])/np.linalg.norm(online['neural']['JO_injected'][win]));dq=q(replay,win)-q(online,win)
 C=dict(JO_relative_L2=sep,online_mean_yaw_deg_s=yo,replay_mean_yaw_deg_s=yr,nominal52_mean_yaw_deg_s=y52,replay_minus_online_yaw_deg_s=yr-yo,replay_minus_online_DNb_L_minus_R_q=dq,nominal52_residual_yaw_deg_s=yr-y52,fraction_mean_yaw_gap_recovered=(yr-yo)/(y52-yo),material=bool(sep>=.01 and abs(yr-yo)>=.02 and abs(dq)>=1.6e-5),classification='PROMETEDOR_NO_CONFIRMADO',native_JO_nonzero_RHS_elements=int(sum(x['neural']['JO_kernel_audit'][:,2].sum() for x in [online,replay])),scope='Causal effect of declared JO replay in basal I; single exposed life, partial feedback, not navigation or sole cause.')
 # Validate both live identity continuations; never trust their PASS field.
 for receiver in ['sham','profile']:
  u,s=[a['qual_'+receiver+'_'+suffix] for suffix in ['untouched','self']]
  for part in ['trace','neural','observer','consumed','panel']:equal(u[part],s[part],'PN live identity '+receiver+'/'+part)
  need(u['events']==s['events'] and u['owners']==s['owners'],'PN live owners/events')
  repaired=d['repair_qual'][receiver];trace_check(repaired,2,'repair identity '+receiver)
  for part in ['trace','neural','observer','consumed','panel']:equal(s[part],repaired[part],'repair identity '+receiver+'/'+part)
  need(s['events']==repaired['events'] and s['owners']==repaired['owners'] and not repaired['jumps']['events'],'repair no-op states/events')
 for name,spec in pp['arms'].items():
  arm=a[name];p=arm['patch'];m=arm['patch_meta'];finite(p,name+' patch')
  selected=ref['PN_donors']['selected_indices'];need(np.array_equal(p['selected_indices'],selected) and np.array_equal(p['pn_ids'],ref['PN_donors']['PN_ids']),'PN patch support')
  before,after=p['before_state'],p['after_state'];outside=np.ones(len(before),bool);outside[selected]=False
  need(np.array_equal(before[selected],ref['PN_donors'][spec['receiver']]),'PN receiver state')
  need(np.array_equal(p['donor_values'],ref['PN_donors'][spec['donor']]),'PN donor state')
  need(np.array_equal(before[outside],after[outside]),'outside PN patch')
  need(np.array_equal(after[selected],ref['PN_donors'][spec['donor']] if spec['apply'] else before[selected]),'patch application')
  need(m['spec']==spec and m['fine_before']==m['fine_after'],'fine PN preservation')
  need(all(m['before'][k]==m['after'][k] for k in m['before'] if k not in ['session','published']),'scientific owner changed by patch')
  if spec['receiver']==spec['donor']:need(np.array_equal(before,after) and m['before']==m['after'],'identity mutation')
  else:
   events=arm['jumps']['events'];need(len(events)==1,'declared GABA jump count');event=events[0]
   need(event['identity']==ref['jump_identities'][spec['receiver']]['identity'],'jump receptor identity')
   need(event['family']=='GABA' and event['source_time_ns']==47486000000 and event['arrival_time_ns']==47486125000,'registered jump owner/clock/delay')
   left,right=np.array(event['before']),np.array(event['after']);need(left.shape==right.shape==(86,) and np.isfinite(left).all() and np.isfinite(right).all() and np.all((left>=0)&(left<=1)) and np.all((right>=0)&(right<=1)),'jump driver bounds')
   need(event['changed_indices']==np.flatnonzero(left!=right).tolist() and len(event['changed_indices'])==7,'jump support')
   mapping=dict(zip(p['pn_ids'].tolist(),p['pn_rows'].tolist()))
   for index,name_id in enumerate(event['names']):
    cell=int(name_id)
    if cell in mapping:need(left[index]==before[mapping[cell]] and right[index]==after[mapping[cell]],'jump endpoint not PN intervention')
    else:need(left[index]==right[index],'jump changed nonPN input')
  c=arm['consumed'];finite(c,name+' consumed');need(c['first'].shape==(spec['duration_ms'],686),'PN consumed shape')
  need(np.array_equal(arm['panel']['ids'],ref['PANEL52']['ids']) and np.array_equal(arm['panel']['rows'],ref['PANEL52']['canonical_rows']),'wide panel identity')
  need(np.all(c['counts']>0) and np.all(c['counts']==c['counts'][:,:1]),'PN consumed RHS coverage')
  need(np.all(c['lo']<=c['first']) and np.all(c['lo']<=c['last']) and np.all(c['hi']>=c['first']) and np.all(c['hi']>=c['last']),'PN consumed bounds')
  if spec['receiver']=='sham' and spec['donor']=='sham':
   n=min(89,spec['duration_ms']);equal({k:v[:n] for k,v in arm['trace'].items()},{k:v[:n] for k,v in ref['54parent_none_traces'].items()},'parent no-op54')
 win=slice(64,128);contrasts={};target_sign=[]
 for receiver,donor in [('sham','profile'),('profile','sham')]:
  base=a['pn_'+receiver+'_from_'+receiver];cross=a['pn_'+receiver+'_from_'+donor];other=a['pn_'+donor+'_from_'+donor]
  base_signal=np.stack([base['consumed'][k][win] for k in ['first','last']]);cross_signal=np.stack([cross['consumed'][k][win] for k in ['first','last']]);relative=float(np.linalg.norm(cross_signal-base_signal)/max(np.linalg.norm(base_signal),np.finfo(float).tiny))
  dy=y(cross,win)-y(base,win);dq=q(cross,win)-q(base,win);direction=y(other,win)-y(base,win);material=relative>=.01 and abs(dy)>=.02 and abs(dq)>=1.6e-5;correct=dy*direction>0
  contrasts[receiver+'_from_'+donor]=dict(consumed_PN_relative_L2=relative,DNb_L_minus_R_change_q=dq,yaw_change_deg_s=dy,other_donor_self_minus_receiver_yaw_deg_s=direction,mean_yaw_self_deg_s=y(base,win),mean_yaw_cross_deg_s=y(cross,win),mean_forward_self_mm_s=float(base['trace']['command_forward_mm_s'][win].mean()),mean_forward_cross_mm_s=float(cross['trace']['command_forward_mm_s'][win].mean()),material=bool(material),toward_other_donor=bool(correct));target_sign.append(material and correct)
  transient=np.rad2deg(cross['trace']['command_yaw_rate_rad_s']-base['trace']['command_yaw_rate_rad_s'])
  contrasts[receiver+'_from_'+donor].update(max_abs_yaw_change_full_deg_s=float(np.max(np.abs(transient))),max_abs_yaw_change_at_ms=int(np.argmax(np.abs(transient))+1),PN_first_RHS_absolute_max_difference=float(np.max(np.abs(cross['consumed']['first'][0]-base['consumed']['first'][0]))),DN_max_abs_difference_full_q=float(np.max(np.abs(cross['trace']['DN_q_actual']-base['trace']['DN_q_actual']))),final_body_xy_separation_mm=float(np.linalg.norm(10*(cross['trace']['qpos'][-1,:2]-base['trace']['qpos'][-1,:2]))))
  dnmask=ref['PANEL52']['superclass']=='descending_neuron';dndiff=np.max(np.abs(cross['panel']['q'][:,dnmask]-base['panel']['q'][:,dnmask]),axis=0)
  contrasts[receiver+'_from_'+donor]['wide_DN_description']=dict(population=int(dnmask.sum()),max_abs_difference_q=float(dndiff.max()),count_above_1e_minus6=int((dndiff>1e-6).sum()),scope='Posthoc coverage description only; not navigation-related cells or an admission criterion.')
  contrasts[receiver+'_from_'+donor]['neural_and_motor_thresholds_met']=bool(abs(dy)>=.02 and abs(dq)>=1.6e-5)
 A=dict(contrasts=contrasts,reciprocal_material_transport=bool(all(target_sign)),classification='PROMETEDOR_NO_CONFIRMADO' if all(target_sign) else 'DESCARTADO',classification_scope='Promotion under the joint frozen screen, including1percent aggregatePN signal criterion. Does not erase an observed neural/motor effect below that input criterion.',input_small_in_any_direction=bool(any(v['consumed_PN_relative_L2']<.01 for v in contrasts.values())),scope='Only686ALPN q/commonfilter sufficient transport at3s+128ms underparent; not allPN memories, no claim absence of route, no navigation.')
 budget=dict(attempted_CNS_ms=0,committed_CNS_ms=0,worker_CPU_s=0.,queue_wall_s=0.)
 need(d['failed_attempt']['status']=='FAILED' and d['failed_attempt']['attempted_ms']==1 and d['failed_attempt']['committed_ms']==0 and 'Graded source history discontinuity' in d['failed_attempt']['error'],'original failed attempt preserved')
 need(d['plans']['PN_REPAIR']['criteria']==pp['criteria'] and d['plans']['PN_REPAIR']['analysis_window_ms']==pp['analysis_window_ms'],'repair changed scientific criteria')
 for label in ['FEEDBACK','PN','PN_REPAIR']:
  if label=='FEEDBACK':recs=[a[n]['result'] for n in ['feedback_online','feedback_replay']]
  elif label=='PN':recs=[a[n]['result'] for n in list(pp['arms'])[:6]]+[d['failed_attempt']]
  else:recs=[v['result'] for v in d['repair_qual'].values()]+[a[n]['result'] for n in ['pn_sham_from_profile','pn_profile_from_sham']]
  queue=d['queues'][label];cpu=sum(r['CPU_s'] for r in recs);ms=sum(r['attempted_ms'] for r in recs);committed=sum(r['committed_ms'] for r in recs)
  need(queue['status']==('STOPPED' if label=='PN' else 'COMPLETE') and len(queue['arms'])==len(recs) and queue['CPU_s']==cpu and queue['attempted_ms']==ms and queue['committed_ms']==committed,'queue receipts')
  need(cpu<=d['plans'][label]['queue_CPU_s'] and ms<=d['plans'][label]['CNS_ms'] and queue['queue_wall_s']<=d['plans'][label]['queue_wall_s'],'subbudget')
  budget['attempted_CNS_ms']+=ms;budget['committed_CNS_ms']+=committed;budget['worker_CPU_s']+=cpu;budget['queue_wall_s']+=queue['queue_wall_s']
 need(budget['attempted_CNS_ms']<=800 and budget['worker_CPU_s']<=4000 and budget['queue_wall_s']<=3600,'aggregate budget')
 return dict(schema='campaign55_recomputed_v1',A=A,C=C,arms=out,budget=budget,repair=dict(original_failed_attempt_preserved=True,identity_continuations_exact=True,registered_GABA_source_jumps=7,source_to_receptor_delay_ns=125000,previous_occupancy_and_history_not_overwritten=True),stage4_admitted=False,stage5_admitted=False,verification_limits='Recorded nominalORN and clocks checked; everyORN RHS equality was enforced live but no rawORNcounterarchive retained. JO audit records everyactualFP32RHS mismatch counts; PN records first/last/min/max perms, not a completeRHS tape. GPUportable resume unqualified.')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=H);ap.add_argument('--write',action='store_true');ap.add_argument('--corruptions',action='store_true');v=ap.parse_args();start=time.process_time();d=load(v.root);result=compute(d);path=v.root/'RESULTADOS.json'
 if v.write:need(not path.exists(),'immutable results');path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 else:need(json.loads(path.read_text())==result,'published metrics/verdict differ')
 tests=[]
 if v.corruptions:
  mutations=[('stage_flag',lambda x:x['plans']['PN']['criteria'].__setitem__('stage4',True)),('threshold',lambda x:x['plans']['FEEDBACK']['criteria'].__setitem__('yaw_window_mean_change_deg_s_min',.0001)),('clock',lambda x:x['arms']['feedback_replay']['trace']['CNS_time_ns'].__setitem__(20,0)),('JO_tape',lambda x:x['arms']['feedback_replay']['neural']['JO_injected'].__setitem__((30,0),.333)),('JO_FP32',lambda x:x['arms']['feedback_online']['neural']['JO_consumed_FP32'].__setitem__((30,0),.333)),('JO_RHS_flag',lambda x:x['arms']['feedback_online']['neural']['JO_kernel_audit'].__setitem__((30,1),1)),('PN_outside',lambda x:x['arms']['pn_sham_from_profile']['patch']['after_state'].__setitem__(0,.8)),('PN_ids',lambda x:x['arms']['pn_sham_from_profile']['patch']['pn_ids'].__setitem__(0,999)),('motor_reader',lambda x:x['arms']['pn_profile_from_sham']['trace']['command_yaw_rate_rad_s'].__setitem__(20,123.)),('RHS_context',lambda x:x['arms']['pn_sham_from_sham']['observer']['records'].__setitem__((0,132),1)),('nonfinite',lambda x:x['arms']['pn_sham_from_profile']['neural']['final_q'].__setitem__(0,np.nan))]
  mutations.extend([('jump_retroactive',lambda x:x['arms']['pn_sham_from_profile']['jumps']['events'][0].__setitem__('arrival_time_ns',47486000000)),('jump_identity',lambda x:x['arms']['pn_sham_from_profile']['jumps']['events'][0].__setitem__('identity','different')),('predictor_context',lambda x:x['arms']['pn_sham_from_profile']['observer']['committed'].__setitem__(0,True))])
  for name,fn in mutations:
   changed=copy.deepcopy(d);fn(changed)
   try:compute(changed)
   except (ValueError,KeyError) as e:tests.append(dict(name=name,rejected=True,reason=str(e)))
   else:raise ValueError('corruption accepted '+name)
  report=dict(tests=tests,CPU_s=time.process_time()-start,optimized=not __debug__)
  if v.write:(v.root/'CORRUPTION_TESTS.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(A=result['A'],C=result['C'],budget=result['budget'],stage4=False,stage5=False,corruptions_rejected=len(tests),CPU_s=time.process_time()-start),allow_nan=False))
if __name__=='__main__':main()
