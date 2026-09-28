"""Rebuild short-pilot quantities from sealed traces, independent of runner verdicts."""
from pathlib import Path
import json,hashlib,sys,argparse,math
import numpy as np
H=Path(__file__).resolve().parent

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def yaw(q):
 w,x,y,z=q[...,3:7].T
 return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def error(q,source):
 bearing=np.arctan2(source[1]-10*q[...,1],source[0]-10*q[...,0]);d=bearing-yaw(q)
 return np.abs(np.rad2deg(np.arctan2(np.sin(d),np.cos(d))))
def compute(root):
 root=Path(root);plan=json.loads((root/'PILOT_PLAN.json').read_text());freeze=json.loads((root/'FREEZE.json').read_text());need(sha(root/'PILOT_PLAN.json')==freeze['plan_sha256'],'frozen plan')
 refs=arrays(root.parent/'analysis_refs/anchor.npz');refmeta=json.loads((root.parent/'analysis_refs/PROVENANCE.json').read_text());need(sha(root.parent/'analysis_refs/anchor.npz')==refmeta['anchor_sha256'],'anchor hash')
 n=plan['duration_ms'];lo,hi=plan['analysis_window_ms'];win=slice(lo-1,hi)
 need((n,lo,hi)==(89,51,89),'prospective repair window');need(plan['criteria']['stage4'] is False and plan['criteria']['stage5'] is False,'stage scope')
 for name in plan['qualifications']:
  need(sha(root.parent/'analysis_refs'/('qual_'+name+'.npz'))==refmeta['qualification_reference_sha256'][name],'qualification reference hash')
  actual=arrays(root/('qual_'+name)/'traces.npz');reference=arrays(root.parent/'analysis_refs'/('qual_'+name+'.npz'))
  for key,v in reference.items():
   newkey='neural_yaw_raw_rad_s' if key=='neural_yaw_unapplied_rad_s' else key
   need(np.array_equal(actual[newkey],v),'qualification '+name+'/'+key)
 data={};out={};pins={}
 for arm,setting in plan['arms'].items():
  d=root/arm;t=arrays(d/'traces.npz');a=arrays(d/'neural_and_inputs.npz');owner=json.loads((d/'SPATIAL_OWNER_FINAL.json').read_text())
  need(t['qpos'].shape==(n,109) and t['qvel'].shape==(n,108),'body shape '+arm)
  for values in [t,a]:
   for k,v in values.items():
    if v.dtype.kind in 'fc':need(np.isfinite(v).all(),'nonfinite '+arm+'/'+k)
  need(np.array_equal(a['initial_q'],refs['initial_q']),'common neural start '+arm)
  need(np.array_equal(t['spatial_sample_qpos'][0],refs['initial_qpos']),'body start '+arm)
  need(np.array_equal(t['spatial_sample_qpos'][1:],t['qpos'][:-1]),'body→sensor1ms clock '+arm)
  for key in ['CNS_time_ns','PN_time_ns','body_time_ns']:need(np.array_equal(t[key],47486000000+1000000*np.arange(1,n+1)),'owner clock '+key)
  need(np.array_equal(t['paso'],np.arange(3001,3001+n)),'interval order')
  need(np.array_equal(t['sensores_usados'],np.zeros((n,3))) and np.array_equal(t['sensores_pendientes'],np.zeros((n,3))),'legacy doubleinput')
  need(np.array_equal(t['DN_q_usada'][1:],t['DN_q_actual'][:-1]),'neural latency')
  need(np.array_equal(t['DN_baseline'],np.broadcast_to(refs['DN_baseline'],(n,4))),'baseline recentered')
  delta=t['DN_q_usada']-t['DN_baseline'];raw=np.tanh(250*(delta[:,2]-delta[:,3]))*math.radians(5.)
  fwd=np.clip(np.mean(delta[:,:2],axis=1),0.,.5)
  need(np.array_equal(raw,t['neural_yaw_raw_rad_s']) and np.array_equal(raw,t['command_yaw_rate_rad_s']),'raw/applied yaw or units')
  need(np.array_equal(fwd,t['command_forward_mm_s']),'forward reader')
  for key in ['ORN_ids','ORN_sides','baseline_Hz','delta_profile_Hz']:need(np.array_equal(a[key],refs[key]),'ORN mapping '+key)
  c=t['spatial_concentration_used'];need(c.shape==(n,2) and ((c>=0)&(c<=1)).all(),'concentration range')
  need(np.array_equal(c[:10],np.zeros((10,2))),'odor prefix')
  need(owner['source_side']==setting['source'],'source label')
  selected=1 if setting['source']=='R' else 0
  need(owner['source_mm']==refs['source_geometry'][selected].tolist() and owner['sigma_mm']==float(refs['sigma_mm']),'world source moved')
  antenna=np.concatenate([refs['initial_antennae_mm'][None],t['antenas_mm'][:-1]],axis=0)
  geometry=np.exp(-np.sum((antenna[:,:,:2]-refs['source_geometry'][selected])**2,axis=2)/(2.*float(refs['sigma_mm'])**2))
  need(np.array_equal(geometry,t['spatial_geometric_concentration']),'independent antenna/source geometry')
  if setting['source']=='none':need(np.array_equal(c,np.zeros((n,2))),'none contaminated')
  else:need(np.array_equal(c[10:],t['spatial_geometric_concentration'][10:]),'input not geometry')
  sides=(a['ORN_sides']=='R').astype(int);expected=a['baseline_Hz']+c[:,sides]*a['delta_profile_Hz']
  need(np.array_equal(expected,a['nominal_Hz']),'actual nominal ORN recomputation')
  # DNg target values come from the instrumented RHS; they are not spikes.
  with np.load(d/'dng100_observed.npz') as z:
   need(str(z['fields'][11])=='final_target' and np.array_equal(z['ids'],[10045,10056]),'DNg observer fields')
   targets=z['records'][:,:128].reshape(-1,4,2,16)[:,:,:,11]
   need(np.isfinite(targets).all(),'target finite');target_max=float(targets.max())
  # ORN target-consumption equality was checked during execution; its transient
  # witness counters were not retained as a separate raw array. Do not claim
  # independent offline replay of every consumed ORN RHS target.
  need(np.array_equal(a['source_geometry'],refs['source_geometry']) and float(a['sigma_mm'])==float(refs['sigma_mm']),'fixed world geometry')
  row={'mean_DNb05_L_minus_R_q':float((t['DN_q_actual'][:,2]-t['DN_q_actual'][:,3])[win].mean()),'mean_yaw_deg_s':float(np.rad2deg(raw[win]).mean()),'mean_forward_mm_s':float(fwd[win].mean()),'DNg_target_max_all_RHS':target_max,'physical_yaw_change_deg':float(np.rad2deg(yaw(t['qpos'][-1])-yaw(refs['initial_qpos']))),'physical_displacement_xy_mm':float(np.linalg.norm(10*(t['qpos'][-1,:2]-refs['initial_qpos'][:2]))),'final_error_L_R_deg':[float(error(t['qpos'][-1],s)) for s in refs['source_geometry']],'distance_change_L_R_mm':[(float(np.linalg.norm(10*t['qpos'][-1,:2]-s))-float(np.linalg.norm(10*refs['initial_qpos'][:2]-s))) for s in refs['source_geometry']],'consumed_ORN_total_mean_Hz':float(a['nominal_Hz'][win].sum(axis=1).mean())}
  data[arm]={'t':t,'a':a};out[arm]=row;pins[arm]={f:sha(d/f) for f in ['traces.npz','neural_and_inputs.npz','dng100_observed.npz','SPATIAL_OWNER_FINAL.json']}
 qmin=plan['criteria']['neural_half_L_minus_R_q_min'];ymin=plan['criteria']['yaw_half_L_minus_R_deg_s_min'];need(qmin==1.6e-5 and ymin==.02,'materiality changed')
 comparisons={}
 for law in ['parent','I']:
  for side in ['L','R']:
   for key in ['qpos','qvel','DN_q_actual','DN_q_usada','command_forward_mm_s','command_yaw_rate_rad_s','spatial_sample_qpos']:
    need(np.array_equal(data[law+'_'+side]['t'][key][:10],data[law+'_none']['t'][key][:10]),'unequal preodor prefix '+law+'/'+side+'/'+key)
  left,right,none=[out[law+'_'+s] for s in ['L','R','none']]
  q=(left['mean_DNb05_L_minus_R_q']-right['mean_DNb05_L_minus_R_q'])/2;y=(left['mean_yaw_deg_s']-right['mean_yaw_deg_s'])/2
  benefit=[none['final_error_L_R_deg'][i]-row['final_error_L_R_deg'][i] for i,row in enumerate([left,right])]
  comparisons[law]={'half_L_minus_R_DNb_q':q,'half_L_minus_R_yaw_deg_s':y,'source_vs_none_yaw_L_R_deg_s':[row['mean_yaw_deg_s']-none['mean_yaw_deg_s'] for row in [left,right]],'source_vs_none_forward_L_R_mm_s':[row['mean_forward_mm_s']-none['mean_forward_mm_s'] for row in [left,right]],'body_error_benefit_vs_none_L_R_deg':benefit,'neural_and_yaw_material':bool(abs(q)>=qmin and abs(y)>=ymin),'both_mirror_benefits_positive':bool(all(v>0 for v in benefit))}
 saturated={}
 for side in ['none','L','R']:
  a=data['I_'+side]['a']['final_q'];p0=data['parent_'+side]['a']['final_q'];eligible=refs['eligible'];fraction=float(np.mean(((a>=.999)&(p0<.999))[eligible]));z=arrays(root/('I_'+side)/'candidate_owner.npz');mask=z['S']>0
  need(np.isfinite(z['min_raw'][mask]).all() and np.isfinite(z['max_raw'][mask]).all(),'candidate bounds')
  saturated[side]={'new_saturated_eligible_fraction':fraction,'excluded':bool(fraction>.05),'base_rows_raw_below0':int((z['min_raw'][mask]<0).sum()),'base_rows_raw_above1':int((z['max_raw'][mask]>1).sum()),'scope':'raw base target before specialized overrides; final saturation evaluated separately'}
 for law in comparisons:
  comparisons[law]['longer_life_screen_supported']=comparisons[law]['neural_and_yaw_material'] and comparisons[law]['both_mirror_benefits_positive'] and (law!='I' or not any(v['excluded'] for v in saturated.values()))
 previous=json.loads((root.parent/'QUEUE_RESULT.json').read_text());queue=json.loads((root/'QUEUE_RESULT.json').read_text())
 need(queue['status']=='COMPLETE' and len(queue['arms'])==8,'incomplete queue')
 receipts=[]
 for label in ['qual_parent','qual_current']+list(plan['arms']):
  rec=json.loads((root/label/'RESULT.json').read_text());expected_ms=2 if label.startswith('qual_') else n
  need(rec['status']=='COMPLETE' and rec['attempted_ms']==expected_ms and rec['committed_ms']==expected_ms,'worker accounting '+label);receipts.append(rec)
 cpu=sum(v['CPU_s'] for v in receipts);attempted=sum(v['attempted_ms'] for v in receipts)
 need(cpu==queue['CPU_s'] and attempted==queue['attempted_ms'],'queue accounting mismatch')
 aggregate={'attempted_CNS_ms':previous['attempted_ms']+attempted,'committed_CNS_ms':previous['committed_ms']+sum(v['committed_ms'] for v in receipts),'worker_CPU_s':previous['CPU_s']+cpu,'queue_wall_s':previous['queue_wall_s']+queue['queue_wall_s'],'initial_failed_attempt_preserved':previous['status']=='STOPPED'}
 need(aggregate['attempted_CNS_ms']<=544 and aggregate['worker_CPU_s']<=3300 and aggregate['queue_wall_s']<=3000,'original aggregate budget exceeded')
 result={'schema':'source_body54_recomputed_v1','duration_ms':n,'window_ms':[lo,hi],'arms':out,'comparisons':comparisons,'saturation':saturated,'stage4_admitted':False,'stage5_admitted':False,'scope':'Short developmental pilot with applied neural commands; not reserved feedback or mechanical recovery test. Unequal anatomical ORN totals retained, not direction free of amount.','input_hashes':pins,'qualification_traces_exact':True,'budget':aggregate,'verification_limit':'Nominal ORN rates and body→sensor geometry independently recomputed offline. Equality at every actual ORN RHS target was checked by the frozen runner; its transient counters were not stored for separate offline recomputation.'}
 return result,data

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=H/'repair02');a.add_argument('--write',action='store_true');v=a.parse_args();result,_=compute(v.root);text=json.dumps(result,indent=2,allow_nan=False)+'\n';out=v.root/'RESULTADOS.json'
 if v.write:
  if out.exists():raise ValueError('immutable results')
  out.write_text(text)
 else:need(json.loads(out.read_text())==result,'published quantities/verdict differ')
 print(json.dumps({'comparisons':result['comparisons'],'stage4_admitted':False,'stage5_admitted':False}))
if __name__=='__main__':main()
