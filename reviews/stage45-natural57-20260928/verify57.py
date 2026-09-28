"""Recompute spatial signal, causal input history, reader and body contrasts from arrays."""
from pathlib import Path
import json,hashlib,argparse,copy,time,math,resource
import numpy as np
from trace_contract57 import need,finite,trace_check
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def load(root):
 root=Path(root);f=json.loads((root/'FREEZE.json').read_text())
 for name,key in [('PLAN.json','plan_sha256'),('SOURCES.json','sources_sha256'),('reference/PROVENANCE.json','reference_provenance_sha256')]:need(sha(root/name)==f[key],'frozen '+name)
 ref=root/'reference';provenance=json.loads((ref/'PROVENANCE.json').read_text())
 for path,d in provenance.items():need(sha(ref/path)==d['sha256'],'reference '+path)
 d={'plan':json.loads((root/'PLAN.json').read_text()),'queue':json.loads((root/'QUEUE_RESULT.json').read_text()),'anchor':arrays(ref/'anchor.npz'),'terminals':arrays(ref/'TERMINALS.npz'),'reference':{},'arms':{}}
 files=dict(trace='traces.npz',neural='neural_and_inputs.npz',panel='wide_observation.npz',observer='dng100_observed.npz')
 for arm in ['parent_none','parent_L','parent_R','qualification56']:
  d['reference'][arm]={k:arrays(ref/arm/v) for k,v in files.items()}
  if arm=='qualification56':
   d['reference'][arm]['consumed']=arrays(ref/arm/'PN_consumed.npz')
   for k,v in [('owners','SCIENTIFIC_WITNESS.json'),('events','EVENTS.json')]:d['reference'][arm][k]=json.loads((ref/arm/v).read_text())
 for arm in d['plan']['arms']:
  a={k:arrays(root/arm/v) for k,v in {**files,'consumed':'PN_consumed.npz','orn_consumed':'ORN_generic_consumed.npz','ORN':'ORN_AUDIT.npz','clock':'CLOCK_BODY.npz'}.items()}
  for k,v in [('result','RESULT.json'),('owners','SCIENTIFIC_WITNESS.json'),('events','EVENTS.json'),('spatial','SPATIAL_OWNER_FINAL.json')]:a[k]=json.loads((root/arm/v).read_text())
  if arm!='qual':a['horizon']=json.loads((root/arm/'HORIZON_EXTENSION.json').read_text())
  d['arms'][arm]=a
 return d
def yaw(q):
 w,x,y,z=q[...,3:7].T;return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def angle_error(q,source):
 bearing=np.arctan2(source[1]-10*q[...,1],source[0]-10*q[...,0]);a=bearing-yaw(q)
 return np.rad2deg(np.arctan2(np.sin(a),np.cos(a)))
def endpoint(observer,n):
 z=observer;r=z['records'];accepted=r[:,138]==1;latest=np.maximum.accumulate(np.where(accepted,np.arange(len(r)),-1));end=z['start_ns']+z['duration_ns'];epochs=np.flatnonzero(z['committed'] & ((end-47486000000)%1000000==0));need(len(epochs)==n,'endpoint epochs');indices=latest[z['offsets'][epochs+1]-1];need(np.all(indices>=z['offsets'][epochs]),'accepted endpoint')
 return r[indices,:128].reshape(-1,4,2,16)[:,3,:,11]
def equal_subset(a,b,label,n=None):
 for k,v in b.items():need(k in a and np.array_equal(a[k] if n is None else a[k][:n],v),'reference '+label+'/'+k)
def check_arm(a,spec,anchor,terminals,name):
 n=spec['duration_ms'];out=trace_check(a,n,name);t=a['trace'];v=a['neural'];z=a['observer'];r=z['records'];finite(a['panel'],name+' panel')
 need(a['result']['arm']==name,'worker label');need(a['spatial']['source_side']==spec['source'],'source label');need(a['spatial']['prefix_ms']==10 and not a['spatial']['uniform_qualification'],'source schedule')
 need(np.array_equal(v['initial_q'],anchor['initial_q']) and np.array_equal(t['spatial_sample_qpos'][0],anchor['initial_qpos']),'common start')
 need(np.array_equal(t['DN_baseline'],np.broadcast_to(anchor['DN_baseline'],(n,4))),'baseline recentered')
 for k in ['ORN_ids','ORN_sides','baseline_Hz','delta_profile_Hz','source_geometry','sigma_mm']:need(np.array_equal(v[k],anchor[k]),'mapping '+k)
 selected=1 if spec['source']=='R' else 0;source=anchor['source_geometry'][selected];sigma=float(anchor['sigma_mm'])
 need(a['spatial']['source_mm']==source.tolist() and a['spatial']['sigma_mm']==sigma,'source geometry')
 antenna=np.concatenate([anchor['initial_antennae_mm'][None],t['antenas_mm'][:-1]],axis=0);geometry=np.exp(-np.sum((antenna[:,:,:2]-source)**2,axis=2)/(2*sigma**2))
 need(np.array_equal(geometry,t['spatial_geometric_concentration']),'antenna geometry')
 c=t['spatial_concentration_used'];need(c.shape==(n,2) and np.isfinite(c).all() and ((c>=0)&(c<=1)).all(),'concentration range');need(not np.any(c[:10]),'basal prefix')
 if spec['source']=='none':need(not np.any(c),'none source')
 else:need(np.array_equal(c[10:],geometry[10:]),'source not current geometry')
 sides=(v['ORN_sides']=='R').astype(int);expected=v['baseline_Hz']+c[:,sides]*v['delta_profile_Hz'];need(np.array_equal(expected,v['nominal_ORN_Hz']),'ORN transduction')
 audit=a['ORN']['counts_and_errors'];need(audit.shape==(n,2) and np.all(audit[:,0]>0) and not np.any(audit[:,1]),'ORN target witness')
 for tag,width in [('consumed',686),('orn_consumed',694)]:
  x=a[tag];finite(x,name+tag)
  for k in ['first','last','lo','hi','counts']:need(x[k].shape==(n,width),'witness shape')
  need(x['first'].dtype==np.float32 and x['last'].dtype==np.float32,'consumed precision')
  need(np.all(x['counts']>0) and np.array_equal(x['counts'],np.broadcast_to(x['counts'][:,:1],(n,width))),'witness count')
  for k in ['first','last']:need(np.all(x[k]>=x['lo']) and np.all(x[k]<=x['hi']),'witness extrema')
  need(np.all(x['lo']>=0) and np.all(x['hi']<=1),'release bounds')
 need(np.array_equal(a['consumed']['counts'][:,0],a['orn_consumed']['counts'][:,0]),'PN ORN same CSR calls')
 # GraphRK23.__init__ executes two four-RHS warmup trials once; advance then
 # resets the DNg trial recorder. The source witness still sees those8calls.
 expected_calls=4*z['trials'].reshape(n,16).sum(axis=1);expected_calls[0]+=8
 need(np.array_equal(a['consumed']['counts'][:,0],expected_calls),'CSR calls including initial8warmup evaluations')
 need(np.array_equal(v['PN_ids'],terminals['ids']) and np.array_equal(v['PN_rows'],terminals['rows']) and v['PN_q'].shape==(n,686),'PN identity')
 need(np.array_equal(v['PN_q'][-1],v['final_q'][terminals['rows']]),'PN endpoint')
 panel=a['panel'];need(np.array_equal(panel['q'][-1],v['final_q'][panel['rows']]) and np.array_equal(panel['ORN_q'][-1],v['final_q'][panel['ORN_rows']]),'panel endpoint')
 need(np.array_equal(panel['time_ns'],t['CNS_time_ns']) and np.array_equal(panel['ORN_ids'],v['ORN_ids']),'panel clock/ORN')
 flags=r[:,138];need(np.array_equal(flags,flags.astype(bool).astype(flags.dtype)),'binary accepted flag')
 o=r[:,:128].reshape(-1,4,2,16);margin=(o[:,:,:,1].astype(np.float32)+o[:,:,:,4].astype(np.float32))-o[:,:,:,5].astype(np.float32)
 need(np.array_equal(margin.astype(np.float64),o[:,:,:,10]),'DNg margin arithmetic')
 need(np.all(o[:,:,:,6]>0) and np.array_equal(o[:,:,:,6],o[:,:,:,6].astype(np.float32).astype(np.float64)),'DNg gain')
 need(np.array_equal(o[:,:,:,7],o[:,:,:,11]) and np.array_equal(o[:,:,:,8],o[:,:,:,12]),'DNg override')
 target=o[:,:,:,11].astype(np.float32);need(np.array_equal(target.astype(np.float64),o[:,:,:,11]),'DNg target precision');need(np.all(target[margin<=0]==0),'DNg rectifier')
 argument=o[:,:,:,6].astype(np.float32)*margin;desired=np.maximum(0,np.tanh(argument.astype(float))).astype(np.float32);ulp=np.abs(target.view(np.uint32).astype(np.int64)-desired.view(np.uint32).astype(np.int64));need(np.all(ulp<=2),'DNg tanhf inherited2ULP')
 clock=a['clock'];finite(clock,name+' clock');expected_time=float(clock['initial_time_s']);need(clock['steps'].dtype==np.int64,'steps integer')
 for j in range(n):
  for _ in range(40):expected_time+=2.5e-5
  need(clock['time_s'][j]==expected_time and clock['steps'][j]==clock['initial_steps']+40*(j+1),'exact body recurrence')
 if n>200:
  ext=a['horizon'];need(ext['old_duration_ms']==3200 and ext['new_duration_ms']==3384 and ext['cold_loaded_ms']==3000 and ext['clock_values_changed'] is False and ext['dt_changed'] is False,'horizon scope')
 out.update(DNg_target_mean_endpoints_q=float(endpoint(z,n).mean()),DNg_margin_max=np.max(margin,axis=(0,1)).tolist(),maximum_positive_target_ULP=int(ulp.max()),PN_RHS_calls=int(a['consumed']['counts'][:,0].sum()))
 return out
def signal_metrics(left,right,none,early,late):
 L=np.asarray(left,dtype=float).reshape(len(left),-1);R=np.asarray(right,dtype=float).reshape(len(right),-1);Z=np.asarray(none,dtype=float).reshape(len(none),-1);d=(L-R)/2;c=(L+R)/2-Z
 e=d[early].mean(axis=0);l=d[late].mean(axis=0);en=float(np.linalg.norm(e));ln=float(np.linalg.norm(l));u=e/en if en>0 else np.zeros_like(e);proj=d@u;dn=np.linalg.norm(d,axis=1);cn=np.linalg.norm(c,axis=1)
 def window(w):return dict(mean_lateral_norm=float(dn[w].mean()),norm_mean_lateral_vector=float(np.linalg.norm(d[w].mean(axis=0))),mean_common_norm=float(cn[w].mean()),mean_projection_on_early=float(proj[w].mean()),mean_L_minus_none_norm=float(np.linalg.norm(L[w]-Z[w],axis=1).mean()),mean_R_minus_none_norm=float(np.linalg.norm(R[w]-Z[w],axis=1).mean()),mean_relative_lateral_norm=float((dn[w]/np.maximum(np.linalg.norm(Z[w],axis=1),np.finfo(float).tiny)).mean()))
 return dict(early=window(early),late=window(late),projection_late_over_early=None if en==0 else float(l@u/en),norm_late_over_early=None if en==0 else ln/en,cosine_early_late=None if en==0 or ln==0 else float(e@l/(en*ln)),late_fraction_positive_early_projection=None if en==0 else float(np.mean(proj[late]>0)),integral_lateral_norm_11_to_384=float(dn[10:].sum()*.001),integral_signed_early_projection_11_to_384=None if en==0 else float(proj[10:].sum()*.001),scope='Descriptive source contrast and 1ms-sampled Riemann summary, not actual integrated synaptic dose. First/last are observed CSR calls including predictor/rejected evaluations; first interval also includes8warmup calls. Within-run template, not independent biological evidence or a motor decoder.'),dict(lateral_norm=dn,common_norm=cn,projection_early=proj)
def compute(d):
 p=d['plan'];need(p['analysis_window_ms']==[257,384] and p['early_window_ms']==[51,89] and p['prefix_ms']==10,'windows');criteria=p['criteria'];need(criteria['neural_half_L_minus_R_q_min']==1.6e-5 and criteria['yaw_half_L_minus_R_deg_s_min']==.02,'thresholds');need(criteria['stage4'] is False and criteria['stage5'] is False,'scope')
 need(p['arms']=={'qual':{'source':'none','duration_ms':2},**{'natural_'+s:{'source':s,'duration_ms':384} for s in ['none','L','R']}},'exposure design')
 early=slice(50,89);late=slice(256,384);allmetrics={};means={};curves={}
 for name,spec in p['arms'].items():
  a=d['arms'][name];allmetrics[name]=check_arm(a,spec,d['anchor'],d['terminals'],name)
  for k in ['ids','rows','ORN_ids','ORN_rows']:need(np.array_equal(a['panel'][k],d['reference']['qualification56']['panel'][k]),'panel identity')
  if name=='qual':
   q=d['reference']['qualification56']
   for k in ['trace','consumed','neural','panel','observer']:equal_subset(a[k],q[k],'qualification/'+k)
   for k in ['owners','events']:need(a[k]==q[k],'qualification/'+k)
   continue
  ref=d['reference']['parent_'+spec['source']];equal_subset(a['trace'],ref['trace'],name+'/54',89)
  t=a['trace'];n=a['neural'];y=np.rad2deg(t['command_yaw_rate_rad_s']);dn=t['DN_q_actual'][:,2]-t['DN_q_actual'][:,3];targets=endpoint(a['observer'],384)
  er=np.stack([angle_error(t['qpos'],s) for s in d['anchor']['source_geometry']],axis=1);curves[name]=dict(yaw_deg_s=y,DNb_difference=dn,error_signed_deg=er,heading_change_deg=np.rad2deg(yaw(t['qpos'])-yaw(d['anchor']['initial_qpos'])))
  means[name]=dict(mean_yaw_deg_s=float(y[late].mean()),mean_DNb_difference_q=float(dn[late].mean()),mean_forward_mm_s=float(t['command_forward_mm_s'][late].mean()),mean_DNg_target_q=float(targets[late].mean()),final_error_L_R_deg=np.abs(er[-1]).tolist(),physical_yaw_change_deg=float(curves[name]['heading_change_deg'][-1]),physical_displacement_xy_mm=float(np.linalg.norm(10*(t['qpos'][-1,:2]-d['anchor']['initial_qpos'][:2]))),ORN_total_mean_Hz=float(n['nominal_ORN_Hz'][late].sum(axis=1).mean()))
 Z,L,R=[d['arms']['natural_'+s] for s in ['none','L','R']]
 for a in [L,R]:
  for k,v in Z['trace'].items():
   # This field records the counterfactual concentration at each source even
   # while the10ms basal gate is closed. It is checked against each geometry
   # above; consumed concentration and all organism traces remain identical.
   if k=='spatial_geometric_concentration':continue
   need(np.array_equal(a['trace'][k][:10],v[:10]),'common10ms prefix '+k)
 getters={'nominal_ORN_Hz':lambda a:a['neural']['nominal_ORN_Hz'],'ORN_generic_first_RHS':lambda a:a['orn_consumed']['first'],'ORN_q_committed':lambda a:a['panel']['ORN_q'],'PN_generic_first_RHS':lambda a:a['consumed']['first'],'PN_generic_last_RHS':lambda a:a['consumed']['last'],'PN_q_committed':lambda a:a['neural']['PN_q'],'DM1_filters_committed':lambda a:a['trace']['ORN_filters'],'fine_PN_KC_gamma_and_additional':lambda a:np.concatenate([a['trace']['PN_gamma_nS'],a['trace']['PN_additional_nS']],axis=1)}
 signals={}
 for name,getter in getters.items():signals[name],curves[name]=signal_metrics(getter(L),getter(R),getter(Z),early,late)
 left,right,none=[means['natural_'+s] for s in ['L','R','none']];half_q=(left['mean_DNb_difference_q']-right['mean_DNb_difference_q'])/2;half_y=(left['mean_yaw_deg_s']-right['mean_yaw_deg_s'])/2;benefit=[none['final_error_L_R_deg'][i]-a['final_error_L_R_deg'][i] for i,a in enumerate([left,right])]
 contrast=dict(half_L_minus_R_DNb_q=half_q,half_L_minus_R_yaw_deg_s=half_y,common_yaw_source_minus_none_deg_s=(left['mean_yaw_deg_s']+right['mean_yaw_deg_s'])/2-none['mean_yaw_deg_s'],source_minus_none_yaw_L_R_deg_s=[v['mean_yaw_deg_s']-none['mean_yaw_deg_s'] for v in [left,right]],body_error_benefit_vs_none_L_R_deg=benefit,neural_and_yaw_material=bool(abs(half_q)>=1.6e-5 and abs(half_y)>=.02),direction_compatible_half_contrast=bool(half_y>0 and half_q>0),both_mirror_benefits_positive=bool(all(x>0 for x in benefit)))
 supported=contrast['neural_and_yaw_material'] and contrast['direction_compatible_half_contrast'] and contrast['both_mirror_benefits_positive']
 records=[d['arms'][n]['result'] for n in p['arms']];fields=['arm','status','attempted_ms','committed_ms','CPU_s','wall_s'];q=d['queue'];need(q['status']=='COMPLETE' and q['unclosed_reserved'] is None and q['arms']==[{k:r[k] for k in fields} for r in records],'queue records');cpu=sum(r['CPU_s'] for r in records);attempted=sum(r['attempted_ms'] for r in records);committed=sum(r['committed_ms'] for r in records)
 need(q['CPU_s']==cpu and q['attempted_ms']==attempted==1154 and q['committed_ms']==committed==1154,'exposure accounting');need(cpu<=p['queue_CPU_s']<=6000 and attempted<=p['CNS_ms']<=1200 and q['queue_wall_s']<=p['queue_wall_s']<=5000,'budget')
 result=dict(schema='natural57_recomputed_v1',means_primary_window=means,contrasts=contrast,signals=signals,arms=allmetrics,budget=dict(worker_CPU_s=cpu,queue_wall_s=q['queue_wall_s'],attempted_CNS_ms=attempted,committed_CNS_ms=committed),qualification_exact=True,prefix54_89ms_exact=True,screen_supported=bool(supported),classification='PROMETEDOR_NO_CONFIRMADO' if supported else 'DESCARTADO',classification_scope='Promotion from this fixed natural spatial diagnostic only, not impossibility of the project or absence of smaller effects.',stage4_admitted=False,stage5_admitted=False,limits=p['criteria']['confounds'])
 return result,curves
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=H);ap.add_argument('--write',action='store_true');ap.add_argument('--corruptions',action='store_true');args=ap.parse_args();start=time.process_time();d=load(args.root);result,curves=compute(d);tests=[]
 if args.corruptions:
  arm='natural_L';mutations=[('criterion',lambda x:x['plan']['criteria'].__setitem__('yaw_half_L_minus_R_deg_s_min',.0001)),('stage_flag',lambda x:x['plan']['criteria'].__setitem__('stage4',True)),('source_label',lambda x:x['arms'][arm]['spatial'].__setitem__('source_side','R')),('geometry',lambda x:x['arms'][arm]['trace']['spatial_geometric_concentration'].__setitem__((300,0),.1)),('transduction',lambda x:x['arms'][arm]['neural']['nominal_ORN_Hz'].__setitem__((300,0),999)),('PN_id',lambda x:x['arms'][arm]['neural']['PN_ids'].__setitem__(0,999)),('PN_RHS_count',lambda x:x['arms'][arm]['consumed']['counts'].__setitem__((300,0),0)),('ORN_RHS_error',lambda x:x['arms'][arm]['ORN']['counts_and_errors'].__setitem__((300,1),1)),('body_clock',lambda x:x['arms'][arm]['clock']['time_s'].__setitem__(300,0)),('neural_clock',lambda x:x['arms'][arm]['trace']['CNS_time_ns'].__setitem__(300,0)),('reader',lambda x:x['arms'][arm]['trace']['command_yaw_rate_rad_s'].__setitem__(300,123)),('baseline',lambda x:x['arms'][arm]['trace']['DN_baseline'].__setitem__((300,2),.7)),('RHS_context',lambda x:x['arms'][arm]['observer']['committed'].__setitem__(0,True)),('accepted_flag',lambda x:x['arms'][arm]['observer']['records'].__setitem__((0,138),2)),('nonfinite',lambda x:x['arms'][arm]['consumed']['first'].__setitem__((300,0),np.nan)),('horizon',lambda x:x['arms'][arm]['horizon'].__setitem__('dt_changed',True)),('budget',lambda x:x['queue'].__setitem__('attempted_ms',0))]
  for name,fn in mutations:
   changed=copy.deepcopy(d);fn(changed)
   try:compute(changed)
   except (ValueError,KeyError) as e:tests.append(dict(name=name,rejected=True,reason=str(e)))
   else:raise ValueError('corruption accepted '+name)
 out=args.root/'RESULTADOS.json';cost=dict(CPU_s=time.process_time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimized=not __debug__,tests=tests)
 if args.write:
  need(not out.exists(),'immutable result');out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');(args.root/'VERIFY_COST.json').write_text(json.dumps(cost,indent=2)+'\n');np.savez_compressed(args.root/'CURVES.npz',**{group+'__'+key:value for group,values in curves.items() for key,value in values.items()})
 else:need(json.loads(out.read_text())==result,'published result differs')
 print(json.dumps(dict(contrasts=result['contrasts'],budget=result['budget'],corruptions_rejected=len(tests),CPU_s=cost['CPU_s'])))
if __name__=='__main__':main()
