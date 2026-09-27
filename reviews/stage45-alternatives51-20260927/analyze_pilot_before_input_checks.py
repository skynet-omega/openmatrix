"""Recompute every51 contrast from saved arrays and prospectively frozen contract."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
H=Path(__file__).resolve().parent

def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compute(base):
 p=json.loads((base/'PILOT_PLAN.json').read_text());fr=json.loads((base/'PILOT_FREEZE.json').read_text());need(sha(base/'PILOT_PLAN.json')==fr['plan_sha256'],'Plan changed')
 need(p['duration_ms']==90 and p['analysis_window_ms']==[51,90] and p['criteria']['DNb05_directional_q_min']==1.6e-5 and p['criteria']['raw_yaw_deg_s_min']==.02,'Frozen decision contract')
 data={};checks={};sources={};window=slice(50,90);ids0=None
 for arm,setting in p['arms'].items():
  d=base/arm;r=json.loads((d/'RESULT.json').read_text());need(r['status']=='COMPLETE' and r['committed_ms']==90 and r['attempted_ms']==90 and r['initial_exact'],'Incomplete '+arm)
  sources[arm]={k:sha(d/k) for k in ['neural_and_inputs.npz','traces.npz','dng100_observed.npz','RESULT.json']}
  with np.load(d/'neural_and_inputs.npz') as z:a={k:z[k].copy() for k in z.files}
  with np.load(d/'traces.npz') as z:t={k:z[k].copy() for k in z.files}
  with np.load(d/'dng100_observed.npz') as z:o=z['records'][:,:128].reshape(-1,4,2,16);dnobs=z['ids'].copy()
  need(np.array_equal(dnobs,[10045,10056]),'DNg observer identity')
  need(a['q'].shape==(90,16) and a['JO_drive'].shape==(90,335) and a['nominal_Hz'].shape==(90,694),'Shapes '+arm)
  need(np.array_equal(a['ids'],[10045,10056,10118,10065,10442,10760,523769,10360,10888,11067,11074,512006,11960,11702,523640,10371]),'Neural identities '+arm)
  need(all(np.isfinite(v).all() for v in a.values()),'Nonfinite scientific input/outputs '+arm);need(np.isfinite(o).all(),'Nonfinite RHS')
  need(np.array_equal(a['q'][:,:4],t['DN_q_actual']),'Trace q alias')
  # Restore one-step motor latency and original reader; labels cannot supply outcome.
  need(np.array_equal(t['DN_q_usada'][1:],t['DN_q_actual'][:-1]),'One-step neural latency')
  baseline=t['DN_baseline'];used=t['DN_q_usada'];delta=used-baseline
  raw=np.tanh(250*(delta[:,2]-delta[:,3]))*np.deg2rad(5.)
  need(np.array_equal(raw,t['neural_yaw_unapplied_rad_s']),'Motor decoder changed')
  need(np.array_equal(t['command_yaw_rate_rad_s'],np.zeros(90)),'Yaw physically applied')
  need(np.array_equal(a['JO_drive'][:10],np.zeros((10,335))),'Air prefix')
  need(np.array_equal(a['nominal_Hz'][:10],np.broadcast_to(a['nominal_Hz'][0],(10,694))),'Odor prefix')
  if setting['odor']:need(np.max(a['nominal_Hz'][10:]-a['nominal_Hz'][0])>0,'Odor absent')
  else:need(np.array_equal(a['nominal_Hz'],np.broadcast_to(a['nominal_Hz'][0],(90,694))),'No-odor contaminated')
  if setting['air']!=0:need(a['JO_drive'][10:].max()>0,'Air absent')
  direction=a['q'][:,2]-a['q'][:,3]
  data[arm]={'a':a,'t':t,'direction':direction,'yaw':raw*180/np.pi,'mean_DNb_direction':float(direction[window].mean()),'mean_yaw_deg_s':float((raw*180/np.pi)[window].mean()),'DNg_target_max':float(o[:,:,:,11].max()),'mean_forward':float(t['forward_unclipped_mm_s'][window].mean())}
  checks[arm]=dict(DNg_RHS_cells=int(np.prod(o.shape[:3])),DN_observer_exact=True,reader_and_latency_exact=True,finite=True,restored_exact=True,physically_applied_yaw=False)
 with np.load(base/'saturation_population.npz') as z:
  eligible=z['eligible'].copy();need(eligible.shape==(166700,) and eligible.dtype==np.bool_,'Saturation population')
 reference=data['air0_odor0']['a'];
 for arm in data:need(np.array_equal(data[arm]['a']['initial_q'],reference['initial_q']),'Common initial published vector')
 air={}
 for odor in [0,1]:
  l=data[f'airL_odor{odor}'];r=data[f'airR_odor{odor}'];s=data[f'air0_odor{odor}']
  qhalf=(l['mean_DNb_direction']-r['mean_DNb_direction'])/2;yhalf=(l['mean_yaw_deg_s']-r['mean_yaw_deg_s'])/2
  air[str(odor)]={'half_L_minus_R_DNb_q':qhalf,'half_L_minus_R_yaw_deg_s':yhalf,'neural_material':bool(abs(qhalf)>=1.6e-5),'yaw_material':bool(abs(yhalf)>=.02),'both_material':bool(abs(qhalf)>=1.6e-5 and abs(yhalf)>=.02),'upwind_signed_expectation':'Fluid +left implies upwind right (negative yaw); evaluate sign only as screen, no movement claim.','mean_yaw_changes_L_R_vs_zero':[l['mean_yaw_deg_s']-s['mean_yaw_deg_s'],r['mean_yaw_deg_s']-s['mean_yaw_deg_s']],'bilateral_direction_sign_consistent_with_upwind':bool(l['mean_yaw_deg_s']<s['mean_yaw_deg_s'] and r['mean_yaw_deg_s']>s['mean_yaw_deg_s'])}
 gq=data['G_odor1']['mean_DNb_direction']-data['G_odor0']['mean_DNb_direction'];iq=data['I_odor1']['mean_DNb_direction']-data['I_odor0']['mean_DNb_direction'];gy=data['G_odor1']['mean_yaw_deg_s']-data['G_odor0']['mean_yaw_deg_s'];iy=data['I_odor1']['mean_yaw_deg_s']-data['I_odor0']['mean_yaw_deg_s']
 conductance=dict(G_odor_effect_q=gq,I_odor_effect_q=iq,G_minus_I_odor_effect_q=gq-iq,G_odor_effect_yaw_deg_s=gy,I_odor_effect_yaw_deg_s=iy,G_minus_I_odor_effect_yaw_deg_s=gy-iy,both_material=bool(abs(gq-iq)>=1.6e-5 and abs(gy-iy)>=.02))
 clipping={}
 for arm in ['G_odor0','G_odor1','I_odor0','I_odor1']:
  z=np.load(base/arm/'candidate_owner.npz');mask=z['S']>0
  need(np.isfinite(z['q0']).all() and np.isfinite(z['S']).all() and np.isfinite(z['gE0']).all(),'Owner finite');need(np.isfinite(z['min_raw'][mask]).all() and np.isfinite(z['max_raw'][mask]).all(),'Generic raw range finite')
  clipping[arm]={'base_rows_initialized':int(mask.sum()),'base_rows_raw_below0':int((z['min_raw'][mask]<0).sum()),'base_rows_raw_above1':int((z['max_raw'][mask]>1).sum()),'max_underflow':float(np.maximum(-z['min_raw'][mask],0).max()),'max_overflow':float(np.maximum(z['max_raw'][mask]-1,0).max()),'scope':'Base coefficient before later specialized overrides, not count of actual clipped final neuron states.'}
  parent='air0_odor'+arm[-1];mask_sat=data[arm]['a']['final_q']>=.999;new_sat=mask_sat & (data[parent]['a']['final_q']<.999);clipping[arm]['new_saturation_all_neurons_fraction']=float(new_sat.mean());clipping[arm]['new_saturation_eligible_fraction']=float(new_sat[eligible].mean());clipping[arm]['saturation_excluded']=bool(new_sat[eligible].mean()>.05)
 conductance['saturation_excluded']=any(v['saturation_excluded'] for v in clipping.values());conductance['promising_screen']=conductance['both_material'] and not conductance['saturation_excluded']
 result={'schema':'pilot51_verified_metrics_v1','checks':checks,'arms':{k:{j:v[j] for j in ['mean_DNb_direction','mean_yaw_deg_s','DNg_target_max','mean_forward']} for k,v in data.items()},'air':air,'odor_modulation_of_air_direction':{'q':air['1']['half_L_minus_R_DNb_q']-air['0']['half_L_minus_R_DNb_q'],'yaw_deg_s':air['1']['half_L_minus_R_yaw_deg_s']-air['0']['half_L_minus_R_yaw_deg_s']},'conductance':conductance,'clipping':clipping,'source_hashes':sources,'stage4_pass':False,'stage5_pass':False,'reason':'Neural transfer screen with yaw deliberately unapplied; no online navigation/replay or reserved mechanical perturbation.'}
 return result,data

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=H);ap.add_argument('--out',type=Path);args=ap.parse_args();r,_=compute(args.root)
 target=args.out or args.root/'PILOT_RESULTS.json';target.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:r[k] for k in ['air','odor_modulation_of_air_direction','conductance','stage4_pass','stage5_pass']}))
if __name__=='__main__':main()
