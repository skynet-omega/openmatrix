"""Independent algebra/identity/consumption gate using saved qualification data."""
from pathlib import Path
import json,time
import numpy as np
H=Path(__file__).resolve().parent;C=H.parent/'etapa45_composicion_20260927_48/sham/final_state'
def need(x,m):
 if not x:raise ValueError(m)
def candidate_check(mode,expected_initial):
 d=H/('qual_'+mode);z=np.load(d/'candidate_owner.npz');a=np.load(d/'dng100_observed.npz');rows=a['rows'];r=a['records'][:,:128].reshape(-1,4,2,16)
 need(np.array_equal(z['q0'],expected_initial),'q0 not restored native state')
 S=z['S'][rows];g0=z['gE0'][rows];q0=z['q0'][rows];X=r[:,:,:,2]+np.maximum(r[:,:,:,4],0);Y=-r[:,:,:,3]+r[:,:,:,5]+np.maximum(-r[:,:,:,4],0)
 need(np.array_equal(S,X[0,0]+Y[0,0]),'S not first actually recorded RHS')
 need(np.array_equal(g0,X[0,0]/S),'Initial E fraction not first RHS')
 E=X/S;I=Y/S;EL=2*q0-g0;total=1+E+I
 if mode=='conductance':raw=(EL+E)/total;rate=total/(2*r[:,:,:,9])
 else:raw=(EL+E*(1-q0)-I*q0+q0)/2;rate=1/r[:,:,:,9]
 target=np.clip(raw,0,1);need(np.array_equal(target,r[:,:,:,7]),'GPU candidate target disagrees with CPU formula')
 need(np.array_equal(rate,r[:,:,:,8]),'GPU candidate rate disagrees with CPU formula')
 need(np.array_equal(r[:,:,:,7],r[:,:,:,11]) and np.array_equal(r[:,:,:,8],r[:,:,:,12]),'DN generic law unexpectedly overridden')
 need(np.isfinite(r).all(),'Nonfinite RHS')
 for k in ['q0','S','gE0']:need(np.isfinite(z[k]).all(),'Nonfinite candidate owner')
 return dict(observed_DNg_RHS_cells=int(np.prod(r.shape[:3])),CPU_target_rate_exact=True,q0_all_166700_exact=True,S_first_recorded_DNg_RHS_exact=True,calibration_scope='S is set by first construction RHS; first consumed recorded RHS equality verified for DNg100; other rows share same initialization code, not separately captured.',raw_target_outside_0_1=float(np.maximum(np.maximum(-raw,raw-1),0).max()))
def main():
 cpu=time.process_time();p=json.loads((H/'PILOT_PLAN.json').read_text());j=json.loads((C/'session.json').read_text())
 with np.load(C/'session.npz') as z:initial=z[j['hybrid']['state']['__array__']][:166700]
 reports={};total=0
 for arm in p['qualifications']:
  d=H/('qual_'+arm);r=json.loads((d/'RESULT.json').read_text());need(r['status']=='COMPLETE' and r['initial_exact'] and r['committed_ms']==4,'Incomplete qualification '+arm);total+=r['attempted_ms']
  with np.load(d/'neural_and_inputs.npz') as a:
   need(all(np.isfinite(a[k]).all() for k in a.files),'Nonfinite publication')
   need(np.array_equal(a['JO_drive'][:2],np.zeros_like(a['JO_drive'][:2])),'Prefix drive')
   if arm=='identity':need(np.array_equal(a['JO_drive'],np.zeros_like(a['JO_drive'])),'Identity added current')
   elif arm=='air':need(a['JO_drive'][2:].max()>0,'Positive air control absent')
  reports[arm]=candidate_check(arm,initial) if arm in ['conductance','current'] else {'consumption_checked_each_ms_in_runner':True}
 need(total==16,'Qualification budget')
 # Candidate parameters must be common between G and I before divergent dynamics.
 with np.load(H/'qual_conductance/candidate_owner.npz') as g,np.load(H/'qual_current/candidate_owner.npz') as i:
  need(all(np.array_equal(g[k],i[k]) for k in ['q0','S','gE0']),'G/current adoption differs')
 out=dict(status='PASS',checks=reports,qualification_attempted_ms=total,adoption_G_I_all_rows_exact=True,CPU_s=time.process_time()-cpu)
 (H/'QUALIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
