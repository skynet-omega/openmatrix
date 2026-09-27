"""Portable verification of recorded results; never imports a CNS module."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from analyze_pilot import compute
from shadow_readers import calculate
from verify_qualification_portable import verify as qualify
from analyze_clipping51 import compute as clipping
from analyze_air_pathway51 import compute as pathway
from analyze_body51 import compute as body
from test_air_body_units import check as units
H=Path(__file__).resolve().parent

def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(root):
 m=json.loads((root/'MANIFEST.json').read_text())
 for name,entry in m['files'].items():
  p=root/name;need(p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Manifest mismatch '+name)
 r,_=compute(root);expected=json.loads((root/'PILOT_RESULTS.json').read_text());need(r==expected,'Recomputed pilot metrics or verdict differ')
 s,_=calculate(root);need(s==json.loads((root/'SHADOW_READERS.json').read_text()),'Shadow reader recomputation')
 need(clipping(root)==json.loads((root/'CLIPPING_SUPPLEMENT.json').read_text()),'Clipping supplement recomputation')
 need(pathway(root)==json.loads((root/'AIR_PATHWAY_SUPPLEMENT.json').read_text()),'Air pathway and input confound')
 need(body(root)==json.loads((root/'BODY_SUPPLEMENT.json').read_text()),'Command versus body motion')
 need(units(root)==json.loads((root/'AIR_BODY_UNITS_TEST.json').read_text()),'Unit repair regression')
 v=json.loads((root/'VERDICT.json').read_text());a=pathway(root)
 material=all(r['air'][str(o)]['both_material'] and all(x['relay_response_present'] for x in a['observations'][str(o)].values()) for o in [0,1])
 need(v['classification']==('PROMETEDOR_NO_CONFIRMADO' if material or r['conductance']['promising_screen'] else 'DESCARTADO'),'Classification from actual contrasts')
 need(v['air_screen_material_with_pathway']==material and v['conductance_screen_promising']==r['conductance']['promising_screen'],'Verdict contrasts')
 need(v['stage4_pass'] is False and v['stage5_pass'] is False and v['physical_relative_air_units_valid'] is False and v['unit_correction_CNS_qualified'] is False,'Scientific scope flags')
 q=qualify(root);need(q['status']=='PASS','Qualification record replay')
 with np.load(root/'reader_projection.npz') as a:
  q=a['q'];times=a['times_ms'];delta=q-q[:,(times>=500)&(times<=1000)].mean(axis=1,keepdims=True);signed=(delta[:,:,:,0]-delta[:,:,:,1])*np.array([1,1,1,-1,1])
  values={'five_pair_median':np.median(signed,axis=2),'five_pair_equal_mean':signed.mean(axis=2),'DNb05_DNb06_opponent':signed[:,:,2:4].mean(axis=2),'DNa02_DNg13_actions':signed[:,:,[1,4]].mean(axis=2)}
  need(all(np.array_equal(v,a[k]) for k,v in values.items()),'Historic reader projection changed')
 with np.load(root/'anatomical_projection.npz') as z,np.load(root/'reader_projection.npz') as a:
  need(np.array_equal(a['ids'].ravel(),z['ids']),'Anatomical IDs');q=a['q'].reshape(4,30,10);dq=q-q[:,(a['times_ms']>=500)&(a['times_ms']<=1000)].mean(axis=1,keepdims=True)
  A=z['A'];v=np.stack([(dq*A[:,j]).sum(axis=2) for j in range(2)],axis=2);need(np.array_equal(v,z['full_channels']),'Anatomical projection changed')
 with np.load(root/'local_projection.npz') as a:
  result=json.loads((root/'LOCAL_RESULTS.json').read_text())
  for arm in ['sham','profile']:
   r=a[arm+'_record'];dx=a[arm+'_delta_input'];x=r[:,6]*(r[:,1]+r[:,4]-r[:,5]);target=lambda y:np.maximum(0,np.tanh(y));finite=(target(x+r[:,6]*dx)-target(x))/r[:,9];J=np.where(x>0,1-np.tanh(x)**2,0)*r[:,6]*dx/r[:,9]
   need(np.array_equal(finite,np.array(result['PN_JVP'][arm]['finite_RHS_delta'])) and np.array_equal(J,np.array(result['PN_JVP'][arm]['analytic_Jv'])),'Local PN JVP recomputation')
 # The replica verifies recorded criteria, not physiological interpretation or GPU restart.
 return dict(status='PASS',files=len(m['files']),scientific_arms=10,neural_ms_recorded=900,qualifications_ms=16,pilot_metrics_exact=True,shadow_readers_exact=True,local_JVP_exact=True,scope='Independent CPU recomputation from packaged recordings; no new CNS, no portable51 resume claim')
if __name__=='__main__':print(json.dumps(check(Path(sys.argv[1]) if len(sys.argv)>1 else H)))
