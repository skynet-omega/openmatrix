"""Portable numerical reconstruction and deliberate corruption checks, no CNS."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,resource,copy,importlib.util
import numpy as np
from unittest.mock import patch
import verify54 as v

H=Path(__file__).resolve().parent

def check(root,corruptions=False):
 start=time.process_time();campaign=Path(root);root=campaign/'repair02'
 expected=json.loads((root/'RESULTADOS.json').read_text());actual,_=v.compute(root)
 v.need(actual==expected,'published quantities/verdict differ')
 donor_path=campaign/'research11/verify_evidence.py';donor_result=None
 if donor_path.is_file():
  spec=importlib.util.spec_from_file_location('donor11_verifier',donor_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);donor_result=module.verify(donor_path.parent)
 cross=json.loads((campaign/'CROSS52_54.json').read_text());computed={}
 for law,old,new in [('parent','air0_odor0','parent_none'),('I','I_odor0','I_none')]:
  oldpath=campaign/'research11/reference52'/old/'traces.npz';newpath=root/new/'traces.npz';aa=v.arrays(oldpath);bb=v.arrays(newpath)
  w0=aa['neural_yaw_unapplied_rad_s'][:89];w1=bb['neural_yaw_raw_rad_s'];v0=aa['command_forward_mm_s'][:89];v1=bb['command_forward_mm_s'];delta=bb['DN_q_actual']-aa['DN_q_actual'][:89];different=np.flatnonzero(np.any(delta!=0,axis=1))
  computed[law]={'window_ms':[51,89],'yaw_without_application_deg_s':float(np.rad2deg(w0[50:]).mean()),'yaw_with_application_deg_s':float(np.rad2deg(w1[50:]).mean()),'last_yaw_without_with_deg_s':[float(np.rad2deg(w0[-1])),float(np.rad2deg(w1[-1]))],'forward_without_with_mm_s':[float(v0[50:].mean()),float(v1[50:].mean())],'max_neural_DN_difference':float(abs(delta).max()),'first_neural_DN_difference_ms':int(different[0]+1) if len(different) else None,'hashes':{'52':v.sha(oldpath),'54':v.sha(newpath)}}
 v.need(computed==cross['results'] and cross['new_CNS_ms']==0,'descriptive cross-campaign quantities')
 if (campaign/'MANIFEST.json').exists():
  m=json.loads((campaign/'MANIFEST.json').read_text())
  for name,item in m['files'].items():
   f=campaign/name;v.need(f.stat().st_size==item['bytes'] and v.sha(f)==item['sha256'],'manifest member changed: '+name)
 rejected=[]
 if corruptions:
  original=v.arrays
  def trial(label,action):
   try:action()
   except ValueError as e:rejected.append({'case':label,'error':str(e)})
   else:raise ValueError('accepted corruption: '+label)
  cases=[('motor_latency','parent_none/traces.npz','DN_q_usada',(17,0),.001),
   ('yaw_units','parent_none/traces.npz','command_yaw_rate_rad_s',(17,),.001),
   ('control_odor_contamination','parent_none/traces.npz','spatial_concentration_used',(5,0),.1),
   ('source_geometry','parent_L/traces.npz','spatial_geometric_concentration',(17,0),.1),
   ('ORN_nominal_amount','I_L/neural_and_inputs.npz','nominal_Hz',(17,0),1.),
   ('anatomical_identity','parent_L/neural_and_inputs.npz','ORN_ids',(0,),1)]
  for label,rel,key,index,delta in cases:
   target=root/rel
   def damaged(path):
    a=original(path)
    if Path(path)==target:a[key][index]+=delta
    return a
   with patch.object(v,'arrays',damaged):trial(label,lambda:v.compute(root))
  read=Path.read_text
  for label,relative,mutate in [
   ('source_context','parent_L/SPATIAL_OWNER_FINAL.json',lambda d:d.update(source_side='R')),
   ('materiality','PILOT_PLAN.json',lambda d:d['criteria'].update(neural_half_L_minus_R_q_min=0.)),
   ('stage_in_contract','PILOT_PLAN.json',lambda d:d['criteria'].update(stage4=True))]:
   target=root/relative
   def changed(path,*args,**kw):
    s=read(path,*args,**kw)
    if path==target:
     d=json.loads(s);mutate(d);return json.dumps(d)
    return s
   with patch.object(Path,'read_text',changed):trial(label,lambda:v.compute(root))
  wrong=copy.deepcopy(expected);wrong['stage4_admitted']=True
  trial('published_stage_flag',lambda:v.need(wrong==actual,'published quantities/verdict differ'))
 return {'status':'PASS','exact_result_reconstruction':True,'preparatory_evidence':donor_result,'corruptions_rejected':rejected,'CPU_s':time.process_time()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'new_CNS_ms':0,'scope':'Analysis of recorded data; no new organism simulation or portable GPU resume.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=H);p.add_argument('--corruptions',action='store_true');a=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(55,60));print(json.dumps(check(a.root.resolve(),a.corruptions),indent=2))
if __name__=='__main__':main()
