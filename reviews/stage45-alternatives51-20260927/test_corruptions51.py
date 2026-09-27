"""Real-data corruption checks; restore exact bytes after each local fixture."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from verify_complete51 import check
H=Path(__file__).resolve().parent

def main():
 root=Path(sys.argv[1]) if len(sys.argv)>1 else H
 if root.resolve()==Path('/home/daroch/AXIOMA_ASTRA/campanas/etapa45_alternativas_20260927_51'):raise ValueError('Run corruption fixtures only in a clean capsule extraction')
 cases=[];manifest=root/'MANIFEST.json';original_manifest=manifest.read_bytes()
 def update_manifest(name):
  m=json.loads(original_manifest);data=(root/name).read_bytes();m['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};manifest.write_text(json.dumps(m)+'\n')
 targets=[('PILOT_RESULTS.json','stage4_pass'),('PILOT_PLAN.json','criterion'),('airL_odor1/RESULT.json','status'),('SHADOW_READERS.json','shadow_flag')]
 for name,kind in targets:
  p=root/name;original=p.read_bytes();j=json.loads(original)
  if kind=='stage4_pass':j['stage4_pass']=True
  elif kind=='criterion':j['criteria']['raw_yaw_deg_s_min']=0.000001
  elif kind=='status':j['status']='FAILED'
  else:j['readers']['five_pair_median']['applied_to_body']=True
  try:
   p.write_text(json.dumps(j)+'\n');update_manifest(name)
   try:check(root)
   except ValueError:cases.append(kind)
   else:raise RuntimeError('Corruption accepted '+kind)
  finally:p.write_bytes(original);manifest.write_bytes(original_manifest)
 p=root/'airL_odor1/neural_and_inputs.npz';original=p.read_bytes()
 for kind in ['neuron_identity','neural_nonfinite','odor_removed']:
  with np.load(p) as z:a={k:z[k].copy() for k in z.files}
  if kind=='neuron_identity':a['ids'][2]+=1
  elif kind=='neural_nonfinite':a['q'][60,2]=np.nan
  else:a['nominal_Hz'][10:]=a['nominal_Hz'][0]
  try:
   np.savez_compressed(p,**a);update_manifest('airL_odor1/neural_and_inputs.npz')
   try:check(root)
   except ValueError:cases.append(kind)
   else:raise RuntimeError('Corruption accepted '+kind)
  finally:p.write_bytes(original);manifest.write_bytes(original_manifest)
 check(root);print(json.dumps({'corruptions_rejected':cases,'count':len(cases),'originals_restored_verified':True,'optimization':not __debug__}))
if __name__=='__main__':main()
