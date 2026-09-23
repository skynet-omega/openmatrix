"""Counterexamples on actual saved states, without inventing new tolerances."""
from pathlib import Path
import sys,json,tempfile,shutil
import numpy as np
H=Path(__file__).resolve().parent;P=H.parents[1]/'campanas/etapa3_motor_nuevo_20260922';sys.path.insert(0,str(P))
from qualify_transport import qualify
source=H.parent/'causal_runtime_20260922/device20_01';checks=[]
for field in ('spatial_voltage','body_velocity'):
 with tempfile.TemporaryDirectory() as t:
  target=Path(t)
  for name in ('brain_final.json','brain_final.npz','body_final.npz','traces.npz'):shutil.copy2(source/name,target/name)
  file=target/('brain_final.npz' if field=='spatial_voltage' else 'body_final.npz')
  with np.load(file,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
  if field=='spatial_voltage':
   d=json.loads((target/'brain_final.json').read_text());key=d['kc_spatial_state']['delta']['__array__'];arrays[key][0,0]+=1000.
  else:arrays['qvel'][0]+=1000.
  np.savez_compressed(file,**arrays)
  result=qualify(source,target)
  if result['full_state_qualified']:raise RuntimeError('Unbounded change admitted')
  checks.append({'field':field,'deliberate_change':1000.,'historical_screen_pass':result['historical_partial_screen']['screen_pass'],'full_state_qualified':result['full_state_qualified']})
(H/'COVERAGE_AFTER.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))

if not qualify(source,source)["full_state_qualified"]:raise RuntimeError("Identical-state positive control rejected")
