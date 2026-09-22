"""Deliberate corruption of physical evidence, never runner PASS flags."""
from pathlib import Path
import copy,json,shutil,sys,tempfile
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/daroch/AXIOMA_FLYWIRE/matrix/src')
from session_io import write_state
from verify_transport import verify,read_state
reference=HERE/'transport_resident_left_01';state=read_state(reference/'brain_final')
checks=[]
with tempfile.TemporaryDirectory(prefix='corrupt-',dir=HERE) as td:
 folder=Path(td)
 for name in ['traces.npz','body_final.npz']:shutil.copy2(reference/name,folder/name)
 cases=[]
 x=copy.deepcopy(state);x['pn_online_state']['receptors']['GABA']['history'][0]['left'][0]+=.01;cases.append(('delay_driver',x))
 x=copy.deepcopy(state);x['time_ns']+=1;cases.append(('clock',x))
 x=copy.deepcopy(state);x['kc_spatial_manifest']['enabled']=not x['kc_spatial_manifest']['enabled'];cases.append(('biological_flag',x))
 x=copy.deepcopy(state);x['state'][0]=np.nan;cases.append(('nonfinite',x))
 for name,value in cases:
  write_state(folder/'brain_final',value)
  try:result=verify(reference,folder);detected=not result['screen_pass']
  except (ValueError,TypeError,KeyError):detected=True
  if not detected:raise RuntimeError('Corruption accepted: '+name)
  checks.append({'case':name,'detected':True})
(HERE/'VERIFIER_CORRUPTION.json').write_text(json.dumps({'checks':checks,'python_optimized':not __debug__},indent=2)+'\n')
print(json.dumps(checks))
