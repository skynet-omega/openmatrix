from pathlib import Path
import sys,json,copy,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import Engine
from model import require
from development import base
h=Path(__file__).resolve().parent
rows=[]
for algorithm in ('A','C'):
 for profile in ('fast','precise'):
  spec=json.loads((h/'examples/inhibited.json').read_text());e=Engine(spec,algorithm=algorithm,profile=profile)
  require(e.gpu.scale.dtype==np.float64 and e.gpu.mass.dtype==np.float64,'double buffer dtype')
  err=e.gpu.attempt(0,.0001)
  with e.gpu.stream:
   y=e.model.initial;fine=e.gpu.fine.get(stream=e.gpu.stream);coarse=e.gpu.coarse.get(stream=e.gpu.stream)
  rtol,atol=e.gpu.rtol,e.gpu.atol
  expected=float(np.max(abs(fine-coarse)/((15 if algorithm=='A' else 3)*(atol*e.model.scale+rtol*np.maximum(abs(y),abs(fine))))))
  require(abs(err-expected)<1e-10*max(1,expected),'GPU norm differs from intended floating scale')
  e=Engine(spec,algorithm=algorithm,profile=profile);e.advance(.1);require(e.scan('cells','v',[0,1])['values']==[0,0],'zero clamp failed')
  rows.append({'algorithm':algorithm,'profile':profile,'norm_error':abs(err-expected),'zero_clamp':True})
# Integer diagonal mass from JSON must mean numeric 2, not its IEEE bit interpretation.
spec=base(2,1);spec['mass']={'row':list(range(4)),'col':list(range(4)),'values':[2,2,3,3]}
e=Engine(spec);cpu=e.model.rhs(0,e.read());gpu=e.gpu.rhs_read(0,e.read());require(np.max(abs(cpu-gpu))<1e-11,'integer mass conversion')
print(json.dumps({'checks':rows,'integer_mass':True},indent=2))
