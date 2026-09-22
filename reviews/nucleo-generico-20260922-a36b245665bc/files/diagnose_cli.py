from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import Engine
h=Path(__file__).resolve().parent
for start in ('direct','mutation'):
 spec=json.loads((h/'examples/inhibited.json').read_text())
 if start=='direct':e=Engine(spec,profile='fast')
 else:
  e=Engine(json.loads((h/'examples/mixed.json').read_text()),profile='fast');e.advance(.02);e.replace(spec)
 try:e.advance(.1)
 except Exception as ex:
  g=e.gpu
  with g.stream:
   print(start,'FAIL',type(ex).__name__,ex,'t,h',e.t,e.next_h,'clock',g.clock.get(stream=g.stream),'flag',g.flag.get(stream=g.stream),flush=True)
   for name in ['x','fine','coarse','mid','par','inputs','outputs','scale','mass']:
    a=getattr(g,name).get(stream=g.stream);idx=np.flatnonzero(~np.isfinite(a));print(name,'nonfinite',idx[:20], 'min/max',np.nanmin(a),np.nanmax(a),flush=True)
  continue
 print(start,'PASS',e.t,flush=True)
