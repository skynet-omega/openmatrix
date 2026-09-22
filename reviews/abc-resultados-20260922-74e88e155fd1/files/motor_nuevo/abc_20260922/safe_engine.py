"""Prospective fail-closed entry point. Frozen unguarded sources remain evidence."""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from guarded_checks import Guarded
from recurrence import write
class SafeEngine(Guarded):
 def require_valid(self):
  flags=int(self.flags.get()[0])
  if flags:raise FloatingPointError(f'Intermediate numerical guard flags={flags}; C negative-source stages are unsupported')
 def run(self,spec,condition,duration,sample,out=None):
  result,states=super().run(spec,condition,duration,sample,None)
  self.require_valid();result['persistent_guard_flags']=0
  if out:
   out=Path(out);out.mkdir(exist_ok=False);np.savez_compressed(out/'states.npz',time_s=result['samples_s'],state=states);write(out/'timing.json',result)
  return result,states
