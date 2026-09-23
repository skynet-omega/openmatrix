"""Prospective local event-resolution guard; unchanged sampled detector."""
from pathlib import Path
import sys,types,hashlib
T=Path(__file__).resolve().parent
sys.path[:0]=[str(T.parent/'causal_runtime_20260922'),str(T.parent/'native_hybrid_20260922/vendor')]
import device_cell
original_source=device_cell.source
GUARD=r'''
  // A model-declared observation needs temporal resolution near its event set.
  // This is not a continuous root finder or an error certificate for hidden peaks.
  double som0=v[n*17+i]*observe[i],soma=va[n*17+i]*observe[i],somb=vb[n*17+i]*observe[i];
  for(int o=16;o>0;o/=2){
   double x=__shfl_down_sync(mask,som0,o),a=__shfl_down_sync(mask,soma,o),b=__shfl_down_sync(mask,somb,o);
   if(i+o<17){som0+=x;soma+=a;somb+=b;}
  }
  bool near=(i==0&&(som0+rest>=-40.||soma+rest>=-40.||somb+rest>=-40.));
  if(i>=5)near|=(v[n*17+i]+rest>=-40.||va[n*17+i]+rest>=-40.||vb[n*17+i]+rest>=-40.);
  if(__any_sync(mask,near)&&h>1562)e=fmax(e,double(h)/1562.);
'''
def guarded_source():
 code=original_source();needle='  return __shfl_sync(mask,e,0);'
 if code.count(needle)!=1:raise ValueError('Unexpected kernel template')
 # All lanes receive the near-event condition before the final error broadcast.
 return code.replace(needle,GUARD+'\n'+needle)

class EventGuardCell(device_cell.DeviceCell):
 def __init__(self,*args,**kw):
  old=device_cell.source;device_cell.source=guarded_source
  try:super().__init__(*args,**kw)
  finally:device_cell.source=old
  self.report.update(event_guard_step_ns=1562,event_guard_threshold_mV=-40.,event_detection='unchanged sampled local maximum; no continuous-root claim')

def attach(session):
 if session.profile!='causal_cuda' or session.cell.core is not None:raise ValueError('Guard requires an unstarted causal session')
 b=session.brain._spatial_batch
 while hasattr(b,'base'):b=b.base
 def advance(self,*args,**kw):
  # The physical wrapper installs its publisher only on the first advance.
  if session.cell.core is None:
   session.cell.core=EventGuardCell(self,session.events)
   session.cell.report=session.cell.core.report
  return session.cell.core.advance(self,*args,**kw)
 b.advance=types.MethodType(advance,b)
 path=Path(__file__).resolve()
 session.implementations['membrane']={'module':__name__,'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 # RuntimeSession retains the original adapter's undo: it restores the same
 # pre-session method, including when construction or an epoch fails.
