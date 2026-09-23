"""Independent decimal replay from saved FP64 operands, not report numbers."""
from pathlib import Path
from decimal import Decimal,localcontext
import json,numpy as np
R=Path(__file__).resolve().parent

def calculate():
 with np.load(R/'domain_01/failure_arrays.npz',allow_pickle=False) as z:
  bad=np.flatnonzero(~np.isfinite(z['fine'])|(z['fine']<0)|(z['fine']>1))
  if len(bad)!=1 or not np.array_equal(bad,z['indices']):raise ValueError('Unexpected domain evidence')
  slot=np.flatnonzero(z['port_qr']==bad[0])
  if len(slot)!=1:raise ValueError('Owner not uniquely mapped')
  slot=int(slot[0]);t=float(z['clock'].sum());q=float(z['q0'][slot]);tau=float(z['tau'][slot]);events=[(float(et),float(ej)) for et,row,ej in zip(z['times'],z['rows'],z['jumps']) if row==slot and et<=t]
  with localcontext() as ctx:
   ctx.prec=90;D=Decimal.from_float;answer=D(q)*(-D(t)/D(tau)).exp()
   for et,j in events:answer+=D(j)*(-(D(t)-D(et))/D(tau)).exp()
   if not answer>1:raise ValueError('Reported operand loss not reproduced')
   result={'coordinate':int(bad[0]),'port_slot':slot,'q0':q,'tau':tau,'time':t,'events':events,'decimal_answer_from_FP64_inputs':str(answer),'recorded_GPU':float(z['fine'][bad[0]]),'pre_projection_valid':all(np.all((z[f'out_{i}_pre']>=0)&(z[f'out_{i}_pre']<=1)) for i in range(3))}
 return result
if __name__=='__main__':print(json.dumps(calculate(),indent=2))
