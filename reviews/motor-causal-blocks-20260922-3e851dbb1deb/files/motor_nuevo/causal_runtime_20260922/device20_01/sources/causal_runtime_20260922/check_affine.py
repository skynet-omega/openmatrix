from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from affine_ports import AffinePorts
H=Path(__file__).resolve().parent;records=[]

def check(name,a,t,j,y,prefixes):
 p=AffinePorts(a,t,j,y);maximum=0.;outputs=[]
 for end in prefixes:
  reference=y.copy()
  for k in range(len(t)-1):
   if t[k]>end:break
   reference+=j[:,k]
   dt=min(end,t[k+1])-t[k]
   for b in range(len(y)):reference[b]=expm(dt*a[b,k])@reference[b]
   if end<t[k+1]:break
  actual=p.prefix(end).get();error=float(abs(actual-reference).max());maximum=max(maximum,error);outputs.append(actual.tolist())
  if not np.allclose(actual,reference,atol=1e-11,rtol=1e-11):raise RuntimeError(name+' failed at '+str(end))
 records.append(dict(name=name,max_abs=maximum,prefixes=prefixes,outputs=outputs));return outputs

# Exact three-filter response, including a late event at fractional nanosecond.
for tau in ([125e-6]*3,[125e-6,126e-6,124e-6],[3e-5,.003,.01]):
 L=np.diag(-1/np.array(tau));L[1,0]=1/tau[1];L[2,1]=1/tau[2]
 t=np.array([0.,112500.375e-9,125e-6]);a=np.tile(L,(1,2,1,1));j=np.zeros((1,2,3));j[0,1,0]=.5
 result=check('late_'+str(tau),a,t,j,np.zeros((1,3)),[0.,100e-6,112500.375e-9,120e-6,125e-6])
 if result[2][0]!=[.5,0.,0.]:raise RuntimeError('Right-continuous jump at prefix lost')
# Equal conductance areas, reversed receptor order: must preserve noncommutation.
g=.8;E1=-1.;E2=1.;L1=np.array([[-g,g*E1],[0.,0.]]);L2=np.array([[-g,g*E2],[0.,0.]])
a=np.array([[L1,L2],[L2,L1]]);t=np.array([0.,1.,2.]);j=np.zeros((2,2,2));y=np.array([[0.,1.],[0.,1.]])
outputs=check('noncommuting_receptors',a,t,j,y,[.5,1.,1.5,2.]);difference=outputs[-1][0][0]-outputs[-1][1][0];expected=(1-np.exp(-g))**2*(E2-E1)
if abs(difference-expected)>1e-11:raise RuntimeError('Event order was lost')
# A larger model can be supplied without editing the runtime.
rng=np.random.default_rng(22);L=rng.uniform(0,.5,(7,7));L-=np.eye(7)*3.
check('new_seven_state_model',np.tile(L,(3,2,1,1)),np.array([0.,.2,.7]),rng.uniform(-.1,.1,(3,2,7)),rng.uniform(0,1,(3,7)),[0.,.1,.2,.4,.7])
report={'status':'PASS','checks':records,'noncommuting_difference':difference,'expected_difference':expected,'scope':'Generic declared affine state blocks and receivers only. Not integrated into nonlinear full-organism CNS; no full-engine speed claim. Taylor truncation bounded by exponential series; FP64 arithmetic tested, not universally certified.'}
(H/'AFFINE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','max_abs':max(r['max_abs'] for r in records),'checks':len(records)}))
