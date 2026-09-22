"""Independent NumPy/Radau checks of equations, recurrence and cache invalidation."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from recurrence import Engine
import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix
from scipy.integrate import solve_ivp
rng=np.random.default_rng(93817);n=16;w=rng.uniform(-.06,.08,(n,n));w[rng.random((n,n))<.4]=0
csr=csr_matrix(w);visual=np.arange(n)%3==0
D={'indptr':csr.indptr.astype(np.int64),'indices':csr.indices.astype(np.int32),'weights':csr.data,'caps':rng.uniform(1,2,n),'visual':visual,'tau':rng.uniform(.012,.029,n),'gain':rng.uniform(.7,1.5,n),'theta':np.full(n,.03),'drive':np.full(n,.2),'photo':np.zeros(n),'scale':np.array(1/30),'connected':np.array(True)}
y0=np.r_[rng.uniform(.4,.75,n),rng.uniform(.25,.5,n)]
def numpy_rhs(t,y,photo=0):
 a=(w*(y[n:]*D['caps'])[None,:]).sum(axis=1)
 target=np.maximum(0,np.tanh(D['gain']*(a+D['drive']-D['theta'])));rate=1/D['tau']
 av=(np.maximum(w,0)*y[n:][None,:]/30).sum(axis=1)+photo
 bv=(-np.minimum(w,0)*y[n:][None,:]/30).sum(axis=1)
 target[visual]=((.25+av)/(1+av+bv))[visual];rate[visual]=((1+av+bv)/D['tau'])[visual]
 release=y[:n].copy();release[visual]=np.clip(2*y[:n][visual]-.375,0,1)
 return np.r_[rate*(target-y[:n]),200*(release-y[n:])]
e=Engine(D,y0);e.rhs(e.y,e.k[0]);error=float(np.max(abs(cp.asnumpy(e.k[0])-numpy_rhs(0,y0))))
if error>1e-12:raise RuntimeError('Independent equation mismatch')
policy=json.loads((Path(__file__).parent/'contract.json').read_text());rows=[]
for condition in ['tonic','photo_pulse']:
 reference=[y0.copy()];state=y0.copy()
 for i in range(10):
  photo=.2 if condition=='photo_pulse' and 2<=i<6 else 0
  sol=solve_ivp(lambda t,y:numpy_rhs(t,y,photo),(i*.0005,(i+1)*.0005),state,method='Radau',rtol=1e-11,atol=1e-13)
  if not sol.success:raise RuntimeError('Independent Radau failed')
  state=sol.y[:,-1];reference.append(state)
 reference=np.asarray(reference)
 for spec in policy['candidates']:
  _,actual=e.run(spec,condition,.005,.0005);error2=float(np.max(abs(actual-reference)))
  if error2>1e-4:raise RuntimeError(f'Independent recurrence mismatch {spec}: {error2}')
  rows.append({'condition':condition,'id':spec['id'],'max_error_vs_Radau':error2})
spec=policy['candidates'][2]
_,a=e.run(spec,'tonic',.005,.0005);_,b=e.run(spec,'photo_pulse',.005,.0005);_,a2=e.run(spec,'tonic',.005,.0005)
if not np.array_equal(a,a2) or np.array_equal(a,b):raise RuntimeError('A-B-A graph state invalidation failed')
# Threshold zero must reproduce dense coefficients after arbitrary source changes.
e.reset({'method':'incremental_rk4'});perturbed=cp.asarray(rng.uniform(0,1,n));e.coefficient(perturbed,0.);t=cp.asnumpy(e.target);r=cp.asnumpy(e.rate);e.coefficient(perturbed)
cache_error=max(float(np.max(abs(t-cp.asnumpy(e.target)))),float(np.max(abs(r-cp.asnumpy(e.rate)))))
if cache_error>1e-12:raise RuntimeError('CSC update differs from dense operator')
result={'independent_rhs_max_error':error,'rows':rows,'A_B_A_exact':True,'incremental_zero_threshold_error':cache_error,'limitations':'16 synthetic coupled cells, no biological or full-organism validation'}
print(json.dumps(result,indent=2))
