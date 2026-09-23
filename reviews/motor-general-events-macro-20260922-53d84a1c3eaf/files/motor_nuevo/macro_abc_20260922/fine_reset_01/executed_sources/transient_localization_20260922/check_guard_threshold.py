"""CUDA falsador: above-threshold peak needs no new upward crossing."""
import numpy as np,cupy as cp
from event_guard import GUARD
src=r'''
extern "C" __global__ void check(const double*v,const double*va,const double*vb,const double*observe,long long h,double*out){
 int n=blockIdx.x,i=threadIdx.x;if(i>=17)return;
 unsigned mask=0x1ffff;double rest=-60.,e=0.;
'''+GUARD+r'''
 if(i==0)out[n]=__shfl_sync(mask,e,0);
}
'''
# The guard uses warp collectives. All17 participating lanes must execute the
# final broadcast before lane0 writes, matching SpatialModel.trial.
src=src.replace('if(i==0)out[n]=__shfl_sync(mask,e,0);','double answer=__shfl_sync(mask,e,0);if(i==0)out[n]=answer;')
v=np.zeros((5,17));a=v.copy();b=v.copy();obs=np.zeros(17);obs[0]=1.
#1 already above at all samples;2 only middle sample;3 axon only;4 none.
v[0,0]=25;a[0,0]=27;b[0,0]=26
a[1,0]=21
v[2,7]=25;a[2,7]=27;b[2,7]=26
#5 threshold reached exactly.
v[4,0]=20
args=[cp.asarray(x) for x in (v,a,b,obs)];out=cp.zeros(5)
k=cp.RawKernel(src,'check',options=('--fmad=false',))
k((5,),(32,),(*args,np.int64(25000),out));actual=out.get()
if not np.array_equal(actual>1.,[True,True,True,False,True]):raise ValueError('Event guard failed threshold cases')
k((5,),(32,),(*args,np.int64(1562),out))
if np.any(out.get()!=0.):raise ValueError('Guard rejects declared limit')
print('PASS: already above, middle peak, axon, quiet, equality and declared limit')
