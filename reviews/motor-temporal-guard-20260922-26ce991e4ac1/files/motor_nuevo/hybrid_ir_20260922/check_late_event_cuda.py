"""Independent CUDA reproduction of the external late-event counterexample."""
from pathlib import Path
from types import SimpleNamespace
import sys,time,json,numpy as np,cupy as cp
from scipy.linalg import expm
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path[:0]=[str(R.parent/'macro_abc_20260922'),str(ROOT/'campanas/etapa3_motor_nuevo_20260922')]
from graph_core import NativeGraph
from reset_ports import ResetFilterPorts
from dependency_ir import EventProgram,Read

def main():
 out=Path(sys.argv[1]) if len(sys.argv)>1 else R/'late_event_cuda_01';out.mkdir(exist_ok=False)
 start=time.perf_counter();h=tau=125e-6;event=.9*h;jump=.5
 p=ResetFilterPorts([0],[1]);p.update(SimpleNamespace(q=np.zeros(1),s=np.zeros(1),tau=np.array([tau]),ts=tau,times=[event],rows=[0],jumps=[jump],post_values=[None]))
 rates=cp.asarray([0.,0.,1/tau])
 def coeff(z):
  target=z.copy();target[2]=z[1];return target,rates
 program=EventProgram(3,[Read('own',[0,1,2],[0,1,2]),Read('release',[0],[1]),Read('filtered',[1],[2])],[0,1])
 continuity=program.classify([0])
 if not continuity['continuous_free_rhs']:raise ValueError('Fixture must have continuous free RHS')
 results={};states={}
 for mode,cuts in (('uncertified',None),('event_cut',np.asarray([event]))):
  g=NativeGraph(np.zeros(3),coeff,rtol=1e-5,atol=1e-7,norm_size=3,project=p.project,native_library=R.parent/'native_hybrid_20260922/libgraph_control_v2.so')
  try:
   next_step,counts,error=g.advance(125000,125000,100,125000,budget=20,boundaries=cuts)
   g.stream.synchronize();states[mode]=g.x.get();results[mode]={'state':states[mode].tolist(),'accepted':counts[0],'rejected':counts[1],'maximum_estimator':error}
  finally:g.close()
 A=np.array([[-1.,0.,0.],[1.,-1.,0.],[0.,1.,-1.]])/tau
 exact=expm(A*(h-event))@np.array([jump,0.,0.]);states['exact']=exact
 for mode in results:results[mode]['max_abs_error']=float(np.max(abs(states[mode]-exact)))
 caught=results['uncertified']['maximum_estimator']==0. and results['uncertified']['max_abs_error']>1e-4
 safe=results['event_cut']['max_abs_error']<=1e-4
 # Uniform source probes may miss a difference reader; individual source test catches it.
 f=lambda q:q[0]-q[1]
 uniform=[f(np.full(2,x))-f(np.array([.2,.2])) for x in (0.,.37,1.)];differential=f(np.array([.37,.2]))
 report={'status':'COUNTEREXAMPLE_REPRODUCED' if caught and safe else 'UNEXPECTED_RESULT','continuity':continuity,'runs':results,'reference_expm':exact.tolist(),'criterion':1e-4,'uniform_probe_differences':uniform,'differential_probe':differential,'wall_s':time.perf_counter()-start,'scope':'Actual native CUDA controller and projector, synthetic3states. Does not assert that published fly trajectories violated their measured criteria.'}
 np.savez_compressed(out/'states.npz',**states);(out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
 if not caught or not safe:raise ValueError('Prospective falsifier had unexpected outcome')
if __name__=='__main__':main()
