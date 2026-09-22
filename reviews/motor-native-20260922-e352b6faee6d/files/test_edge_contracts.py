from pathlib import Path
import sys,json,numpy as np,cupy as cp
from model import Model,require
from native_ops import NativeOps
from development import state,scalar
from implicit_blocks import Implicit

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    # A fixed nontriangular3x3 operator requiring pivot swaps.
    J=np.array([[1.,-2.,-1.],[-3.,1.,-4.],[-5.,-6.,-6.]])
    names=['x','y','z'];spec={'version':1,'populations':[{'id':'p','count':1,'states':{k:state('1',.1*(i+1),1,'rate*('+ '+'.join(f'({J[i,j]})*{v}' for j,v in enumerate(names))+')') for i,k in enumerate(names)},'parameters':{'rate':scalar('1/s',1)}}]}
    e=Implicit(spec);rhs=np.array([1.,2.,3.]);errors=[]
    with e.stream:
        r=cp.asarray(rhs);z=cp.empty(3)
        require(e._call(5,0,e.x.data.ptr,None,None,1.)==0,'block setup failed')
        for gamma in [1.,.003,1.]:
            require(e._call(3,0,e.x.data.ptr,r.data.ptr,z.data.ptr,gamma)==0,'block solve failed')
            truth=np.linalg.solve(np.eye(3)-gamma*J,rhs);err=float(np.max(abs(z.get(stream=e.stream)-truth)));require(err<1e-11,'pivot/gamma error');errors.append(err)
        original=e.x.copy();e.x[0]=cp.nan
        require(e._call(5,0,e.x.data.ptr,None,None,1.)==-1,'nonfinite setup hidden')
        e.x[:]=original;require(e._call(5,0,e.x.data.ptr,None,None,1.)==0,'failed setup persisted')
    e.close()
    m=Model(spec);stream=cp.cuda.Stream(non_blocking=True);op=NativeOps(m,stream)
    with stream:
        xs=[cp.asarray(m.initial),cp.asarray(m.initial)];dest=[cp.full(3,42.),cp.full(3,-19.)]
        for i in range(8):
            x=xs[i%2];x*=1.01;inactive=dest[(i+1)%2].get(stream=stream);host=x.get(stream=stream)
            require(op.call(0,.017+i*.003,x,None,dest[i%2])==0,'pointer call failed')
            require(np.array_equal(inactive,dest[(i+1)%2].get(stream=stream)),'inactive destination overwritten')
            require(np.max(abs(dest[i%2].get(stream=stream)-m.raw_rhs(.017+i*.003,host)))<1e-12,'stale source/destination')
    op.close();result={'pivot_gamma_absolute_errors':errors,'nonfinite_setup_rejected_then_recovered':True,'inactive_destinations_unchanged_over_eight_calls':True}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
