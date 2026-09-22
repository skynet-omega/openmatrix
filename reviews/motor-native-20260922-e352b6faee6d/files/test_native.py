from pathlib import Path
import sys,json,copy,time,hashlib
import numpy as np
import cupy as cp
from model import Model,require
from development import base,hh,state,scalar
from extensions import fixture
from pilot import manufactured
from native_ops import NativeOps
from implicit import Implicit
from session import Session

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);rows=[];started=time.perf_counter()
    for name,spec in [('network',base(96,95)),('hh',hh(3)),('mixed',fixture()),('stiff_mass',manufactured())]:
        m=Model(spec);stream=cp.cuda.Stream(non_blocking=True);op=NativeOps(m,stream);rng=np.random.default_rng(78)
        worst=0.;worstfd=0.
        with stream:
            xs=[cp.asarray(m.initial),cp.asarray(m.initial)];vs=[cp.asarray(rng.normal(size=m.n)),cp.asarray(rng.normal(size=m.n))];targets=[cp.empty(m.n),cp.empty(m.n)]
            for k,t in enumerate([.003,.018,.017,.01]):
                x=xs[k%2];v=vs[(k//2)%2];dest=targets[(k+1)%2];x*=1.001 # includes reuse of same address with changed state
                host=x.get(stream=stream);direction=v.get(stream=stream)
                require(op.call(0,t,x,None,dest)==0,op.error());f=dest.get(stream=stream);truth=m.raw_rhs(t,host);worst=max(worst,float(np.max(abs(f-truth)/(1+abs(truth)))))
                require(op.call(1,t,x,v,dest)==0,op.error());j=dest.get(stream=stream);eps=1e-6;fd=(m.raw_rhs(t,host+eps*direction)-m.raw_rhs(t,host-eps*direction))/(2*eps);worstfd=max(worstfd,float(np.max(abs(j-fd)/(1+abs(fd)))))
                require(op.call(2,t,None,v,dest)==0,op.error());mass=dest.get(stream=stream);require(np.max(abs(mass-m.mass@direction))<1e-11,'mass callback changed')
            x[0]=np.nan;require(op.call(0,0,x,None,targets[0])==-1,'nonfinite hidden');x[:]=cp.asarray(m.initial);require(op.call(0,0,x,None,targets[0])==0,'flag not reset after error')
        require(worst<1e-11 and worstfd<1e-5,'native equations/JVP mismatch')
        rows.append({'case':name,'rhs_error':worst,'JVP_error':worstfd,'pointer_alternation_and_reuse':True,'invalid_then_valid_detected':True});op.close()
    spec={'version':1,'populations':[{'id':'p','count':1,'states':{'x':state('1',.1,1,'rate*(x+2*y)'),'y':state('1',.2,1,'rate*(-2*x-3*y)')},'parameters':{'rate':scalar('1/s',1)}}]}
    fallback=[]
    for implementation in ['python','native']:
        e=Implicit(spec,implementation=implementation)
        with e.stream:
            v=cp.ones(2);z=cp.empty(2)
            if implementation=='native':code=e.operator.call(3,0,e.x,v,z,1.)
            else:code=e._call(3,0,e.x.data.ptr,v.data.ptr,z.data.ptr,1.)
            values=z.get(stream=e.stream);require(code==0 and np.array_equal(values,[1.,.25]),'singular diagonal fallback incorrect')
        fallback.append({'implementation':implementation,'values':values.tolist(),'finite':True});e.close()
    s=Session(base(2,1));old=s.solver;oldclose=old.close;changed=copy.deepcopy(s.model.spec);changed['connections']=[]
    def fail():raise RuntimeError('injected cleanup')
    old.close=fail;receipt=s.replace(changed);require(receipt['committed'] and receipt['cleanup']=='failed' and s.edits[-1]==receipt and s.solver is not old,'transaction finalization false rejection')
    old.close=oldclose;require(s.retry_cleanup()==0,'cleanup retry');s.close()
    result={'operators':rows,'singular_preconditioner':fallback,'cleanup_failure_committed_receipt':receipt,'cleanup_retry_success':True,'wall_s':time.perf_counter()-started}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
