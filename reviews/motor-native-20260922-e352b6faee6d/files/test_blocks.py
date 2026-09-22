from pathlib import Path
import sys,time,json,numpy as np,cupy as cp
from model import Model,require
from coupled import Coupled
from block_preconditioner import Blocks
from development import base,hh,state,scalar

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);rows=[]
    sign={'version':1,'populations':[{'id':'p','count':1,'states':{'x':state('1',.1,1,'k*(x+2*y)'),'y':state('1',.2,1,'k*(-2*x-3*y)')},'parameters':{'k':scalar('1/s',1)}}]}
    selfloop=base(3,1);selfloop['connections'][0]['offsets']=[0]
    for name,spec in [('invertible',sign),('self_connection',selfloop),('hh',hh(1))]:
        m=Model(spec);stream=cp.cuda.Stream(non_blocking=True);op=Coupled(m,stream);b=Blocks(m,op,stream);x=m.initial;eps=1e-6;J=np.stack([(m.raw_rhs(.017,x+eps*v)-m.raw_rhs(.017,x-eps*v))/(2*eps) for v in np.eye(m.n)],axis=1)
        with stream:
            xx=cp.asarray(x);r=cp.asarray(np.arange(m.n)+1.,dtype=cp.float64);z=cp.empty(m.n)
            for gamma in [1.,.0003,.07]:
                b.prepare(.017,xx,gamma);b.solve(.017,xx,gamma,r,z);got=z.get(stream=stream)
                # These fixtures contain no cross-cell couplings, so the block solve should match full W.
                truth=np.linalg.solve(m.mass.toarray()-gamma*J,np.arange(m.n)+1.);err=float(np.max(abs(got-truth)/(1+abs(truth))))
                require((int(b.flag.get(stream=stream)[0])&1)==0 and err<1e-5,'block solve mismatch')
                rows.append({'case':name,'gamma':gamma,'error':err})
    (out/'RESULT.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
