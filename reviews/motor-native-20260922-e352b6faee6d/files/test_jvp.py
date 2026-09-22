from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from model import Model,require
from coupled import Coupled
from development import base,hh,state,scalar
from extensions import fixture

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);rows=[]
    sign={'version':1,'populations':[{'id':'p','count':1,'states':{'x':state('1',.1,1,'(-x+u)/tau')},'parameters':{'tau':scalar('s',.1)},'inputs':{'u':scalar('1',0)},'outputs':{'xout':{'unit':'1','expr':'x'}}}],
          'connections':[{'source':['p','xout'],'target':['p','u'],'pattern':'coo','row':[0],'col':[0],'weights':[2]}]}
    for name,spec in [('sign',sign),('network',base(96,95)),('hh',hh(3)),('mixed',fixture())]:
        m=Model(spec);c=Coupled(m);x=m.initial.copy();v=np.random.default_rng(72).normal(size=m.n)
        f,j=c.checked(.012,x,v);cpu=m.raw_rhs(.012,x)
        require(np.max(abs(f-cpu)/(1+abs(cpu)))<1e-11,'raw RHS CPU/GPU')
        if name=='sign':require(abs(j[0]-10*v[0])<1e-10,'JVP connection sign wrong')
        eps=1e-6;numeric=(m.raw_rhs(.012,x+eps*v)-m.raw_rhs(.012,x-eps*v))/(2*eps)
        err=float(np.max(abs(j-numeric)/(1+abs(numeric))));require(err<1e-5,'global JVP finite-difference mismatch')
        np.savez_compressed(out/(name+'.npz'),state=x,direction=v,jvp=j,finite_difference=numeric,rhs=f)
        rows.append({'case':name,'states':m.n,'edges':m.connection.nnz,'scaled_error':err})
    (out/'RESULT.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
