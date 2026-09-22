"""Generated raw F and exact directional derivative F_x v + F_u W H_x v.
The diagonal derivative is used ONLY for preconditioning, never as the full JVP.
"""
import numpy as np
import cupy as cp
from model import Model,require
from autodiff import derivative
from runtime import KERNELS


def codegen(model):
    lines=['#include <math_constants.h>',
           '__device__ double om_exprel(double x) {return fabs(x)<1e-5 ? 1+x*(.5+x*(1.0/6+x*(1.0/24+x/120))) : expm1(x)/x;}',
           '__device__ double om_dexprel(double x) {return fabs(x)<1e-4 ? .5+x*(1.0/3+x*(1.0/8+x/30)) : (x+(x-1)*expm1(x))/(x*x);}']
    for j,p in enumerate(model.pops):
        for phase in ('output','rhs'):
            lines.append(f'extern "C" __global__ void {phase}_{j}(const double* x,const double* v,const double* par,const double* inp,const double* dinp,double* out,double* dout,double* diag,const bool* clamp,double t,int* flag) {{ int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>={p["n"]}) return;double v_t=t;')
            env={}
            for k,sl in p['states'].items():
                lines.append(f'double v_{k}=x[{sl.start}+i];double d_{k}=v[{sl.start}+i];if(!isfinite(v_{k})||!isfinite(d_{k}))atomicOr(flag,1);')
                env[k]='d_'+k
            for k,sl in p['params'].items():lines.append(f'double v_{k}=par[{sl.start}+i];')
            if phase=='output':
                for k,(sl,ex) in p['outputs'].items():
                    lines.append(f'double a_{k}={ex.cuda}, b_{k}={derivative(ex.node,env)};if(!isfinite(a_{k})||!isfinite(b_{k}))atomicOr(flag,1);out[{sl.start}+i]=a_{k};dout[{sl.start}+i]=b_{k};')
            else:
                for k,sl in p['inputs'].items():
                    lines.append(f'double v_{k}=inp[{sl.start}+i];double d_{k}=dinp[{sl.start}+i];if(!isfinite(v_{k})||!isfinite(d_{k}))atomicOr(flag,1);');env[k]='d_'+k
                for k,ex in p['derived'].items():
                    lines.append(f'double v_{k}={ex.cuda};double d_{k}={derivative(ex.node,env)};if(!isfinite(v_{k})||!isfinite(d_{k}))atomicOr(flag,1);');env[k]='d_'+k
                for k,ex in p['rhs'].items():
                    sl=p['states'][k]
                    lines.append(f'double a_{k}={ex.cuda},b_{k}={derivative(ex.node,env)};if(!isfinite(a_{k})||!isfinite(b_{k}))atomicOr(flag,1);out[{sl.start}+i]=clamp[{sl.start}+i]?0:a_{k};dout[{sl.start}+i]=clamp[{sl.start}+i]?0:b_{k};')
                    denv={k:'1.0'}
                    lines.append('{')
                    for dk,de in p['derived'].items():
                        dn='dd_'+dk;lines.append(f'double {dn}={derivative(de.node,denv)};');denv[dk]=dn
                    lines.append(f'diag[{sl.start}+i]=clamp[{sl.start}+i]?0:({derivative(ex.node,denv)});}}')
            lines.append('}')
    # Reuse the generic warp-per-row projection from the v1 implementation.
    projection=KERNELS[KERNELS.index('extern "C" __global__ void project'):KERNELS.index('extern "C" __global__ void stage')]
    lines.append(projection)
    return '\n'.join(lines)

class Coupled:
    def __init__(self,model,stream=None):
        self.model=model;self.stream=stream or cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.par=cp.asarray(model.parameters);self.clamp=cp.asarray(model.clamp_mask)
            self.bias=cp.asarray(model.port_bias);self.zero_bias=cp.zeros(model.nin)
            self.cval=cp.asarray(model.connection.data);self.ccol=cp.asarray(model.connection.indices,dtype=cp.int32);self.cptr=cp.asarray(model.connection.indptr,dtype=cp.int64)
            mass=model.mass.tocsr();self.mval=cp.asarray(mass.data);self.mcol=cp.asarray(mass.indices,dtype=cp.int32);self.mptr=cp.asarray(mass.indptr,dtype=cp.int64);self.mzero=cp.zeros(model.n)
            self.out=cp.empty(model.nout);self.dout=cp.empty(model.nout);self.inp=cp.empty(model.nin);self.dinp=cp.empty(model.nin)
            self.f=cp.empty(model.n);self.j=cp.empty(model.n);self.diag=cp.empty(model.n);self.zero=cp.zeros(model.n);self.flag=cp.zeros(1,dtype=cp.int32)
            self.module=cp.RawModule(code=codegen(model));self.ko=[self.module.get_function(f'output_{i}') for i in range(len(model.pops))];self.kf=[self.module.get_function(f'rhs_{i}') for i in range(len(model.pops))];self.project=self.module.get_function('project')
        self.stream.synchronize()
    def evaluate(self,t,x,v=None):
        if v is None:v=self.zero
        self.flag.fill(0)
        for p,k in zip(self.model.pops,self.ko):k(((p['n']+255)//256,),(256,),(x,v,self.par,self.inp,self.dinp,self.out,self.dout,self.diag,self.clamp,np.float64(t),self.flag))
        if self.model.nin:
            grid=((self.model.nin+7)//8,)
            self.project(grid,(256,),(self.cval,self.ccol,self.cptr,self.out,self.bias,self.inp,np.int32(self.model.nin),self.flag))
            self.project(grid,(256,),(self.cval,self.ccol,self.cptr,self.dout,self.zero_bias,self.dinp,np.int32(self.model.nin),self.flag))
        for p,k in zip(self.model.pops,self.kf):k(((p['n']+255)//256,),(256,),(x,v,self.par,self.inp,self.dinp,self.f,self.j,self.diag,self.clamp,np.float64(t),self.flag))
        return self.f,self.j
    def mass(self,v,out):
        self.project(((self.model.n+7)//8,),(256,),(self.mval,self.mcol,self.mptr,v,self.mzero,out,np.int32(self.model.n),self.flag));return out
    def checked(self,t,x,v=None):
        with self.stream:
            xx=cp.asarray(x);vv=None if v is None else cp.asarray(v);f,j=self.evaluate(t,xx,vv)
            a=f.get(stream=self.stream);b=j.get(stream=self.stream);flag=int(self.flag.get(stream=self.stream)[0])
        require(flag==0 and np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite generated coupled operator')
        return a,b
