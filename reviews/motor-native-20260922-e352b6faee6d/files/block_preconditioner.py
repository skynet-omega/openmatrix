"""Generated per-cell dependency blocks; preparation and application are separate.
Interblock connections remain in the global residual/JVP. No anatomical dispatch.
"""
import numpy as np
import cupy as cp
from autodiff import derivative
from model import require
class Blocks:
    def __init__(self,model,operator,stream):
        self.model=model;self.operator=operator;self.stream=stream;self.parts=[];self.gamma=None;self.ready=False;self.setups=0;self.factorizations=0;self.applies=0
        code=['#include <math_constants.h>', '__device__ double om_exprel(double x){return fabs(x)<1e-5?1+x*(.5+x*(1.0/6+x*(1.0/24+x/120))):expm1(x)/x;}', '__device__ double om_dexprel(double x){return fabs(x)<1e-4?.5+x*(1.0/3+x*(1.0/8+x/30)):(x+(x-1)*expm1(x))/(x*x);}']
        mass=model.mass.tocoo()
        with stream:
            self.flag=cp.zeros(1,dtype=cp.int32)
            for j,p in enumerate(model.pops):
                names=list(p['states']);d=len(names);n=p['n'];require(d<=16,'block prototype admits at most16 states per cell');start=next(iter(p['states'].values())).start
                mask=(mass.row>=start)&(mass.row<start+n*d)&(mass.col>=start)&(mass.col<start+n*d)&((mass.row-start)%n==(mass.col-start)%n)
                rr=mass.row[mask]-start;cc=mass.col[mask]-start;mb=np.zeros((n,d,d));mb[rr%n,rr//n,cc//n]=mass.data[mask]
                coupling=[];terms={}
                for inp,isl in p['inputs'].items():
                    terms[inp]=[]
                    for out,(osl,ex) in p['outputs'].items():
                        diag=model.connection[isl,osl].diagonal()
                        if np.any(diag):idx=len(coupling);coupling.append(diag);terms[inp].append((idx,ex))
                links=np.array(coupling).reshape(-1,n) if coupling else np.zeros((0,n))
                self.parts.append({'p':p,'d':d,'mass':cp.asarray(mb),'J':cp.empty_like(cp.asarray(mb)),'LU':cp.empty_like(cp.asarray(mb)),'piv':cp.empty((n,d),dtype=cp.int32),'links':cp.asarray(links)})
                code.append(f'extern "C" __global__ void jac_{j}(const double* x,const double* par,const double* inp,const double* links,const bool* clamp,double t,double* J,int* flag){{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>={n})return;double v_t=t;')
                for k,sl in p['states'].items():code.append(f'double v_{k}=x[{sl.start}+i];')
                for k,sl in p['params'].items():code.append(f'double v_{k}=par[{sl.start}+i];')
                for k,sl in p['inputs'].items():code.append(f'double v_{k}=inp[{sl.start}+i];')
                for k,ex in p['derived'].items():code.append(f'double v_{k}={ex.cuda};')
                for col,key in enumerate(names):
                    env={key:'1.0'};code.append('{')
                    for inp,terms_i in terms.items():env[inp]='('+ '+'.join(f'links[{idx*n}+i]*({derivative(ex.node,{key:"1.0"})})' for idx,ex in terms_i)+')' if terms_i else '0.0'
                    for k,ex in p['derived'].items():dn='dd_'+k;code.append(f'double {dn}={derivative(ex.node,env)};');env[k]=dn
                    for row,k in enumerate(names):
                        ex=p['rhs'][k];sl=p['states'][k]
                        code.append(f'double a_{k}=clamp[{sl.start}+i]?0:({derivative(ex.node,env)});if(!isfinite(a_{k}))atomicOr(flag,1);J[i*{d*d}+{row*d+col}]=a_{k};')
                    code.append('}')
                code.append('}')
                code.append(f'''extern "C" __global__ void factor_{j}(const double* M,const double* J,double gamma,double* LU,int* piv,int* flag){{
                int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>={n})return;double a[{d*d}];bool fallback=false;
                for(int k=0;k<{d*d};k++){{a[k]=M[i*{d*d}+k]-gamma*J[i*{d*d}+k];if(!isfinite(a[k]))atomicOr(flag,1);}}
                for(int k=0;k<{d};k++){{int p=k;double best=fabs(a[k*{d}+k]);for(int r=k+1;r<{d};r++)if(fabs(a[r*{d}+k])>best){{p=r;best=fabs(a[r*{d}+k]);}}
                  if(best==0||!isfinite(1.0/best)){{fallback=true;break;}}piv[i*{d}+k]=p;
                  for(int c=0;c<{d};c++){{double z=a[k*{d}+c];a[k*{d}+c]=a[p*{d}+c];a[p*{d}+c]=z;}}
                  for(int r=k+1;r<{d};r++){{a[r*{d}+k]/=a[k*{d}+k];for(int c=k+1;c<{d};c++)a[r*{d}+c]-=a[r*{d}+k]*a[k*{d}+c];}}
                }}
                if(fallback){{atomicOr(flag,2);for(int r=0;r<{d};r++){{piv[i*{d}+r]=r;for(int c=0;c<{d};c++)a[r*{d}+c]=r==c?M[i*{d*d}+r*{d}+c]:0;}}}}
                for(int k=0;k<{d*d};k++){{LU[i*{d*d}+k]=a[k];if(!isfinite(a[k]))atomicOr(flag,1);}}
                }}
                extern "C" __global__ void solve_{j}(const double* LU,const int* piv,const double* rhs,double* result,int* flag){{
                int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>={n})return;double b[{d}];for(int k=0;k<{d};k++)b[k]=rhs[{start}+k*{n}+i];
                // Apply ALL pivot swaps before triangular substitution (LU stores already permuted L rows).
                for(int k=0;k<{d};k++){{int p=piv[i*{d}+k];double z=b[k];b[k]=b[p];b[p]=z;}}
                for(int r=0;r<{d};r++)for(int c=0;c<r;c++)b[r]-=LU[i*{d*d}+r*{d}+c]*b[c];
                for(int r={d}-1;r>=0;r--){{for(int c=r+1;c<{d};c++)b[r]-=LU[i*{d*d}+r*{d}+c]*b[c];b[r]/=LU[i*{d*d}+r*{d}+r];if(!isfinite(b[r]))atomicOr(flag,1);result[{start}+r*{n}+i]=b[r];}}
                }}''')
            self.module=cp.RawModule(code='\n'.join(code));self.kernels=[tuple(self.module.get_function(f'{k}_{j}') for k in ['jac','factor','solve']) for j in range(len(model.pops))]
        stream.synchronize()
    def prepare(self,t,x,gamma):
        self.operator.evaluate(t,x);self.flag.fill(0)
        for b,ks in zip(self.parts,self.kernels):ks[0](((b['p']['n']+63)//64,),(64,),(x,self.operator.par,self.operator.inp,b['links'],self.operator.clamp,np.float64(t),b['J'],self.flag))
        self.ready=True;self.setups+=1;self.factor(gamma)
    def factor(self,gamma):
        for b,ks in zip(self.parts,self.kernels):ks[1](((b['p']['n']+63)//64,),(64,),(b['mass'],b['J'],np.float64(gamma),b['LU'],b['piv'],self.flag))
        self.gamma=gamma;self.factorizations+=1
    def solve(self,t,x,gamma,r,out):
        self.flag.fill(0)
        if not self.ready:self.prepare(t,x,gamma)
        elif gamma!=self.gamma:self.factor(gamma)
        for b,ks in zip(self.parts,self.kernels):ks[2](((b['p']['n']+63)//64,),(64,),(b['LU'],b['piv'],r,out,self.flag))
        self.applies+=1
