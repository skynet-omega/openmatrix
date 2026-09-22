"""Compile generic equations to independent native CUDA callback operations."""
from model import require
from autodiff import derivative
from runtime import KERNELS

def source(model):
    lines=['#include <math_constants.h>', 'struct Dynamic {const double* x;const double* v;double* dest;double t;double gamma;};',
    '__device__ double om_exprel(double x){return fabs(x)<1e-5 ? 1+x*(.5+x*(1.0/6+x*(1.0/24+x/120))) : expm1(x)/x;}',
    '__device__ double om_dexprel(double x){return fabs(x)<1e-4 ? .5+x*(1.0/3+x*(1.0/8+x/30)) : (x+(x-1)*expm1(x))/(x*x);}']
    for j,p in enumerate(model.pops):
        for mode in ('output','output_jvp','rhs','jvp','prec'):
            tangent=mode in ('output_jvp','jvp');output=mode.startswith('output')
            lines.append(f'extern "C" __global__ void {mode}_{j}(const Dynamic* d,const double* par,const double* inp,const double* dinp,double* ports,double* dports,const bool* clamp,const double* md,int* flag){{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>={p["n"]})return;double v_t=d->t;')
            env={}
            for k,sl in p['states'].items():
                lines.append(f'double v_{k}=d->x[{sl.start}+i];if(!isfinite(v_{k}))atomicOr(flag,1);')
                if tangent:lines.append(f'double d_{k}=d->v[{sl.start}+i];if(!isfinite(d_{k}))atomicOr(flag,1);');env[k]='d_'+k
            for k,sl in p['params'].items():lines.append(f'double v_{k}=par[{sl.start}+i];')
            if output:
                for k,(sl,ex) in p['outputs'].items():
                    lines.append(f'double a_{k}={ex.cuda};if(!isfinite(a_{k}))atomicOr(flag,1);ports[{sl.start}+i]=a_{k};')
                    if tangent:lines.append(f'double b_{k}={derivative(ex.node,env)};if(!isfinite(b_{k}))atomicOr(flag,1);dports[{sl.start}+i]=b_{k};')
            else:
                for k,sl in p['inputs'].items():
                    lines.append(f'double v_{k}=inp[{sl.start}+i];if(!isfinite(v_{k}))atomicOr(flag,1);')
                    if tangent:lines.append(f'double d_{k}=dinp[{sl.start}+i];if(!isfinite(d_{k}))atomicOr(flag,1);');env[k]='d_'+k
                for k,ex in p['derived'].items():
                    lines.append(f'double v_{k}={ex.cuda};if(!isfinite(v_{k}))atomicOr(flag,1);')
                    if tangent:lines.append(f'double d_{k}={derivative(ex.node,env)};if(!isfinite(d_{k}))atomicOr(flag,1);');env[k]='d_'+k
                for k,ex in p['rhs'].items():
                    sl=p['states'][k];idx=f'{sl.start}+i';lines.append('{')
                    if mode=='prec':
                        denv={k:'1.0'}
                        for dk,de in p['derived'].items():
                            dn='dd_'+dk;lines.append(f'double {dn}={derivative(de.node,denv)};');denv[dk]=dn
                        lines.append(f'double jac=clamp[{idx}]?0:({derivative(ex.node,denv)});double den=md[{idx}]-d->gamma*jac;double inv=1.0/den;double r=d->v[{idx}];'
                                     f'if(!isfinite(jac)||!isfinite(den)||!isfinite(r))atomicOr(flag,1);'
                                     f'bool fallback=(den==0.0||!isfinite(inv));if(fallback)atomicOr(flag,2);'
                                     f'double a=fallback?r/md[{idx}]:r/den;')
                    else:
                        excode=derivative(ex.node,env) if tangent else ex.cuda
                        lines.append(f'double a=clamp[{idx}]?0:({excode});')
                    lines.append(f'if(!isfinite(a))atomicOr(flag,1);d->dest[{idx}]=a;}}')
            lines.append('}')
    projection=KERNELS[KERNELS.index('extern "C" __global__ void project'):KERNELS.index('extern "C" __global__ void stage')]
    lines.append(projection)
    lines.append('''extern "C" __global__ void mass(const Dynamic* d,const double* a,const int* col,const long long* ptr,int n,int* flag){
      int lane=threadIdx.x&31;int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5;if(row>=n)return;
      double sum=0;for(long long q=ptr[row]+lane;q<ptr[row+1];q+=32)sum+=a[q]*d->v[col[q]];
      for(int off=16;off>0;off>>=1)sum+=__shfl_down_sync(0xffffffff,sum,off);
      if(lane==0){if(!isfinite(sum))atomicOr(flag,1);d->dest[row]=sum;}}
      extern "C" __global__ void massprec(const Dynamic* d,const double* md,int n,int* flag){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){double z=d->v[i]/md[i];if(!isfinite(z))atomicOr(flag,1);d->dest[i]=z;}}''')
    return '\n'.join(lines)
