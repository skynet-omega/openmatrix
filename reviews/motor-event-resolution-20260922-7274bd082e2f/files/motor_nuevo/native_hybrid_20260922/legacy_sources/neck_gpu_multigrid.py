"""GPU-resident solve of the unchanged fine PN RC operator, in float64.

Red/black volume relaxation changes only the preconditioner. Every fine node,
membrane coefficient and exterior/collar coupling remains in the solved matrix.
The small final coarse solve stays on CPU. No model reduction or CNS admission.
"""
import time
import numpy as np
from numba import njit
from scipy.sparse import diags
from scipy.sparse.linalg import splu


@njit(cache=True)
def _bipartite(indptr,indices,parity):
    n=len(parity)
    for i in range(n):
        for k in range(indptr[i],indptr[i+1]):
            j=indices[k]
            if j<n and j!=i and parity[i]==parity[j]:return False
    return True


CUDA=r'''
extern "C" __global__ void sweep(int n, const int* ptr, const int* col,
 const double* a, const double* c, const unsigned char* parity, int color,
 double shift, const double* b, double* x) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i>=n || parity[i]!=color) return;
 double diagonal=shift*c[i], sum=0.;
 for(int k=ptr[i];k<ptr[i+1];++k) {
  int j=col[k]; if(j==i) diagonal+=a[k]; else sum+=a[k]*x[j];
 }
 x[i]=(b[i]-sum)/diagonal;
}
extern "C" __global__ void restrict_rows(int n,const int* ptr,const int* col,
 const double* fine,double* coarse) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 double sum=0.;for(int k=ptr[i];k<ptr[i+1];++k)sum+=fine[col[k]];
 coarse[i]=sum;
}
__device__ void add_compensated(double term,double &sum,double &correction) {
 double value=sum+term;
 if(fabs(sum)>=fabs(term))correction+=(sum-value)+term;
 else correction+=(term-value)+sum;
 sum=value;
}
extern "C" __global__ void long_rows(const int* rows,const int* ptr,const int* col,
 const double* a,const double* x,double* out) {
 int row=rows[blockIdx.x],lane=threadIdx.x;double sum=0.,correction=0.;
 for(int k=ptr[row]+lane;k<ptr[row+1];k+=blockDim.x)
  add_compensated(a[k]*x[col[k]],sum,correction);
 __shared__ double totals[128];__shared__ double corrections[128];
 totals[lane]=sum;corrections[lane]=correction;__syncthreads();
 if(lane==0) {
  sum=0.;correction=0.;
  for(int j=0;j<blockDim.x;++j) {
   add_compensated(totals[j],sum,correction);
   add_compensated(corrections[j],sum,correction);
  }
  out[row]=sum+correction;
 }
}
extern "C" __global__ void row_sums(int n,const int* ptr,const double* a,double* sums) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 double sum=0.,correction=0.;
 for(int k=ptr[i];k<ptr[i+1];++k)add_compensated(a[k],sum,correction);
 sums[i]=sum+correction;
}
extern "C" __global__ void difference_rows(int n,const int* ptr,const int* col,
 const double* a,const double* sums,const double* x,double* out) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n || ptr[i+1]-ptr[i]>=32)return;
 double sum=0.,correction=0.,xi=x[i];
 for(int k=ptr[i];k<ptr[i+1];++k)
  add_compensated(a[k]*(x[col[k]]-xi),sum,correction);
 out[i]=(sum+correction)+sums[i]*xi;
}
extern "C" __global__ void long_differences(const int* rows,const int* ptr,const int* col,
 const double* a,const double* sums,const double* x,double* out) {
 int row=rows[blockIdx.x],lane=threadIdx.x;double sum=0.,correction=0.,xi=x[row];
 for(int k=ptr[row]+lane;k<ptr[row+1];k+=blockDim.x)
  add_compensated(a[k]*(x[col[k]]-xi),sum,correction);
 __shared__ double totals[128];__shared__ double corrections[128];
 totals[lane]=sum;corrections[lane]=correction;__syncthreads();
 if(lane==0) {
  sum=0.;correction=0.;
  for(int j=0;j<blockDim.x;++j) {
   add_compensated(totals[j],sum,correction);
   add_compensated(corrections[j],sum,correction);
  }
  out[row]=(sum+correction)+sums[row]*xi;
 }
}
'''


class GPUJointPNMultigrid:
    def __init__(self,cpu,*,reserve_bytes=2*1024**3,action_mode='csr'):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
        cp.cuda.Device(0).use()
        if action_mode not in ('csr','voltage_differences'):raise ValueError('Unknown fine action arithmetic')
        self.action_mode=action_mode
        self.cp=cp;self.cpu=cpu;self.levels=[];self.shift=None;self.coarse_factor=None
        self._cpu_operators=[(l.G,l.C) for l in cpu.levels]
        # Matrix, mass, maps, parity and a conservative workspace allowance.
        required=0
        for l in cpu.levels:
            required+=l.G.data.nbytes+l.G.indices.nbytes+l.G.indptr.nbytes+l.C.nbytes+l.volume_parity.nbytes
            if hasattr(l,'P'):required+=l.P.indices.nbytes+l.R.indptr.nbytes+l.R.indices.nbytes
            required+=8*l.G.shape[0]*8
            if action_mode=='voltage_differences':required+=l.C.nbytes
        free,total=cp.cuda.runtime.memGetInfo()
        if required+reserve_bytes>free:raise MemoryError(f'GPU requires {required} bytes plus reserve; free {free}')
        self.memory_plan=dict(estimated_bytes=required,reserve_bytes=reserve_bytes,free_before_bytes=free,total_bytes=total)
        module=cp.RawModule(code=CUDA,options=('--fmad=false',),name_expressions=['sweep','restrict_rows','long_rows','row_sums','difference_rows','long_differences'])
        self.sweep=module.get_function('sweep');self.restrict=module.get_function('restrict_rows');self.long_rows=module.get_function('long_rows')
        self.row_sums=module.get_function('row_sums');self.difference_rows=module.get_function('difference_rows');self.long_differences=module.get_function('long_differences')
        self.shift_kernel=cp.ElementwiseKernel('float64 shift, raw float64 c, raw float64 x','float64 y','y += shift*c[i]*x[i];','pn_shift')
        self.prolong_kernel=cp.ElementwiseKernel('raw int32 mapping, raw float64 coarse','float64 x','x += coarse[mapping[i]];','pn_prolong')
        for l in cpu.levels:
            if not _bipartite(l.G.indptr,l.G.indices,l.volume_parity):raise ValueError('Volume is not bipartite; parallel GS would race')
            row=dict(G=csr_matrix(l.G),C=cp.asarray(l.C),parity=cp.asarray(l.volume_parity),nv=l.volume_nodes,n=len(l.C),
                     long=cp.asarray(np.flatnonzero(np.diff(l.G.indptr)>=32).astype(np.int32)))
            if hasattr(l,'P'):
                if not (np.all(np.diff(l.P.indptr)==1) and np.all(l.P.data==1) and np.all(l.R.data==1)):
                    raise ValueError('Only explicitly certified one-hot Galerkin maps supported')
                row.update(mapping=cp.asarray(l.P.indices),rp=cp.asarray(l.R.indptr),ri=cp.asarray(l.R.indices),coarse_n=l.P.shape[1])
            if action_mode=='voltage_differences':
                row['row_sums']=cp.empty(row['n'],dtype=cp.float64)
                self.row_sums(((row['n']+127)//128,),(128,),(np.int32(row['n']),row['G'].indptr,row['G'].data,row['row_sums']))
            self.levels.append(row)
        cp.cuda.get_current_stream().synchronize()
        self.memory_plan['free_after_upload_bytes']=cp.cuda.runtime.memGetInfo()[0]

    def assert_model(self):
        if len(self.cpu.levels)!=len(self._cpu_operators) or any(l.G is not g or l.C is not c for l,(g,c) in zip(self.cpu.levels,self._cpu_operators)):
            raise ValueError('CPU model changed; rebuild GPU buffers explicitly')

    def action(self,v,*,shift=0.,level=0):
        l=self.levels[level];G=l['G']
        if self.action_mode=='voltage_differences':
            # Algebraically identical for the STORED G, including its actual
            # row sums. Avoid subtracting large almost equal axial products.
            # Never replace the row sum by an assumed zero or a fitted leak.
            out=self.cp.empty_like(v)
            self.difference_rows(((l['n']+127)//128,),(128,),(np.int32(l['n']),G.indptr,G.indices,G.data,l['row_sums'],v,out))
            if len(l['long']):self.long_differences((len(l['long']),),(128,),(l['long'],G.indptr,G.indices,G.data,l['row_sums'],v,out))
        else:
            out=G@v
            if len(l['long']):self.long_rows((len(l['long']),),(128,),(l['long'],G.indptr,G.indices,G.data,v,out))
        if shift:self.shift_kernel(float(shift),l['C'],v,out)
        return out

    def set_shift(self,shift):
        if not np.isfinite(shift) or shift<0:raise ValueError('Finite nonnegative real shift required')
        if self.shift!=shift:
            l=self.cpu.levels[-1]
            self.coarse_factor=splu((l.G+diags(shift*l.C)).tocsc())
            self.shift=float(shift)

    def cycle(self,b,level=0):
        cp=self.cp;l=self.levels[level]
        if level==len(self.levels)-1:return cp.asarray(self.coarse_factor.solve(cp.asnumpy(b)))
        x=cp.zeros_like(b);G=l['G'];grid=((l['nv']+127)//128,)
        for _ in range(self.cpu.smoothing_steps):
            for color in (0,1):
                self.sweep(grid,(128,),(np.int32(l['nv']),G.indptr,G.indices,G.data,l['C'],l['parity'],np.int32(color),self.shift,b,x))
        residual=b-self.action(x,shift=self.shift,level=level)
        coarse=cp.empty(l['coarse_n'],dtype=cp.float64)
        self.restrict(((l['coarse_n']+127)//128,),(128,),(np.int32(l['coarse_n']),l['rp'],l['ri'],residual,coarse))
        correction=self.cycle(coarse,level+1);self.prolong_kernel(l['mapping'],correction,x)
        for _ in range(self.cpu.smoothing_steps):
            for color in (1,0):
                self.sweep(grid,(128,),(np.int32(l['nv']),G.indptr,G.indices,G.data,l['C'],l['parity'],np.int32(color),self.shift,b,x))
        return x

    def solve(self,current_pA,*,shift=0.,rtol=1e-9,atol=2e-12,maxiter=220,initial=None,progress=None,diagonal_update=None,diagonal_is_jacobian=False):
        cp=self.cp
        from cupyx.scipy.sparse.linalg import LinearOperator,cg
        self.assert_model()
        if (not np.isscalar(rtol) or np.iscomplexobj(rtol) or not np.isfinite(rtol) or rtol<=0
                or not np.isscalar(atol) or np.iscomplexobj(atol) or not np.isfinite(atol) or atol<0
                or type(maxiter) is not int or maxiter<1):raise ValueError('Finite tolerances and integer budget required')
        if cp.iscomplexobj(current_pA):raise ValueError('Real DC/time-domain current required')
        b=cp.asarray(current_pA,dtype=cp.float64)
        if b.shape!=self.levels[0]['C'].shape or not bool(cp.isfinite(b).all()):raise ValueError('Finite full current required')
        bnorm=float(cp.linalg.norm(b));threshold=max(atol,rtol*bnorm)
        if not np.isfinite(bnorm) or not np.isfinite(threshold):raise ValueError('Finite norm and threshold required')
        # Updates affect this solve, never the stored passive G/C. Physical
        # conductances must be nonnegative. Explicit Newton derivatives may
        # be signed only under the sufficient SPD bound checked below.
        if type(diagonal_is_jacobian) is not bool:raise ValueError('Explicit Jacobian flag required')
        nodes=values=None
        if diagonal_update is not None:
            if not isinstance(diagonal_update,tuple) or len(diagonal_update)!=2:
                raise ValueError('Diagonal update requires (unique nodes, nS)')
            nodes,values=map(cp.asarray,diagonal_update)
            if (nodes.ndim!=1 or nodes.dtype.kind not in 'iu' or values.shape!=nodes.shape
                    or values.dtype.kind not in 'fiu' or not bool(cp.isfinite(values).all())
                    or (not diagonal_is_jacobian and bool(cp.any(values<0))) or bool(cp.any(nodes<0)) or bool(cp.any(nodes>=len(b)))
                    or len(cp.unique(nodes))!=len(nodes)):
                raise ValueError('Finite nonnegative diagonal and unique in-range nodes required')
            # A signed ionic derivative is a Newton linearization, not a
            # negative physical conductance. Retain CG only with this explicit
            # sufficient SPD condition on top of the unchanged passive G.
            if diagonal_is_jacobian:
                mass=self.levels[0]['C'][nodes]*shift+values
                if not bool(cp.isfinite(mass).all()) or bool(cp.any(mass<0)):
                    raise ValueError('Signed Jacobian lacks the sufficient mass-plus-diagonal SPD bound')
        self.set_shift(shift);N=len(b);iterations=0;start=time.monotonic();restarts=0
        def action(x):
            out=self.action(x,shift=shift)
            if nodes is not None:out[nodes]+=values*x[nodes]
            return out
        A=LinearOperator((N,N),matvec=action,dtype=cp.float64)
        M=LinearOperator((N,N),matvec=self.cycle,dtype=cp.float64)
        def callback(x):
            nonlocal iterations
            iterations+=1
            if progress and (iterations==1 or iterations%20==0):
                progress(dict(iteration=iterations,seconds=time.monotonic()-start,fine_relative_residual=float(cp.linalg.norm(b-A@x))/max(bnorm,1e-300)))
        x=initial
        while True:
            # Installed CuPy 13.6 names its relative tolerance ``tol``.
            x,info=cg(A,b,x0=x,tol=rtol*.5**restarts,atol=atol*.5**restarts,maxiter=maxiter-iterations,M=M,callback=callback)
            norm=float(cp.linalg.norm(b-A@x))
            if norm<=threshold or info!=0 or iterations>=maxiter or restarts>=3:break
            restarts+=1
        return x,dict(info=int(info),iterations=iterations,residual_l2_pA=norm,current_l2_pA=bnorm,threshold_pA=threshold,
                      relative_residual=norm/max(bnorm,1e-300),fine_residual_passed=bool(np.isfinite(norm) and norm<=threshold),
                      residual_restarts=restarts,seconds=time.monotonic()-start,backend='GPU_float64_full_fine',shift_real=float(shift))
