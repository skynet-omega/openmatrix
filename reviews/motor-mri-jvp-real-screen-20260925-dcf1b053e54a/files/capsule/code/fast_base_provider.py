"""Same-run fast-provider prototype; original owner stack remains active.
No whole-brain integration or continuous error certificate. Dedicated worker.
CLI tests only CPU: the CuPy path is untested. Original MRI is not modified.
"""
import hashlib, json, math, time
from contextlib import contextmanager
import numpy as np
CUDA=r'''
extern "C" __global__ void supply(int n,const long long*ptr,const int*idx,
 const double*w,const double*s,const double*cap,const bool*vis,const double*tau,
 const double*gain,const double*theta,const double*drive,const double*photo,
 double scale,bool connected,double*target,double*rate,
 const long long*lp,const long long*pos,const bool*live,double*anchor,int mode){
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
 if(row>=n)return; double a=0.,b=0.,fa=0.,fb=0.;
 long long start=mode==1?lp[row]:ptr[row],end=mode==1?lp[row+1]:ptr[row+1];
 for(long long k=start+lane;k<end;k+=32){
  long long e=mode==1?pos[k]:k;int c=idx[e];double x=0.,z=0.;
  if(vis[row]){double v=(w[e]*scale)*s[c];if(v>=0.)x=v;else z=-v;}
  else if(connected||!vis[c])x=w[e]*(s[c]*cap[c]);
  a+=x;b+=z;if(mode==0&&!live[c]){fa+=x;fb+=z;}
 }
 for(int d=16;d;d/=2){a+=__shfl_down_sync(0xffffffff,a,d);b+=__shfl_down_sync(0xffffffff,b,d);
 fa+=__shfl_down_sync(0xffffffff,fa,d);fb+=__shfl_down_sync(0xffffffff,fb,d);}
 if(lane==0){
  if(mode==0){anchor[2*row]=fa;anchor[2*row+1]=fb;}
  if(mode==1){a+=anchor[2*row];b+=anchor[2*row+1];}
  if(vis[row]){a+=photo[row];double total=1.+a+b;target[row]=(.25+a)/total;rate[row]=total/tau[row];}
  else{target[row]=fmax(0.,tanh(gain[row]*(a+drive[row]-theta[row])));rate[row]=1./tau[row];}
 }
}
'''
def need(ok,msg):
    if not ok: raise ValueError(msg)
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def token(fn):
    x=fn();need(set(x)=={'operator','owners','events','boundary'},'Four versions required')
    need(all(isinstance(v,str) and len(v)==64 and set(v)<=set('0123456789abcdef') for v in x.values()),'Use content-derived hashes')
    return json.dumps(x,sort_keys=True)

class RealFast:
    """MRI Model interface: project/full/fast/version/setup_edges.
    It preserves original model/owner equations by calling core.coefficient.
    ONLY self.kernel (the base CSR consumer) is substituted. All specialized
    kernels still run and their declared edge upper bound is charged EVERY call.
    owners_edges counts ALL other active kernel edge reads; unknown blocks.
    CPU MRI transfers candidates; this is NOT a native speed implementation.
    """
    def __init__(self,adapter,live_sources,guard,owners_edges,budget_edges):
        import cupy as cp
        from gpu_coefficient_buffers import CoefficientBuffers
        import event_ports as ep
        self.cp=cp;self.ad=adapter;self.b=adapter.brain;self.g=adapter.core
        self.stream=self.g.stream;self.guard=guard;self.start=time.monotonic()
        self.expected=token(guard);self.valid=True;self.seeded=False
        need(type(owners_edges) is int and owners_edges>=0,'Unknown owner work: stop')
        self.owner_work=owners_edges;self.E=int(self.b.cuda['indices'].size)
        need(type(budget_edges) is int and 0<budget_edges<=6*self.E,'Edge budget')
        self.limit=budget_edges;self.spent=0;self.log=[];self.cached=None
        with self.stream:
            ptr=cp.asnumpy(self.b.cuda['indptr']);idx=cp.asnumpy(self.b.cuda['indices'])
            n=len(ptr)-1;mask=np.asarray(live_sources)
            need(mask.dtype==bool and mask.shape==(n,),'Declared live-source mask')
            ports=np.asarray(adapter.events.rows)
            need(ports.dtype.kind in 'iu' and np.all((ports>=0)&(ports<n)) and mask[ports].all(),'Every prescribed source must be live')
            need(ptr.dtype==np.int64 and idx.dtype==np.int32 and ptr[0]==0 and ptr[-1]==self.E and np.all(np.diff(ptr)>=0),'CSR layout')
            need(np.all((idx>=0)&(idx<n)),'CSR indices')
            selected=mask[idx];prefix=np.r_[np.int64(0),np.cumsum(selected,dtype=np.int64)]
            lp=prefix[ptr];positions=np.flatnonzero(selected).astype(np.int64)
            self.live_edges=len(positions);self.n=n
            self.lp=cp.asarray(lp);self.pos=cp.asarray(positions);self.mask=cp.asarray(mask)
            self.anchor=cp.zeros((n,2),dtype=cp.float64)
            self.kernel=cp.RawKernel(CUDA,'supply',options=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true'))
            self.kernel.compile();self.buffers=CoefficientBuffers(self.g.n)
            self.clock=cp.zeros(2,dtype=cp.float64)
            # Private left/right variant.
            code=ep.CODE
            old='const double*clock,double fraction)'
            need(code.count(old)==1 and code.count('et[p]<=t')==2 and code.count('mark>t')==1,'Unsupported projector source; do not approximate left by nextafter')
            code=code.replace(old,'const double*clock,double fraction,int right)')
            code=code.replace('et[p]<=t','(et[p]<t||(right&&et[p]==t))').replace('mark>t','(mark>t||(!right&&mark==t))')
            self.port_kernel=cp.RawKernel(code,'port',options=('--fmad=false',));self.port_kernel.compile()
        self.stream.synchronize();self.setup_edges=self.E
        self.structure_hash=hashlib.sha256(ptr.tobytes()+idx.tobytes()+mask.tobytes()).hexdigest()
        self.spent=self.setup_edges;self._check()
    def _check(self):
        need(self.valid,'Failed provider: dispose worker')
        try:
            need(time.monotonic()-self.start<=120,'120s budget')
            need(token(self.guard)==self.expected,'Operator/owner/event/boundary changed')
            need(self.spent<=self.limit,'Edge budget exhausted')
        except BaseException:
            self.valid=False;raise
    def version(self):return token(self.guard)
    def project(self,t,z,side):
        need(side in ('left','right') and 0<=t<=125e-6 and np.isfinite(t),'Projection side/time')
        self._check();cp=self.cp;p=self.ad.ports
        with self.stream:
            out=cp.asarray(z).copy();need(out.dtype==cp.float64 and out.shape==(self.g.n,),'State')
            self.clock.set(np.array([t,0.],np.float64))
            args=(out,p.qr,p.sr,p.q,p.s,p.tq,p.ts,p.times,p.jumps,p.sets,p.posts,p.counts,
                  np.int32(p.n),np.int32(p.width),self.clock,np.float64(0),np.int32(side=='right'))
            self.port_kernel(((p.n+255)//256,),(256,),args,stream=self.stream)
            result=cp.asnumpy(out,stream=self.stream)
        need(np.isfinite(result).all(),'Nonfinite projection');return result
    @contextmanager
    def _context(self,mode):
        b=self.b;s=b._online_source
        old_kernel=b.kernel;old_reader=s.general_transmission
        old_buffers=getattr(b,'_coefficient_buffers',None);stats=dict(b.statistics)
        calls=[0]
        def supply(grid,block,args,**kw):
            need(len(args)==16 and int(args[0])==self.n,'Base kernel signature')
            calls[0]+=1
            if mode==2:return old_kernel(grid,block,args,**kw)
            self.kernel(grid,block,tuple(args)+(self.lp,self.pos,self.mask,self.anchor,np.int32(mode)),**kw)
        try:
            need(b._edge_buffer_active and not b._general_buffer_active,'Inactive owner view')
            b.kernel=supply;s.general_transmission=lambda:self.ad.pn;b._coefficient_buffers=self.buffers
            yield calls
        finally:
            b.kernel=old_kernel;s.general_transmission=old_reader
            if old_buffers is None:del b._coefficient_buffers
            else:b._coefficient_buffers=old_buffers
            b.statistics.clear();b.statistics.update(stats)
    def _evaluate(self,t,z,mode):
        self._check();cp=self.cp;work=(self.live_edges if mode==1 else self.E)+self.owner_work
        if self.spent+work>self.limit:
            self.valid=False
            raise RuntimeError('Next callback exceeds total edge budget')
        self.spent+=work;self.log.append({'mode':mode,'time':t,'edge_upper_bound':work})
        try:
            with self.stream,self._context(mode) as calls:
                y=cp.asarray(z).copy();need(y.dtype==cp.float64 and y.shape==(self.g.n,),'Candidate')
                a,r=self.g.coefficient(y);a=a.copy();r=r.copy()
                out=[cp.asnumpy(v,stream=self.stream) for v in (a,r)]
                need(calls[0]==1,'Unexpected base-consumer multiplicity')
            self.stream.synchronize();self._check()
            need(all(v.shape==z.shape and np.isfinite(v).all() for v in out),'Nonfinite coefficients')
            return out,work
        except BaseException:
            self.valid=False;raise
    def seed(self,t,z,consumed_target,consumed_rate):
        """Seed once against same-run consumed outputs; reuse as slow stage0."""
        need(not self.seeded,'Already seeded')
        (a,r),work=self._evaluate(t,z,0)
        if not (bits(a,consumed_target) and bits(r,consumed_rate)):
            self.valid=False
            raise RuntimeError('Seed differs from consumed full operator')
        self.cached=(float(t),z.copy(),(r*(a-z)).copy(),work);self.seeded=True
    def full(self,t,z,side):
        need(self.seeded,'Seed first')
        self._check()
        if self.cached is not None:
            tc,x,f,work=self.cached
            need(float(t)==tc and bits(z,x),'First slow stage must be seed')
            self.cached=None;return f.copy(),work
        (a,r),work=self._evaluate(t,z,2);return r*(a-z),work
    def fast(self,t,z,side):
        need(self.seeded,'Seed first')
        (a,r),work=self._evaluate(t,z,1);return r*(a-z),work


def missed_audits(original_module):
    """Test the original MRI unchanged, on a fixed autonomous counterexample."""
    from scipy.integrate import quad
    T=125e-6;te=.9*T;L=(T-te)/T;c=.5*L*np.exp(-L)
    class Model:
        setup_edges=0
        def version(self):return 'fixed-test-model'
        def project(self,t,z,side):
            a=z.copy();u=(t-te)/T
            a[:2]=[0.,0.] if u<0 or (u==0 and side=='left') else [.5*np.exp(-u),.5*u*np.exp(-u)]
            return a
        def fast(self,t,z,side):return np.array([0.,0.,-z[2]/T]),0
        def full(self,t,z,side):
            f=self.fast(t,z,side)[0];f[2]+=z[1]*(c-z[1])/T;return f,1
    fun=lambda u:np.exp(-(L-u))*(.5*u*np.exp(-u))*(c-.5*u*np.exp(-u))
    ref,err=quad(fun,0.,L,epsabs=1e-18,epsrel=1e-12)
    exact=Model().project(T,np.zeros(3),'right');exact[2]=ref
    e=original_module.Ensayo(Model(),np.zeros(3),np.array([1,1,0],bool),(-np.inf,np.inf),0.,T,[te],60,6)
    report,high,low=e.run(exact)
    need(report['embedded_3_2']==0 and report['sampled_defect_normalized']==0 and report['endpoint_normalized']>1,'Falsifier not exercised')
    return {'reference_z':ref,'quadrature_error_estimate':err,'candidate_z':float(high[2]),
        'embedded':report['embedded_3_2'],'sampled_defect':report['sampled_defect_normalized'],
        'normalized_endpoint':report['endpoint_normalized'],'full_calls':report['full_calls'],
        'fast_calls':report['fast_calls'],'status':report['status'],
        'original_code_unchanged':True,'CUDA_executed':False}

if __name__=='__main__':
    import argparse,importlib.util,pathlib
    p=argparse.ArgumentParser();p.add_argument('--original',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    need(hashlib.sha256(a.original.read_bytes()).hexdigest()=='03ffdf6038772226d4797b837c5ed08275d470019925f697c56d5a23857a15fc','Original hash mismatch')
    a.out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('original_mri',a.original);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    result=missed_audits(m);result['source_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
