"""Global resident RK4 step doubling; CPU reference; transactional model edits."""
from __future__ import annotations
import copy,json,time,hashlib,sys,platform
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from model import Model,Unsupported,require,digest

PROFILES={'fast':(1e-3,1e-5),'precise':(1e-7,1e-9)}

KERNELS=r'''
#include <math_constants.h>
extern "C" __global__ void project(const double* val,const int* col,const long long* ptr,const double* x,const double* bias,double* y,int n,int* flag) {
 int lane=threadIdx.x&31; int row=(blockDim.x*blockIdx.x+threadIdx.x)/32;
 if(row>=n) return;
 double s=0.; for(long long k=ptr[row]+lane;k<ptr[row+1];k+=32) s+=val[k]*x[col[k]];
 for(int offset=16;offset;offset/=2) s+=__shfl_down_sync(0xffffffff,s,offset);
 if(lane==0) {s+=bias[row];if(!isfinite(s)) atomicOr(flag,1);y[row]=s;}
}
extern "C" __global__ void stage(const double* y,const double* k,double* z,const double* clock,double frac,int n,int* flag) {
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n) return;
 double v=y[i]+clock[1]*frac*k[i]; if(!isfinite(v)) atomicOr(flag,1); z[i]=v;
}
extern "C" __global__ void finish(const double* y,const double* a,const double* b,const double* c,const double* d,double* z,const double* clock,double frac,int n,int* flag) {
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n) return;
 double v=y[i]+(clock[1]*frac/6)*(a[i]+2*b[i]+2*c[i]+d[i]); if(!isfinite(v)) atomicOr(flag,1); z[i]=v;
}
extern "C" __global__ void expstage(const double* y,const double* f,const double* d,double* z,const double* clock,double frac,int n,int* flag) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 double h=clock[1]*frac;double v=y[i]+h*om_exprel(h*d[i])*f[i];
 if(!isfinite(v))atomicOr(flag,1);z[i]=v;
}
extern "C" __global__ void expfinish(const double* y,const double* mid,const double* f,const double* d,double* z,const double* clock,double frac,int n,int* flag) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 double h=clock[1]*frac;double v=y[i]+h*om_exprel(h*d[i])*(f[i]+d[i]*(y[i]-mid[i]));
 if(!isfinite(v))atomicOr(flag,1);z[i]=v;
}
extern "C" __global__ void errornorm(const double* y,const double* fine,const double* coarse,const double* scale,double rtol,double atol,double divisor,double* err,int n,int* flag) {
 __shared__ double mx[256]; int i=blockDim.x*blockIdx.x+threadIdx.x;
 double e=0;
 if(i<n) {
   double a=y[i],b=fine[i],c=coarse[i];
   if(!isfinite(a)||!isfinite(b)||!isfinite(c)) atomicOr(flag,1);
   e=fabs(b-c)/(divisor*(atol*scale[i]+rtol*fmax(fabs(a),fabs(b))));
   if(!isfinite(e)) {atomicOr(flag,1);e=0;}
 }
 mx[threadIdx.x]=e; __syncthreads();
 for(int s=128;s;s/=2){if(threadIdx.x<s) mx[threadIdx.x]=fmax(mx[threadIdx.x],mx[threadIdx.x+s]);__syncthreads();}
 if(threadIdx.x==0) atomicMax((unsigned long long*)err,__double_as_longlong(mx[0]));
}
extern "C" __global__ void summary(const double* err,const int* flag,double* out) {out[0]=err[0];out[1]=flag[0];}
'''

class GPU:
    def __init__(self,model,profile,algorithm):
        import cupy as cp
        if not model.diagonal_mass: raise Unsupported('GPU general non-diagonal mass backend remains unimplemented')
        self.cp=cp;self.model=model;self.profile=profile;self.algorithm=algorithm;self.rtol,self.atol=PROFILES[profile]
        # Initialization, graph capture, all input mutations and launches use this SAME stream.
        self.stream=cp.cuda.Stream(non_blocking=True)
        with self.stream:
            self.x=cp.asarray(model.initial);self.par=cp.asarray(model.parameters)
            self.scale=cp.asarray(model.scale);self.bias=cp.asarray(model.port_bias)
            self.mass=cp.asarray(model.mass_diag);self.clamp=cp.asarray(model.clamp_mask)
            require(model.nout < 2**31 and model.nin < 2**31, '32-bit column domain exceeded')
            self.cdata=cp.asarray(model.connection.data)
            self.cindices=cp.asarray(model.connection.indices,dtype=cp.int32)
            self.cindptr=cp.asarray(model.connection.indptr,dtype=cp.int64)
            self.outputs=cp.empty(model.nout);self.inputs=cp.empty(model.nin)
            self.diag=cp.empty(model.n);self.z=cp.empty(model.n);self.coarse=cp.empty(model.n);self.mid=cp.empty(model.n);self.fine=cp.empty(model.n)
            self.k=[cp.empty(model.n) for _ in range(4)]
            self.clock=cp.asarray([0.,1e-4]);self.err=cp.zeros(1);self.flag=cp.zeros(1,dtype=cp.int32);self.status=cp.zeros(2)
            mod=cp.RawModule(code=model.cuda_source()+KERNELS)
            self.outkernels=[mod.get_function(f'output_{i}') for i in range(len(model.pops))]
            self.rhskernels=[mod.get_function(f'rhs_{i}') for i in range(len(model.pops))]
            self.stage=mod.get_function('stage');self.finish=mod.get_function('finish');self.project=mod.get_function('project')
            self.errornorm=mod.get_function('errornorm');self.summary=mod.get_function('summary')
            self.expstage=mod.get_function('expstage');self.expfinish=mod.get_function('expfinish')
            self.grid=((model.n+255)//256,);self.block=(256,)
            # Warm library kernels/handles before capture. Trial never commits x.
            self.trial();self.stream.synchronize()
            self.stream.begin_capture();self.trial();self.graph=self.stream.end_capture()
        self.stream.synchronize()
        self.transfers={'control_H2D_bytes':0,'control_D2H_bytes':0,'scan_D2H_bytes':0}
        self.accepted=0;self.rejected=0

    def rhs(self,y,tfrac,out,diagonal=False):
        cp=self.cp
        for p,k in zip(self.model.pops,self.outkernels):
            k(((p['n']+255)//256,),self.block,(y,self.par,self.inputs,self.outputs,self.clock,np.float64(tfrac),self.mass,self.clamp,self.flag,self.diag,np.int32(diagonal)))
        if self.model.nin:
            self.project(((self.model.nin+7)//8,),self.block,(self.cdata,self.cindices,self.cindptr,self.outputs,self.bias,self.inputs,np.int32(self.model.nin),self.flag))
        for p,k in zip(self.model.pops,self.rhskernels):
            k(((p['n']+255)//256,),self.block,(y,self.par,self.inputs,out,self.clock,np.float64(tfrac),self.mass,self.clamp,self.flag,self.diag,np.int32(diagonal)))

    def rk4(self,y,out,frac,tstart):
        a,b,c,d=self.k;n=np.int32(self.model.n)
        self.rhs(y,tstart,a)
        self.stage(self.grid,self.block,(y,a,self.z,self.clock,np.float64(frac/2),n,self.flag))
        self.rhs(self.z,tstart+frac/2,b)
        self.stage(self.grid,self.block,(y,b,self.z,self.clock,np.float64(frac/2),n,self.flag))
        self.rhs(self.z,tstart+frac/2,c)
        self.stage(self.grid,self.block,(y,c,self.z,self.clock,np.float64(frac),n,self.flag))
        self.rhs(self.z,tstart+frac,d)
        self.finish(self.grid,self.block,(y,a,b,c,d,out,self.clock,np.float64(frac),n,self.flag))

    def exponential(self,y,out,frac,tstart):
        a,b,_,_=self.k;n=np.int32(self.model.n)
        self.rhs(y,tstart,a,True)
        self.expstage(self.grid,self.block,(y,a,self.diag,self.z,self.clock,np.float64(frac/2),n,self.flag))
        self.rhs(self.z,tstart+frac/2,b,True)
        self.expfinish(self.grid,self.block,(y,self.z,b,self.diag,out,self.clock,np.float64(frac),n,self.flag))

    def trial(self):
        self.flag.fill(0);self.err.fill(0)
        advance=self.rk4 if self.algorithm=='A' else self.exponential
        advance(self.x,self.coarse,1.,0.)
        advance(self.x,self.mid,.5,0.)
        advance(self.mid,self.fine,.5,.5)
        self.errornorm(self.grid,self.block,(self.x,self.fine,self.coarse,self.scale,np.float64(self.rtol),np.float64(self.atol),np.float64(15 if self.algorithm=='A' else 3),self.err,np.int32(self.model.n),self.flag))
        self.summary((1,),(1,),(self.err,self.flag,self.status))

    def attempt(self,t,h):
        with self.stream:
            self.clock.set(np.asarray([t,h],dtype=np.float64));self.graph.launch(self.stream)
            status=self.status.get(stream=self.stream)
            self.transfers['control_H2D_bytes']+=16;self.transfers['control_D2H_bytes']+=16
            require(np.isfinite(status).all() and status[1]==0,'nonfinite intermediate: step not committed')
            error=float(status[0])
            if error<=1:
                self.cp.copyto(self.x,self.fine);self.accepted+=1
            else: self.rejected+=1
        return error

    def read(self,indices=None):
        with self.stream:
            result=self.x.get(stream=self.stream) if indices is None else self.x[indices].get(stream=self.stream)
            self.transfers['scan_D2H_bytes']+=result.nbytes
        return result

    def write(self,y):
        with self.stream: self.x.set(y)
        self.stream.synchronize()

    def rhs_read(self,t,y):
        with self.stream:
            state=self.cp.asarray(y);self.clock.set(np.array([t,0.]));self.flag.fill(0)
            self.rhs(state,0,self.k[0]);out=self.k[0].get(stream=self.stream);flag=int(self.flag.get(stream=self.stream)[0])
        require(flag==0,'GPU RHS nonfinite')
        return out

class Engine:
    def __init__(self,spec,backend='gpu',profile='precise',algorithm='A'):
        require(backend in ('cpu','gpu') and profile in PROFILES and algorithm in ('A','C'),'invalid execution configuration')
        start=time.perf_counter();self.model=Model(spec);self.backend=backend;self.profile=profile;self.algorithm=algorithm
        self.tolerances=tuple(PROFILES[profile]);self.gpu=GPU(self.model,profile,algorithm) if backend=='gpu' else None
        self.y=self.model.initial.copy() if backend=='cpu' else None
        self.t=0.;self.next_h=1e-4;self.edits=[];self.setup_s=time.perf_counter()-start
        self.cpu_nfev=0;self.cpu_method='DOP853'

    def read(self): return self.gpu.read() if self.gpu else self.y.copy()

    def advance(self,end,wall_limit_s=300):
        require(math_finite(end) and end>=self.t,'time must be finite and monotonic')
        start=time.perf_counter()
        if end==self.t:return
        if self.gpu:
            attempts=0
            while self.t<end:
                require(time.perf_counter()-start<wall_limit_s and attempts<200000,'finite advance budget exceeded')
                h=min(self.next_h,end-self.t,.02)
                require(h>max(1e-14,abs(self.t)*1e-15),'step underflow')
                err=self.gpu.attempt(self.t,h);attempts+=1
                factor=3. if err==0 else min(3.,max(.2,.9*err**(-(.2 if self.algorithm=='A' else 1/3))))
                self.next_h=h*factor
                if err<=1:
                    self.t=end if h==end-self.t else self.t+h
        else:
            rtol,atol=self.tolerances
            def rhs(t,y):
                require(time.perf_counter()-start<wall_limit_s,'CPU advance budget exceeded')
                return self.model.rhs(t,y)
            sol=solve_ivp(rhs,(self.t,end),self.y,method=self.cpu_method,rtol=rtol,atol=atol*self.model.scale)
            require(sol.success,'CPU integrator failed: '+sol.message)
            self.y=sol.y[:,-1].copy();self.t=end;self.cpu_nfev+=sol.nfev

    def scan(self,population,state,cells=None):
        p=self.model.by_id[population];sl=p['states'][state]
        ids=p['ids'];idx=np.arange(sl.start,sl.stop)
        if cells is not None:
            lookup={int(v):i for i,v in enumerate(ids)};positions=[lookup[v] for v in cells]
            ids=ids[positions];idx=idx[positions]
        values=self.gpu.read(idx) if self.gpu else self.y[idx].copy()
        desc=next(p for p in self.model.spec['populations'] if p['id']==population)['states'][state]
        return {'time':self.t,'population':population,'state':state,'cell_ids':ids.tolist(),'unit':desc['unit'],'values':values.tolist(),'model_sha256':self.model.identity}

    def replace(self,new_spec):
        # Prepare/validate/compile outside the live object. Failure leaves EVERYTHING live unchanged.
        fresh=Engine(new_spec,self.backend,self.profile,self.algorithm)
        old=self.read();new=fresh.model.initial.copy()
        for name,p in fresh.model.by_id.items():
            if name not in self.model.by_id: continue
            op=self.model.by_id[name]
            _,oi,ni=np.intersect1d(op['ids'],p['ids'],return_indices=True)
            for key,sl in p['states'].items():
                if key in op['states']:
                    oldsl=op['states'][key]
                    require(fresh.model.units[sl.start]==self.model.units[oldsl.start],'state identity cannot change unit')
                    new[sl.start+ni]=old[oldsl.start+oi]
        new[fresh.model.clamp_mask]=fresh.model.clamp_values[fresh.model.clamp_mask]
        require(np.isfinite(new).all(),'nonfinite migrated state')
        fresh.model.raw_rhs(self.t,new)  # Current time and migrated state, not descriptor initial state.
        if fresh.gpu:fresh.gpu.rhs_read(self.t,new)
        if fresh.gpu: fresh.gpu.write(new)
        else: fresh.y=new
        receipt={'time':self.t,'before':self.model.identity,'after':fresh.model.identity,'kind':'atomic descriptor replacement','setup_s':fresh.setup_s}
        self.model,self.gpu,self.y=fresh.model,fresh.gpu,fresh.y
        self.next_h=1e-4;self.edits.append(receipt)
        return receipt

    def execution_contract(self):
        import scipy
        h=Path(__file__).resolve().parent
        contract={'schema':2,'dtype':'float64','numeric_IR':'fp64-v2','ordered_layout':self.model.layout,
                  'source_hashes':{n:hashlib.sha256((h/n).read_bytes()).hexdigest() for n in ('model.py','runtime.py','autodiff.py')},
                  'backend':self.backend,'algorithm':self.algorithm,'profile':self.profile,'effective_tolerances':list(self.tolerances),
                  'cpu_method':self.cpu_method,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__}
        if self.gpu:
            cp=self.gpu.cp
            contract.update(cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
                            cuda_source=hashlib.sha256((self.model.cuda_source()+KERNELS).encode()).hexdigest(),device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
        return contract

    def checkpoint(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        state=self.read();np.save(path/'state.npy',state)
        meta={'version':2,'execution_contract':self.execution_contract(),'descriptor_sha256':self.model.descriptor_hash,'spec':self.model.spec,'model_sha256':self.model.identity,'backend':self.backend,'profile':self.profile,'algorithm':self.algorithm,'time':self.t,'next_h':self.next_h,'edits':self.edits,'cpu_method':self.cpu_method,'state_sha256':__import__('hashlib').sha256((path/'state.npy').read_bytes()).hexdigest(),'rng':'absent: deterministic domain','event_queue':'absent: only external edits at explicit advance boundaries'}
        raw=(json.dumps(meta,indent=2,allow_nan=False)+'\n').encode()
        (path/'checkpoint.json').write_bytes(raw)
        (path/'checkpoint.sha256').write_text(__import__('hashlib').sha256(raw).hexdigest()+'\n')

    @classmethod
    def restore(cls,path):
        path=Path(path);raw=(path/'checkpoint.json').read_bytes()
        require(__import__('hashlib').sha256(raw).hexdigest()==(path/'checkpoint.sha256').read_text().strip(),'checkpoint metadata corruption')
        meta=json.loads(raw)
        require(meta['version']==2,'checkpoint schema requires explicit migration')
        require(__import__('hashlib').sha256((path/'state.npy').read_bytes()).hexdigest()==meta['state_sha256'],'checkpoint state corruption')
        require(digest(meta['spec'])==meta['descriptor_sha256'],'checkpoint descriptor corruption')
        e=cls(meta['spec'],meta['backend'],meta['profile'],meta['algorithm'])
        require(e.model.identity==meta['model_sha256'],'checkpoint executable layout/identity mismatch')
        require(meta['cpu_method'] in ('DOP853','Radau','BDF'),'unsupported CPU method')
        e.cpu_method=meta['cpu_method']
        require(e.execution_contract()==meta['execution_contract'],'checkpoint execution contract changed; explicit migration required')
        y=np.load(path/'state.npy',allow_pickle=False)
        require(y.dtype==np.float64 and y.shape==(e.model.n,) and np.isfinite(y).all(),'invalid checkpoint state')
        require(math_finite(meta['time']) and meta['time']>=0 and math_finite(meta['next_h']) and meta['next_h']>0,'invalid checkpoint controller')
        require(np.array_equal(y[e.model.clamp_mask],e.model.clamp_values[e.model.clamp_mask]),'clamp state corruption')
        if e.gpu:e.gpu.write(y)
        else:e.y=y.copy()
        e.t=meta['time'];e.next_h=meta['next_h'];e.edits=meta['edits'];e.cpu_method=meta['cpu_method']
        return e

def math_finite(x): return isinstance(x,(float,int)) and np.isfinite(x)
