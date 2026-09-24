"""Same hybrid equations, FP64 CUDA with fixed warp reductions, no atomics.

This changes summation order relative to the CPU reference. It is an explicit
numerical backend, validated against that reference, never biological tuning.
The CPU arrays remain the authoritative portable checkpoint representation.
"""
import copy
import cupy as cp
import numpy as np
from hybrid_visual_brain import HybridVisualBrain, exponential_midpoint

CUDA_SOURCE = r'''
extern "C" __global__ void coefficient(
 int n, const long long* ptr, const int* idx, const double* w,
 const double* release, const double* caps, const bool* visual,
 const double* tau, const double* gain, const double* theta,
 const double* drive, const double* photo, double scale, bool connected,
 double* target, double* rate) {
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 if(row>=n) return;
 double a=0., b=0.;
 for(long long e=ptr[row]+lane; e<ptr[row+1]; e+=32) {
   int c=idx[e];
   if(visual[row]) {
     double v=(w[e]*scale)*release[c];
     if(v>=0.) a+=v; else b-=v;
   } else if(connected || !visual[c]) a+=w[e]*(release[c]*caps[c]);
 }
 for(int d=16;d>0;d/=2) {
   a+=__shfl_down_sync(0xffffffff,a,d);
   b+=__shfl_down_sync(0xffffffff,b,d);
 }
 if(lane==0) {
   if(visual[row]) {
     a+=photo[row]; double total=1.+a+b;
     target[row]=(.25+a)/total;rate[row]=total/tau[row];
   } else {
     target[row]=fmax(0.,tanh(gain[row]*(a+drive[row]-theta[row])));
     rate[row]=1./tau[row];
   }
 }
}
'''


class GpuVisualBrain(HybridVisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_fp64_cuda_v1'

    def _build(self):
        super()._build()
        self.cuda = dict(indptr=cp.asarray(self.brain.W.indptr,dtype=cp.int64),
            indices=cp.asarray(self.brain.W.indices,dtype=cp.int32),
            weights=cp.asarray(self.weights64), caps=cp.asarray(self.caps),
            visual=cp.asarray(self.visual_mask), tau=cp.asarray(self.tau),
            gain=cp.asarray(self.rate_gain), theta=cp.asarray(self.rate_theta),
            vi=cp.asarray(self.vi), pi=cp.asarray(self.pi))
        self.kernel = cp.RawKernel(CUDA_SOURCE, 'coefficient',
            # CuPy appends ftz=true for single precision; this kernel contains
            # only double arithmetic, for which CUDA retains subnormals.
            options=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true'))

    @classmethod
    def adopt(cls, reference):
        state = reference.state_dict()
        state['schema'] = cls.SCHEMA
        return cls.from_state(reference.brain, state)

    def sync_plastic_weights(self, plasticity):
        super().sync_plastic_weights(plasticity)
        self.cuda['weights'][cp.asarray(plasticity.positions)] = cp.asarray(self.weights64[plasticity.positions])

    def coefficients_gpu(self, state, drive, light):
        self.statistics['evaluations'] += 1
        n,m=self.brain.n_neurons,len(self.pi)
        p,c=self.parameters,self.cuda
        release=state[:n].copy()
        release[c['vi']]=cp.clip((80.*release[c['vi']]-15.)/40.,0.,1.)
        fast,adapt=state[n:n+m],state[n+m:]
        photo=cp.zeros(n,dtype=cp.float64)
        photo[c['pi']]=p['photoconductance_max']*fast/(p['photo_half']+p['adaptation_strength']*adapt+fast)
        target,rate=cp.empty_like(state),cp.empty_like(state)
        args=(np.int32(n),c['indptr'],c['indices'],c['weights'],release,c['caps'],c['visual'],
            c['tau'],c['gain'],c['theta'],drive,photo,np.float64(p['conductance_per_stored_weight']),
            np.bool_(self.visual_output_connected),target,rate)
        self.kernel(((n*32+255)//256,),(256,),args)
        target[n:n+m],rate[n:n+m]=light,1./p['photo_fast_tau_s']
        target[n+m:],rate[n+m:]=fast,1./p['photo_adaptation_tau_s']
        return target,rate

    def advance(self, dt_ns, drive, light):
        b,p=self.brain,self.parameters
        drive,light=np.asarray(drive,dtype=float),np.asarray(light,dtype=float)
        if type(dt_ns) is not int or dt_ns<=0 or b.time_ns!=self.time_ns:
            raise ValueError('Invalid neural duration/clock')
        if drive.shape!=(b.n_neurons,) or light.shape!=(len(self.pi),) or not np.isfinite(drive).all() or not np.isfinite(light).all() or np.any((light<0)|(light>1)) or np.any(drive[self.vi]):
            raise ValueError('Invalid sensory input')
        y=cp.asarray(self.state)
        dg,lg=cp.asarray(drive),cp.asarray(light)
        coefficients=lambda z:self.coefficients_gpu(z,dg,lg)
        def midpoint(z,h):
            target,rate=coefficients(z)
            middle=z+(-cp.expm1(-.5*h*rate))*(target-z)
            target,rate=coefficients(middle)
            return z+(-cp.expm1(-h*rate))*(target-z)
        remaining,attempts=dt_ns,0
        while remaining:
            attempts+=1
            if attempts>10000:
                raise RuntimeError('Numerical work bound exceeded')
            h_ns=min(remaining,self.next_step_ns,p['maximum_step_ns'])
            h=h_ns*1e-9
            full=midpoint(y,h)
            half=midpoint(midpoint(y,.5*h),.5*h)
            scale=p['atol']+p['rtol']*cp.maximum(cp.abs(full),cp.abs(half))
            error=float(cp.max(cp.abs(half-full)/(3.*scale)))
            if not np.isfinite(error) or not bool(cp.isfinite(half).all()):
                raise FloatingPointError('Nonfinite neural integration')
            if error<=1.:
                if bool(cp.any((half<0)|(half>1))):
                    raise FloatingPointError('Neural state left its domain')
                y=half
                remaining-=h_ns
                self.statistics['accepted']+=1
                self.statistics['minimum_accepted_ns']=min(self.statistics['minimum_accepted_ns'],h_ns)
                self.statistics['maximum_accepted_error']=max(self.statistics['maximum_accepted_error'],error)
                self.next_step_ns=min(p['maximum_step_ns'],h_ns*2 if error<.1 else h_ns)
            else:
                self.statistics['rejected']+=1
                if h_ns//2<p['minimum_step_ns']:
                    raise FloatingPointError('Requested accuracy unattainable')
                self.next_step_ns=h_ns//2
        self.state=cp.asnumpy(y)
        self.time_ns+=dt_ns
        self.publish_rates()

    @staticmethod
    def backend_identity():
        return dict(cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
            device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            reduction='one warp per CSR row; fixed binary tree; no atomics; FP64; no FMA or fast math')
