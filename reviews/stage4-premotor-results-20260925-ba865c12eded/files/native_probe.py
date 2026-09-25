"""Native fixed pulse and conservative all-query observations on generic rate rows.

Probe state is experimental apparatus, not an operative component of the animal.
All-query extrema include rejected numerical trials; only nonpositive upper bounds
can establish no positive target in any sampled accepted update.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true')


def need(ok,msg):
    if not ok:raise ValueError(msg)


def replace_once(s,a,b):
    need(s.count(a)==1,'Unexpected native source: '+a)
    return s.replace(a,b)


def instrument_source(source):
    helper='''
__device__ void probe_max(double* ptr, double value) {
    unsigned long long* bits=(unsigned long long*)ptr;
    unsigned long long old=*bits, assumed;
    while(value>__longlong_as_double(old)) {
        assumed=old;
        old=atomicCAS(bits,assumed,__double_as_longlong(value));
        if(old==assumed) break;
    }
}
'''
    s=replace_once(source,'double* target, double* rate) {',
      'double* target, double* rate, const int* probe_watch, const int* probe_sources, const int* probe_on, double* probe_dose, int* probe_locked, double* probe_anchor, double* probe_last, double* probe_maxarg, unsigned long long* probe_count, const int probe_n) {')
    before='target[row]=fmax(0.,tanh(gain[row]*(a+drive[row]-theta[row])));'
    after='''
        const int j=probe_watch[row];
        double extra=0.;
        if(j>=0 && probe_on[0] && probe_sources[j]) {
            if(!probe_locked[j]) {
                double u=gain[row]*(a+drive[row]-theta[row]);
                double t0=fmax(0.,tanh(u));
                double tstar=t0+0.25*(1.-t0);
                double delta=(atanh(tstar)-u)/gain[row];
                probe_dose[j]=delta;
                probe_anchor[j]=a;
                probe_anchor[probe_n+j]=drive[row];
                probe_anchor[2*probe_n+j]=t0;
                probe_anchor[3*probe_n+j]=tstar;
                probe_locked[j]=1;
            }
            extra=probe_dose[j];
        }
        if(extra==0.) { target[row]=fmax(0.,tanh(gain[row]*(a+drive[row]-theta[row]))); }
        else { target[row]=fmax(0.,tanh(gain[row]*(a+(drive[row]+extra)-theta[row]))); }
        if(j>=0) {
            const double u=gain[row]*(a+(drive[row]+extra)-theta[row]);
            probe_last[j]=a;
            probe_last[probe_n+j]=drive[row];
            probe_last[2*probe_n+j]=extra;
            probe_last[3*probe_n+j]=target[row];
            probe_max(&probe_maxarg[j],u);
            atomicAdd(&probe_count[j],1ULL);
            if(target[row]>0.) atomicAdd(&probe_count[probe_n+j],1ULL);
        }
'''
    return helper+replace_once(s,before,after)


class KernelProbe:
    def __init__(self,kernel,args):self.kernel,self.args=kernel,args
    def __call__(self,grid,block,args,**kwargs):return self.kernel(grid,block,(*args,*self.args),**kwargs)


class NativeProbe:
    def __init__(self,h,ids,source_ids,out):
        import cupy as cp
        import gpu_visual_brain
        self.h=h;self.ids=np.asarray(ids,dtype=np.int64);self.n=len(ids);self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=False)
        self.rows=np.searchsorted(h.brain.node_ids,self.ids)
        need(len(set(ids))==len(ids) and np.array_equal(h.brain.node_ids[self.rows],self.ids),'Panel identity')
        need(not np.any(h.visual_mask[self.rows]),'Panel is not ordinary rate')
        for name in ('_pn_rows','_orn_rows','regional_rows','_cvn7_rows'):
            need(not len(np.intersect1d(self.rows,getattr(h,name,[]))),'Panel has a special owner: '+name)
        for name in ('kcgamma_output_manifest','regional_manifest','pnkc_receptor_manifest'):
            m=getattr(h,name,{})
            need(not m.get('enabled') or not len(np.intersect1d(self.rows,m.get('target_rows',[]))),'Panel replacement: '+name)
        need(set(source_ids)<=set(ids),'Source outside fixed panel')
        watch=np.full(h.brain.n_neurons,-1,dtype=np.int32);watch[self.rows]=np.arange(self.n,dtype=np.int32)
        self.sources=np.isin(self.ids,source_ids)
        self.watch=cp.asarray(watch);self.flags=cp.asarray(self.sources,dtype=cp.int32);self.on=cp.zeros(1,dtype=cp.int32)
        self.dose=cp.zeros(self.n,dtype=cp.float64);self.locked=cp.zeros(self.n,dtype=cp.int32)
        self.anchor=cp.full((4,self.n),cp.nan,dtype=cp.float64)
        self.last=cp.full((4,self.n),cp.nan,dtype=cp.float64)
        self.maximum=cp.full(self.n,-cp.inf,dtype=cp.float64)
        self.count=cp.zeros((2,self.n),dtype=cp.uint64)
        source=instrument_source(gpu_visual_brain.CUDA_SOURCE)
        self.kernel=cp.RawKernel(source,'coefficient',options=OPTIONS)
        self.args=(self.watch,self.flags,self.on,self.dose,self.locked,self.anchor,self.last,self.maximum,self.count,np.int32(self.n))
        self.records=[];self.old_kernel=h.kernel
        self.old_coeff=h.coefficients_gpu
        self.own_coeff=h.__dict__.get('coefficients_gpu')
        self.had_coeff='coefficients_gpu' in h.__dict__
        self.final_coeff=None;self.build_calls=0
        def coefficients(state,drive,light):
            target,rate=self.old_coeff(state,drive,light)
            self.final_coeff=(target,rate)
            self.build_calls+=1
            return target,rate
        h.coefficients_gpu=coefficients
        h.kernel=KernelProbe(self.kernel,self.args)
        (self.out/'kernel.cu').write_text(source)
        (self.out/'METADATA.json').write_text(json.dumps(dict(ids=self.ids.tolist(),rows=self.rows.tolist(),source_ids=list(source_ids),
            theta=h.rate_theta[self.rows].tolist(),gain=h.rate_gain[self.rows].tolist(),tau=h.tau[self.rows].tolist(),caps=h.caps[self.rows].tolist(),
            original_kernel_sha256=hashlib.sha256(gpu_visual_brain.CUDA_SOURCE.encode()).hexdigest(),
            instrumented_kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),
            coverage='All executed generic coefficient queries, including rejected trials and initial graph setup. Maxima upper-bound sampled accepted maxima; not accepted-time statistics.'),indent=2)+'\n')

    def begin(self,phase,step):
        self.phase,self.step=phase,step
        self.on.fill(int(phase=='ensayo' and 21<=step<=60 and np.any(self.sources)))
        self.last.fill(np.nan);self.maximum.fill(-np.inf);self.count.fill(0)

    def sample(self):
        import cupy as cp
        raw=cp.asnumpy(self.last);maximum=cp.asnumpy(self.maximum);counts=cp.asnumpy(self.count)
        h=self.h
        need(np.isfinite(raw).all() and np.isfinite(maximum).all() and np.all(counts[0]>0),'Missing/nonfinite native observation')
        reconstructed=np.maximum(0.,np.tanh(h.rate_gain[self.rows]*(raw[0]+(raw[1]+raw[2])-h.rate_theta[self.rows])))
        need(np.max(abs(reconstructed-raw[3]))<=1e-12,'Consumed native target differs')
        need(self.final_coeff is not None and self.build_calls==18,'Unexpected coefficient graph construction')
        actual=cp.asnumpy(self.final_coeff[0][self.rows])
        rate=cp.asnumpy(self.final_coeff[1][self.rows])
        need(np.array_equal(actual,raw[3]),'A later owner replaced the observed target')
        need(np.max(abs(rate-1./h.tau[self.rows]))<=1e-12,'A later owner replaced the observed rate')
        need(np.all(counts[1]<=counts[0]),'Invalid query counters')
        q=h.release()[self.rows]
        need(np.isfinite(q).all() and np.all((q>=0)&(q<=1)),'Panel state outside domain')
        self.records.append(dict(phase=self.phase,step=self.step,time_ns=int(h.time_ns),q=q.copy(),raw=raw,maximum=maximum,counts=counts))
        if self.phase=='ensayo' and self.step==21 and np.any(self.sources):
            self.validate_dose()

    def validate_dose(self):
        import cupy as cp
        from chatgpt_pulse_original import preparar_pulso
        anchor=cp.asnumpy(self.anchor);dose=cp.asnumpy(self.dose);locked=cp.asnumpy(self.locked)
        s=self.sources;need(np.all(locked[s]==1) and np.all(locked[~s]==0),'Dose initialization scope')
        donor=preparar_pulso(tuple(self.ids[s].tolist()),anchor[0,s]+anchor[1,s],self.h.rate_theta[self.rows[s]],self.h.rate_gain[self.rows[s]])
        need(np.allclose(dose[s],donor.corriente_adicional,rtol=1e-12,atol=1e-12),'Native/ChatGPT dose mismatch')
        need(np.allclose(anchor[3,s],donor.target_objetivo_instantaneo,rtol=1e-10,atol=1e-12),'Native target dose mismatch')
        (self.out/'DOSE.json').write_text(json.dumps(dict(ids=self.ids[s].tolist(),raw_input=anchor[0,s].tolist(),preexisting_drive=anchor[1,s].tolist(),
            additional_drive=dose[s].tolist(),initial_target=anchor[2,s].tolist(),instantaneous_target=anchor[3,s].tolist(),
            computed_once_in_native_consumer=True,not_physiological_calibration=True),indent=2,allow_nan=False)+'\n')

    def save(self):
        if not self.records:return
        np.savez_compressed(self.out/'PANEL.npz',ids=self.ids,**{k:np.asarray([r[k] for r in self.records]) for k in self.records[0]})

    def close(self):
        self.h.kernel=self.old_kernel
        if self.had_coeff:self.h.coefficients_gpu=self.own_coeff
        else:self.h.__dict__.pop('coefficients_gpu',None)
