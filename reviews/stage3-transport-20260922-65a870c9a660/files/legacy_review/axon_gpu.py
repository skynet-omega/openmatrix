"""GPU event/filter publisher, synced fully at each CNS coupling boundary."""
import cupy as cp
import numpy as np

CODE=r'''
extern "C" __global__ void axon(int N,double dt,double rest,double ts,const double*v,const double*caps,const double*tau,const double*gain,
 double*q,double*s,double*last,double*slope0,double*trough,long long*counts,long long*clipped){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=N*12)return;int cell=j/12,port=j%12;
 double tq=tau[cell],eq=exp(-dt/tq),es=exp(-dt/ts),factor;
 if(fabs(tq-ts)<=1e-15)factor=dt/ts*es;else factor=tq*(eq-es)/(tq-ts);
 double voltage=v[cell*17+5+port]+rest,slope=voltage-last[j];bool peak=slope0[j]>0.&&slope<=0.;
 bool event=peak&&last[j]>-40.&&last[j]-trough[j]>=20.;
 double snext=s[j]*es+gain[cell]*q[j]*factor,qn=q[j]*eq+(event?1./(caps[cell]*tq):0.);
 clipped[j]+=(qn>1.);q[j]=fmin(qn,1.);s[j]=snext;counts[j]+=event;
 trough[j]=peak?voltage:fmin(trough[j],voltage);last[j]=voltage;slope0[j]=slope;
}
'''
KERNEL=cp.RawKernel(CODE,'axon',options=('--fmad=false',))

class Publisher:
    def __init__(self,wrapper,batch):
        self.wrapper=wrapper;self.batch=batch;p=wrapper.publisher;p.validate()
        if not np.array_equal(p.caps,batch.host(batch.caps)) or not np.array_equal(p.tau,batch.host(batch.tau)):raise ValueError('Axonal GPU kinetic mismatch')
        self.fields={k:cp.asarray(v) for k,v in p.state.items() if isinstance(v,np.ndarray)}
        self.gain=cp.asarray(wrapper.gain());self.elapsed=0
    def __call__(self,ns):
        b=self.batch;d=self.fields;p=self.wrapper.publisher
        KERNEL(((b.n*12+255)//256,),(256,), (np.int32(b.n),np.float64(ns*1e-9),np.float64(b.rest),np.float64(p.synaptic_tau),
            b.delta,b.caps,b.tau,self.gain,d['q'],d['s'],d['last_voltage'],d['previous_slope'],d['trough'],d['counts'],d['clipped']))
        self.elapsed+=ns
    def finish(self,dt_ns):
        if self.elapsed!=dt_ns:raise ValueError('Incomplete GPU axonal clock')
        p=self.wrapper.publisher
        state={k:cp.asnumpy(v) for k,v in self.fields.items()};state['elapsed_ns']=p.state['elapsed_ns']+dt_ns
        previous=p.state;p.state=state
        try:p.validate()
        except BaseException:p.state=previous;raise
