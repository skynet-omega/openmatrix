"""Time-resolved event filter ownership inside all recurrent integration stages.

Uses the unchanged spatial/LIF physical update for each block. Only generic
KC transmission is integrated analytically; other interfaces remain explicit.
No order or long-horizon accuracy is presumed from this local exact flow.
"""
import numpy as np
import cupy as cp
import kc_spatial_brain as spatial
import kc_adaptive
from prosthetic_olfactory_brain import GpuProstheticOlfactoryBrain
from event_waveform import Waveform,lif_record


class EventCoupling:
    def __init__(self,brain):
        self.brain=brain;self.active=None;self.batch=brain._spatial_batch
        while hasattr(self.batch,'base'):self.batch=self.batch.base
        self.rows=brain._kc_rows;self.rgpu=cp.asarray(self.rows);self.sgpu=cp.asarray(brain.transmission_start+self.rows)
        self.gamma=np.searchsorted(self.rows,brain._spatial_inputs.rows).astype(np.int64)
        self.gamma_local=set(int(x) for x in self.gamma)
        self.other=brain._spatial_other;self.report={'blocks':0,'events':0,'max_q_reconstruction_difference':0.,'global_trials':0,'time_resolved':True}
        self.audit=[]

    def project(self,state,t):
        q,s=self.active.at(t,cp);out=state.copy();out[self.rgpu]=q;out[self.sgpu]=s;return out

    def step(self,brain,dt_ns,drive,light):
        if self.active is None:raise RuntimeError('Missing physical KC event waveform')
        drive,light=brain._validated_inputs(dt_ns,drive,light)
        p=brain.parameters;y=cp.asarray(brain.state);dg,lg=cp.asarray(drive),cp.asarray(light)
        remaining=dt_ns;used=0;attempts=0;norm_size=brain._norm_size()
        def coeff(z,t):
            z=self.project(z,t);target,rate=brain.coefficients_gpu(z,dg,lg)
            target[self.rgpu]=z[self.rgpu];target[self.sgpu]=z[self.sgpu]
            rate[self.rgpu]=0.;rate[self.sgpu]=0.
            return z,target,rate
        def midpoint(z,ns,start):
            dt=ns*1e-9;t=start*1e-9
            z,target,rate=coeff(z,t);middle=z+(-cp.expm1(-.5*dt*rate))*(target-z)
            _,target,rate=coeff(middle,t+.5*dt)
            return self.project(z+(-cp.expm1(-dt*rate))*(target-z),t+dt)
        while remaining:
            attempts+=1;self.report['global_trials']+=1
            if attempts>10000:raise RuntimeError('Event-aware CNS work limit')
            ns=min(remaining,brain.next_step_ns,p['maximum_step_ns'])
            full=midpoint(y,ns,used);half=midpoint(midpoint(y,.5*ns,used),.5*ns,used+.5*ns)
            scale=p['atol']+p['rtol']*cp.maximum(cp.abs(full[:norm_size]),cp.abs(half[:norm_size]))
            error=float(cp.max(cp.abs(half[:norm_size]-full[:norm_size])/(3.*scale)))
            if not np.isfinite(error) or not bool(cp.isfinite(half).all()):raise FloatingPointError('Nonfinite event-aware CNS state')
            if error<=1:
                if bool(cp.any((half<0)|(half>1.))):raise FloatingPointError('Event-aware CNS domain failure')
                y=half;used+=ns;remaining-=ns;brain.statistics['accepted']+=1
                brain.statistics['minimum_accepted_ns']=min(brain.statistics['minimum_accepted_ns'],ns)
                brain.statistics['maximum_accepted_error']=max(brain.statistics['maximum_accepted_error'],error)
                brain.next_step_ns=min(p['maximum_step_ns'],ns*2 if error<.1 else ns)
            else:
                brain.statistics['rejected']+=1
                if ns//2<p['minimum_step_ns']:raise FloatingPointError('Event-aware CNS accuracy limit')
                brain.next_step_ns=ns//2
        brain.state=cp.asnumpy(y);brain.time_ns+=dt_ns;brain.publish_rates()


def install(brain):
    engine=EventCoupling(brain);old_spatial=spatial.GpuKcSpatialBrain.advance;old_lif=spatial.lif_events
    old_commit=kc_adaptive.commit;old_gpu=GpuProstheticOlfactoryBrain.advance
    def begin(self,dt_ns,drive,light):
        if self is not brain:return old_spatial(self,dt_ns,drive,light)
        if engine.active is not None or dt_ns>125000:raise ValueError('Nested or unsupported event block')
        engine.active=Waveform(self.state[engine.rows],self.state[self.transmission_start+engine.rows],self.tau[engine.rows],self.parameters['synaptic_tau_s'],dt_ns*1e-9)
        engine.start_elapsed=engine.batch.elapsed_ns
        try:
            result=old_spatial(self,dt_ns,drive,light)
            recorded=[]
            for t,local,jump,post in zip(engine.active.times,engine.active.rows,engine.active.jumps,engine.active.posts):
                row=int(engine.rows[int(local)])
                recorded.append({'row':row,'neuron_id':int(self.brain.node_ids[row]),
                                 'producer':'gamma_cuda' if int(local) in engine.gamma_local else 'nongamma_lif',
                                 'time_s':float(t),'jump':float(jump),
                                 'post_q':float(post) if np.isfinite(post) else None})
            engine.audit.append({'block':len(engine.audit),
                                 'start_elapsed_ns':int(engine.start_elapsed),
                                 'duration_ns':int(dt_ns),'events':recorded})
            error=float(np.max(abs(engine.active.at(dt_ns*1e-9)[0]-self.state[engine.rows])))
            engine.report['max_q_reconstruction_difference']=max(engine.report['max_q_reconstruction_difference'],error)
            if error>2e-12:raise ValueError('Analytic waveform changed physical q owner')
            engine.report['events']+=len(engine.active.times);engine.report['blocks']+=1
            return result
        finally:engine.active=None
    def recorded_lif(v,r,c,q,vinf,rate,reset,threshold,caps,tau,dt,*args):
        if engine.active is None:return old_lif(v,r,c,q,vinf,rate,reset,threshold,caps,tau,dt,*args)
        clips,t,j,post=lif_record(v,r,c,q,vinf,rate,reset,threshold,caps,tau,dt,*args)
        rr,cc=np.where(np.isfinite(t));engine.active.add(t[rr,cc],engine.other[rr].astype(np.int64),j[rr,cc],posts=post[rr,cc]);return clips
    def recorded_commit(b,v,g,ns):
        if b is not engine.batch or engine.active is None:return old_commit(b,v,g,ns)
        before=b.q.copy();old_commit(b,v,g,ns)
        delta=b.host(b.q-before*b.xp.exp(-ns*1e-9/b.tau));rows=np.flatnonzero(delta>0)
        if len(rows):engine.active.add(np.full(len(rows),(b.elapsed_ns-engine.start_elapsed)*1e-9),engine.gamma[rows],delta[rows],posts=b.host(b.q)[rows])
    def gpu(self,ns,drive,light):
        if self is not brain:return old_gpu(self,ns,drive,light)
        return engine.step(self,ns,drive,light)
    spatial.GpuKcSpatialBrain.advance=begin;spatial.lif_events=recorded_lif;kc_adaptive.commit=recorded_commit
    GpuProstheticOlfactoryBrain.advance=gpu
    def restore():
        spatial.GpuKcSpatialBrain.advance=old_spatial;spatial.lif_events=old_lif;kc_adaptive.commit=old_commit
        GpuProstheticOlfactoryBrain.advance=old_gpu
    return engine,restore
