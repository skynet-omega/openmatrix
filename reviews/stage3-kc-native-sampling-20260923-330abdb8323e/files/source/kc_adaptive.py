"""Step-doubled midpoint with explicit local error and causal event commits."""
import types
import numpy as np

V_ATOL=2e-5
G_ATOL=2e-7

def numpy_step(b,v,g,ge,gi,dt,current):
    xp=b.xp;conductance=(ge+gi)@b.shuntG;drive=(-b.rest*ge+(-68.-b.rest)*gi)@b.shuntb+current
    def electrical(gates):
        m,h,p,n=(gates[:,:,j] for j in range(4));f=xp.stack((m*m*m*h,p,n**4),axis=-1).reshape(b.n,-1)
        return (f@b.chanG+conductance).reshape(b.n,b.ports,b.ports)+b.G,(f*b.ena)@b.chanb+drive
    s,t=b.rates(v+b.rest);gh=s+(g-s)*xp.exp(-.5*dt/t);K,bb=electrical(gh);C2=b.C*(2/dt)
    pred=xp.linalg.solve(K+C2,(v@C2.T+bb)[:,:,None])[:,:,0]
    sm,tm=b.rates(pred+b.rest);gm=sm+(g-sm)*xp.exp(-.5*dt/tm);K,bb=electrical(gm)
    rhs=v@C2.T-xp.einsum('nij,nj->ni',K,v)+2*bb
    return xp.linalg.solve(K+C2,rhs[:,:,None])[:,:,0],sm+(g-sm)*xp.exp(-dt/tm)

def commit(b,v,g,ns):
    xp=b.xp;b.delta=v;b.gates=g;dt=ns*1e-9
    siz=b.observe()[:,1];slope=siz-b.last_siz;peak=(b.previous_slope>0)&(slope<=0)
    event=peak&(b.last_siz>-40.)&(b.last_siz-b.trough>=20.)
    q=b.q*xp.exp(-dt/b.tau)+event/(b.caps*b.tau)
    b.clipped+=(q>1).astype(xp.int64);b.q=xp.minimum(q,1.);b.counts+=event.astype(xp.int64)
    b.trough=xp.where(peak,siz,xp.minimum(b.trough,siz));b.previous_slope=slope;b.last_siz=siz;b.elapsed_ns+=ns
    callback=getattr(b,'_motor_axonal_callback',None)
    if callback:callback(ns)

def advance(b,dt_ns,ge_nS,gi_nS,*,current_pA=None,inner_step_ns=25000):
    xp=b.xp
    if type(dt_ns) is not int or dt_ns<=0 or type(inner_step_ns) is not int or not 0<inner_step_ns<=25000:raise ValueError('Invalid adaptive KC clock')
    for x in (ge_nS,gi_nS):
        if x.shape!=(b.n,4) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('Invalid conductance')
    ge,gi=xp.asarray(ge_nS),xp.asarray(gi_nS);current=xp.zeros_like(b.delta) if current_pA is None else xp.asarray(current_pA)
    if current.shape!=b.delta.shape or not bool(xp.isfinite(current).all()):raise ValueError('Invalid current')
    if b.backend=='cuda':
        from kc_fused_warp import step
    else:step=numpy_step
    left=dt_ns;h=min(inner_step_ns,left);attempts=0
    stats=getattr(b,'_motor_adaptive_stats',None)
    if stats is None:stats=b._motor_adaptive_stats={'accepted':0,'rejected':0,'min_step_ns':25000,'max_estimated_error':0.}
    while left:
        h=min(h,left)
        # Merge the last integer-nanosecond tail; e.g.4*3906 leaves1ns.
        # The complete merged trial is error-tested before it can commit.
        if 0<left-h<200:h=left
        h1=h//2;h2=h-h1;attempts+=1
        if h1<1 or attempts>10000:raise RuntimeError('Adaptive KC numerical work limit')
        v,g=b.delta,b.gates
        if b.backend=='cuda':
            from kc_trial_graph import propose
            va,ga,vb,gb,e=propose(b,h,v,g,ge,gi,current,V_ATOL,G_ATOL)
        else:
            vf,gf=step(b,v,g,ge,gi,h*1e-9,current)
            va,ga=step(b,v,g,ge,gi,h1*1e-9,current)
            vb,gb=step(b,va,ga,ge,gi,h2*1e-9,current)
            cubes=(h1/h)**3+(h2/h)**3;factor=cubes/(1-cubes)
            e=max(float(xp.max(xp.abs(vb-vf)))*factor/V_ATOL,float(xp.max(xp.abs(gb-gf)))*factor/G_ATOL)
        if not np.isfinite(e):raise FloatingPointError('Nonfinite KC step estimator')
        if e<=1:
            if b.backend!='cuda':
                for vv,gg in ((va,ga),(vb,gb)):
                    if not bool(xp.isfinite(vv).all()) or not bool(xp.isfinite(gg).all()) or bool(xp.any((gg<0)|(gg>1))):raise FloatingPointError('Invalid adaptive KC state')
            commit(b,va,ga,h1);commit(b,vb,gb,h2);left-=h
            stats['accepted']+=1;stats['min_step_ns']=min(stats['min_step_ns'],h1,h2);stats['max_estimated_error']=max(stats['max_estimated_error'],e)
            h=min(inner_step_ns,h*2 if e<.1 else h)
        else:
            stats['rejected']+=1
            if h//2<200:raise FloatingPointError('KC temporal accuracy requires less than200ns')
            h//=2
    return b.host(b.q)

def install(brain):
    wrapper=brain._spatial_batch;batch=wrapper
    while hasattr(batch,'base'):batch=batch.base
    old=batch.advance;old_wrapper=wrapper.advance
    batch.advance=types.MethodType(advance,batch)
    def sample(w,dt_ns,ge,gi,*,inner_step_ns=25000,**kwargs):
        if hasattr(batch,'_motor_axonal_callback'):raise RuntimeError('Nested adaptive axonal callback')
        from axon_gpu import Publisher
        publisher=Publisher(w,batch);batch._motor_axonal_callback=publisher
        try:
            result=batch.advance(dt_ns,ge,gi,inner_step_ns=inner_step_ns,**kwargs)
            publisher.finish(dt_ns);return result
        finally:del batch._motor_axonal_callback
    if wrapper is not batch:wrapper.advance=types.MethodType(sample,wrapper)
    def restore():
        batch.advance=old;wrapper.advance=old_wrapper
    return restore
