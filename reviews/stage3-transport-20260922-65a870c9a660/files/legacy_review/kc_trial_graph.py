"""All three KC trials and their checks in one captured GPU transaction."""
import cupy as cp
from kc_fused_warp import step

class Trial:
    def __init__(self,b,h,v,g,ge,gi,current,v_atol,g_atol):
        self.h=h;self.pool=cp.cuda.MemoryPool();stream=cp.cuda.Stream(non_blocking=True)
        h1=h//2;h2=h-h1;cubes=(h1/h)**3+(h2/h)**3;factor=cubes/(1-cubes)
        cp.cuda.get_current_stream().synchronize()
        with cp.cuda.using_allocator(self.pool.malloc),stream:
            self.inputs=[cp.array(x,copy=True) for x in (v,g,ge,gi,current)]
            vv,gg,ee,ii,cc=self.inputs
            def build():
                vf,gf,ef=step(b,vv,gg,ee,ii,h*1e-9,cc,check=False)
                va,ga,ea=step(b,vv,gg,ee,ii,h1*1e-9,cc,check=False)
                vb,gb,eb=step(b,va,ga,ee,ii,h2*1e-9,cc,check=False)
                valid=cp.isfinite(ef).all()&cp.isfinite(ea).all()&cp.isfinite(eb).all()
                norm=cp.maximum(cp.max(cp.abs(vb-vf))*factor/v_atol,cp.max(cp.abs(gb-gf))*factor/g_atol)
                error=cp.where(valid,norm,cp.inf)
                return va,ga,vb,gb,error
            temporary=build();stream.synchronize();del temporary
            stream.begin_capture()
            try:self.outputs=build()
            finally:self.graph=stream.end_capture()
        stream.synchronize()
    def run(self,v,g,ge,gi,current):
        for dst,src in zip(self.inputs,(v,g,ge,gi,current)):
            if dst.shape!=src.shape or dst.dtype!=src.dtype:raise ValueError('KC graph layout drift')
            cp.copyto(dst,src)
        self.graph.launch(stream=cp.cuda.get_current_stream())
        va,ga,vb,gb,error=self.outputs
        # Accepted physical state must never alias a later rejected trial.
        # The next replay overwrites these four graph-owned output allocations.
        return va.copy(),ga.copy(),vb.copy(),gb.copy(),float(error)

def propose(b,h,v,g,ge,gi,current,v_atol,g_atol):
    cache=getattr(b,'_motor_trial_graphs',None)
    if cache is None:cache=b._motor_trial_graphs={}
    if h not in cache:
        if len(cache)>=32:raise RuntimeError('KC graph count budget exhausted')
        cache[h]=Trial(b,h,v,g,ge,gi,current,v_atol,g_atol)
        if sum(t.pool.total_bytes() for t in cache.values())>1024**3:raise RuntimeError('KC private graph budget exceeds1GiB')
    return cache[h].run(v,g,ge,gi,current)
