"""GPU-only hot queries over immutable private epoch buffers; no CUDA graphs."""
from pathlib import Path
import numpy as np
from event_sparse import EpochGuard, need


class GpuPortOperator:
    def __init__(self, cp, ptr, indices, weights, caps, visual, ports, part,
                 query_times, scale, connected):
        self.cp=cp
        self.pool=cp.cuda.MemoryPool()
        self.stream=cp.cuda.Stream(non_blocking=True)
        self.guard=EpochGuard(part.weight_version,ports.version)
        self.ports=ports
        self.times=np.array(query_times,copy=True)
        self.n,self.p,self.m=len(ptr)-1,len(ports.rows),len(query_times)
        self.active_count=len(part.active_rows)
        self.scale,self.connected=np.float64(scale),np.bool_(connected)
        module=cp.RawModule(code=(Path(__file__).parent/'sparse_kernels.cu').read_text(),
                            options=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true'),
                            name_expressions=['project_ports','csr_current'])
        module.compile()
        self.project_kernel=module.get_function('project_ports')
        self.current_kernel=module.get_function('csr_current')
        with cp.cuda.using_allocator(self.pool.malloc), self.stream:
            self.full=tuple(cp.asarray(x) for x in (ptr,indices,weights))
            self.reduced=tuple(cp.asarray(x) for x in (part.port_ptr,part.port_sources,part.port_weights))
            self.caps,self.visual,self.rows=cp.asarray(caps),cp.asarray(visual),cp.asarray(part.active_rows)
            self.port_arrays=tuple(cp.asarray(x) for x in (query_times,ports.rows,ports.q,ports.s,ports.tau,
                                                           ports.ptr,ports.times,ports.jumps,ports.sets,ports.posts))
            self.release=cp.zeros(self.n,dtype=cp.float64)
            self.q_trace=cp.empty((self.m,self.p),dtype=cp.float64)
            self.s_trace=cp.empty_like(self.q_trace)
            self.full_out=cp.zeros((self.m,self.n,2),dtype=cp.float64)
            self.reduced_out=cp.zeros_like(self.full_out)
        self.stream.synchronize()

    def project(self, k, weight_version, event_version):
        need(isinstance(k,int) and 0<=k<self.m,'Invalid query index')
        self.guard.query(float(self.times[k]),weight_version,event_version)
        times,rows,q,s,tau,eptr,et,ej,es,posts=self.port_arrays
        self.project_kernel(((self.p+255)//256,),(256,),
                            (np.int32(self.p),np.int32(k),times,rows,q,s,tau,np.float64(self.ports.ts),
                             eptr,et,ej,es,posts,self.release,self.q_trace,self.s_trace),stream=self.stream)

    def current(self,k,reduced):
        data=self.reduced if reduced else self.full
        count=self.active_count if reduced else self.n
        out=self.reduced_out if reduced else self.full_out
        self.current_kernel(((count*32+255)//256,),(256,),
                            (np.int32(count),np.int32(self.n),np.int32(k),self.rows,np.bool_(reduced),
                             *data,self.release,self.caps,self.visual,self.scale,self.connected,out),stream=self.stream)

    def close(self):
        self.stream.synchronize()
