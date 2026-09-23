"""Second numerical prototype: uncompressed RA34PW2 with physical event commits."""
import numpy as np,cupy as cp
from native_cell import Cell,HERE,WARP
from ros_tableau import header

def kernel():return cp.RawKernel(WARP.split('extern "C"')[0]+header()+(HERE/'ros_step.cu').read_text(),'ros_trial',options=('--fmad=false',))

class RosCell(Cell):
 def trial(self):
  if not hasattr(self,'ros_kernel'):self.ros_kernel=kernel();self.report['method']='RA34PW2_block_diagonal_W_v1'
  b=self.b;v,g=self.state['delta'],self.state['gates']
  self.va=cp.empty_like(v);self.vb=cp.empty_like(v);self.ga=cp.empty_like(g);self.gb=cp.empty_like(g)
  kv=cp.empty((b.n,4,17));kg=cp.empty((b.n,4,17,4));err=cp.zeros(b.n)
  self.ros_kernel((b.n,),(32,),(np.int32(b.n),self.clock,np.float64(b.rest),v,g,self.ge,self.gi,self.current,b.C,b.G,b.chanG,b.chanb,b.shuntG,b.shuntb,b.ena,kv,kg,self.va,self.ga,self.vb,self.gb,err))
  self.error=cp.max(err)
