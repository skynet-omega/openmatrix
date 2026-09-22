"""One real smoke trajectory with decision-cause tracing; no changed solver."""
from pathlib import Path
import sys,json,ctypes as ct
import numpy as np
H=Path(__file__).resolve().parent
import run_pipeline
from motor_runtime import load
import runtime_session,organism_adapter
from graph_core import NativeGraph

class TracedGraph(NativeGraph):
 def __init__(self,*args,**kw):
  kw['native_library']=H/'libgraph_control_trace.so';super().__init__(*args,**kw)
 def trace(self):
  self.lib.engine_trace_count.argtypes=[ct.c_void_p];self.lib.engine_trace_count.restype=ct.c_long
  self.lib.engine_trace_copy.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double),ct.c_long]
  n=self.lib.engine_trace_count(self.handle);a=np.zeros((n,10))
  if self.lib.engine_trace_copy(self.handle,a.ctypes.data_as(ct.POINTER(ct.c_double)),a.size)!=0:raise RuntimeError('Trace copy failed')
  return a

records=[];chosen={};original=organism_adapter.OrganismAdapter.step
def traced(self,b,ns,drive,light):
 waveform=self.events.active
 inputs={'times_s':list(waveform.times),'source_waveform_rows':list(waveform.rows),'jumps':list(waveform.jumps)}
 result=original(self,b,ns,drive,light)
 trace=self.core.trace();phase='accepted_exchange' if ns==125000 else 'discarded_predictor' if ns==62500 else 'UNKNOWN'
 records.append({'phase':phase,'epoch_ns':ns,'clock_end_ns':b.time_ns,**inputs,'event_neuron_rows':np.asarray(self.events.rows)[np.asarray(waveform.rows,dtype=int)].tolist(),'trace':trace.tolist()})
 if not chosen and phase=='accepted_exchange' and waveform.times:
  import cupy as cp
  from scipy.sparse import csr_matrix
  from scipy.sparse.csgraph import dijkstra
  idx=b.cuda['indices'].get();ptr=b.cuda['indptr'].get();weights=b.cuda['weights'].get();n=b.brain.n_neurons
  sources=np.unique(np.asarray(self.events.rows)[np.asarray(waveform.rows,dtype=int)])
  active=weights!=0
  positions=np.flatnonzero(active & np.isin(idx,sources));direct=np.unique(np.searchsorted(ptr,positions,side='right')-1)
  graph=csr_matrix((active.astype(np.float64),idx,ptr),shape=(n,n));graph.eliminate_zeros();outgoing=graph.transpose().tocsr()
  distance=dijkstra(outgoing,directed=True,indices=sources,min_only=True,unweighted=True);reachable=np.isfinite(distance)
  chosen.update(record_index=len(records)-1,source_neuron_rows=sources.tolist(),source_node_ids=b.brain.node_ids[sources].tolist(),direct_CSR_receivers=int(len(direct)),CSR_reachable_neurons=int(reachable.sum()),neurons=n,nonzero_CSR_edges=int(graph.nnz),complete_operator_dependency_map=False,scope='Effective nonzero CSR only; specialized replaced routes, masks, filters and cycles beyond CSR not mapped. Reachability is descriptive; not measured need for recomputation and not an exact causal partition proof.')
 return result

organism_adapter.NativeGraph=TracedGraph
organism_adapter.OrganismAdapter.step=traced
try:code=run_pipeline.main(['--out',str(H/'trace_smoke_01'),'--odor','sham','--engine','causal_cuda','--ms','1','--smoke-test'])
finally:
 organism_adapter.NativeGraph=NativeGraph;organism_adapter.OrganismAdapter.step=original
 (H/'CNS_TRACES.json').write_text(json.dumps({'epochs':records,'chosen':chosen},indent=2)+'\n')
if code:raise SystemExit(code)
summary={}
for phase in ('discarded_predictor','accepted_exchange'):
 blocks=[r for r in records if r['phase']==phase];a=np.asarray([row for r in blocks for row in r['trace']]);bits=a[:,7].astype(int)
 summary[phase]={'epochs':len(blocks),'trials':len(a),'accepted':int(a[:,9].sum()),'rejected':int(len(a)-a[:,9].sum()),'reason_event':int(((bits&1)!=0).sum()),'reason_end':int(((bits&2)!=0).sum()),'reason_proposed_step':int(((bits&4)!=0).sum()),'reason_maximum':int(((bits&8)!=0).sum()),'ties_count_as_multiple_reasons':True,'max_error':float(a[:,8].max()),'proposal_limited_with_error_below_point1':int((((bits&4)!=0)&(a[:,8]<.1)).sum())}
(H/'CNS_CAUSES.json').write_text(json.dumps({'summary':summary,'chosen':chosen},indent=2)+'\n');print(json.dumps(summary))
