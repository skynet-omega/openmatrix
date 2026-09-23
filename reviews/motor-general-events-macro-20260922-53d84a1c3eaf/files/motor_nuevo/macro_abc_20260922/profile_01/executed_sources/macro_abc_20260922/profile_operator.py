"""Isolated timings on actual buffers. No biological state advances here."""
import numpy as np,cupy as cp

def profile(session):
 a=session.adapter;g=a.core;b=session.brain;c=b.cuda;n=b.brain.n_neurons
 g.stream.synchronize();old_clock=g.clock.get(stream=g.stream)
 with g.stream:
  before=g.x.copy();weights=c['weights'].copy();g.clock.set(np.asarray([0.,1e-6]))
  target=cp.empty(n);rate=cp.empty(n);photo=cp.zeros(n);m=len(b.pi)
  fast=g.x[n:n+m];adaptation=g.x[n+m:n+2*m];p=b.parameters
  photo[c['pi']]=p['photoconductance_max']*fast/(p['photo_half']+p['adaptation_strength']*adaptation+fast)
  args=(np.int32(n),c['indptr'],c['indices'],c['weights'],g.x[b.transmission_start:],c['caps'],c['visual'],c['tau'],c['gain'],c['theta'],a.drive,photo,np.float64(p['conductance_per_stored_weight']),np.bool_(b.visual_output_connected),target,rate)
  def base():b.kernel(((n*32+255)//256,),(256,),args)
  base();g.stream.synchronize();g.stream.begin_capture();base();base_graph=g.stream.end_capture()
  def timed(graph,repetitions):
   graph.launch(g.stream);g.stream.synchronize();start,end=cp.cuda.Event(),cp.cuda.Event();start.record(g.stream)
   for _ in range(repetitions):graph.launch(g.stream)
   end.record(g.stream);end.synchronize();return cp.cuda.get_elapsed_time(start,end)/repetitions/1000
  full=[timed(g.graph,8) for _ in range(3)];base_times=[timed(base_graph,48) for _ in range(3)]
  preserved=bool(cp.array_equal(g.x,before)) and bool(cp.array_equal(c['weights'],weights))
  g.clock.set(old_clock)
 g.stream.synchronize()
 if not preserved:raise RuntimeError('Profiling changed physical buffers')
 return {'six_evaluation_trial_s':full,'base_CSR_call_s':base_times,'six_base_calls_fraction_of_trial':6*float(np.median(base_times))/float(np.median(full)), 'preserved_state_and_weights':preserved,'scope':'Isolated GPU graph replay on actual166700-neuron layout; base operator input preparation excluded. Full trial includes six operators, projection, updates and checks. Not whole-organism speedup and not an accuracy comparison.'}
