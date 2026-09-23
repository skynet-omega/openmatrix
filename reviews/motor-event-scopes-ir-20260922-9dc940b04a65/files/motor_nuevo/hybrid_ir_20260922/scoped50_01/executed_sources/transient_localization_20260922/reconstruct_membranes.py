"""Recompute the saved held-input comparison without another GPU execution."""
from pathlib import Path
import json,numpy as np
T=Path(__file__).resolve().parent
with np.load(T/'replay_causal_1562.npz',allow_pickle=False) as z:fine={k:z[k].copy() for k in z.files}
r=json.loads((T/'MEMBRANE_REPLAY.json').read_text());results={}
for name in r['runs']:
 with np.load(T/f'replay_{name}.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 for v in a.values():
  if not np.isfinite(v).all():raise ValueError('Nonfinite fixture')
 events,ef=a['events'],fine['events'];same=events.shape==ef.shape and np.array_equal(events[:,1],ef[:,1])
 results[name]={'event_rows_equal_to_refined':bool(same),'max_mark_difference_s':float(np.max(abs(events[:,0]-ef[:,0]),initial=0)) if same else None,'voltage_max_difference':float(np.max(abs(a['delta']-fine['delta']))),'q_max_difference':float(np.max(abs(a['q']-fine['q']))),'counts_equal':np.array_equal(a['counts'],fine['counts'])}
if results!=r['comparison_to_predeclared1562ns_refinement']:raise ValueError('Report disagrees with actual arrays')
print('PASS: all five membrane comparisons reconstructed from saved arrays')
