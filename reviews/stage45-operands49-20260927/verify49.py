"""Recompute the complete captured diagnostic from a clean extraction."""
from pathlib import Path
import json,hashlib,time
from analyze49 import compute,render
H=Path(__file__).resolve().parent
start=time.process_time()
if (H/'MANIFEST.json').exists():
 m=json.loads((H/'MANIFEST.json').read_text())
 for name,e in m['files'].items():
  p=H/name
  if hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']:raise ValueError('Changed file '+name)
v=compute();saved=json.loads((H/'RESULTADOS.json').read_text())
a=dict(v);b=dict(saved);a.pop('CPU_s');b.pop('CPU_s')
if a!=b:raise ValueError('Recomputed diagnostic differs')
if render(v)!=(H/'RESULTADOS.md').read_text():raise ValueError('Report differs')
print(json.dumps(dict(schema='verify49_complete_captured_diagnostic_v1',CPU_s=time.process_time()-start,exact_sums=v['total_exact_sums'],stage4=False,stage5=False,scope='All captured windows; not a replay of full CNS/physical simulation')))
