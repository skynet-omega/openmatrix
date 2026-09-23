"""Read-only structural discriminator; row overlap alone does NOT prove dead code."""
import numpy as np

def inventory(b, events):
 def host(x):return x.get() if hasattr(x,'get') else np.asarray(x)
 ptr=host(b.cuda['indptr']);n=len(ptr)-1;degree=np.diff(ptr)
 masks={}
 for name in ('_output_cache','_pnkc_cache','_regional_cache','_orn_terminal_cuda','_orn_pn_cuda','_retinal_cuda','_gaba_cuda','_boundary_cuda'):
  d=getattr(b,name,{})
  if 'rows' in d:masks[name]=host(d['rows']).astype(np.int64)
 for name in ('_dynamic_gpu_rows','_pvlp_cuda_rows','_cvn7_rows','_retinal_port_cuda_rows','_mi9_rows'):
  if hasattr(b,name):masks[name]=host(getattr(b,name)).astype(np.int64)
 masks['physical_ports']=host(events.rows).astype(np.int64)
 result={};union=np.zeros(n,dtype=bool)
 for name,rows in masks.items():
  rows=np.unique(rows[(rows>=0)&(rows<n)]);union[rows]=True
  result[name]={'rows':len(rows),'base_edges_in_rows':int(degree[rows].sum())}
 visual=host(b.cuda['visual']).astype(bool)
 return {'base_neurons':n,'base_edges':int(ptr[-1]),'state_variables':len(b.state),'row_sets':result,'union_rows':int(union.sum()),'union_base_edges':int(degree[union].sum()),'nonvisual_union_edges':int(degree[union&~visual].sum()),'scope':'Overlapping row inventories, not safe-removal proof. Intermediate reads and rate dependencies must be checked. No runtime gain inferred.'}
