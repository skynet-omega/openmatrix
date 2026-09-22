"""Proportional operator-basis compiler, independent of anatomy/channel names.

Groups only [matrix, forcing vector] rows proportional within 32 FP64 eps.
All channel factors and reversal potentials remain independent coefficients.
The original arrays remain authoritative for full operator residual checks.
"""
import numpy as np

def compile_basis(G,b):
 G=np.asarray(G);b=np.asarray(b)
 if G.dtype!=np.float64 or b.dtype!=np.float64 or G.ndim!=3 or G.shape[1]!=G.shape[2] or b.shape!=G.shape[:2]:raise ValueError('FP64 square operator basis required')
 if not np.isfinite(G).all() or not np.isfinite(b).all():raise ValueError('Nonfinite basis')
 a=np.concatenate((G.reshape(len(G),-1),b),axis=1);groups=[];bases=[];residual=0.
 for i,row in enumerate(a):
  scale=np.max(abs(row))
  if scale==0:continue
  found=False
  for k,base in enumerate(bases):
   pivot=np.argmax(abs(base));factor=row[pivot]/base[pivot]
   err=float(np.max(abs(row-factor*base))/scale)
   if err<=32*np.finfo(float).eps:
    groups[k].append((i,float(factor)));residual=max(residual,err);found=True;break
  if not found:bases.append(row.copy());groups.append([(i,1.)])
 ng=G.shape[1]**2
 out=np.stack(bases) if bases else np.empty((0,a.shape[1]))
 return out[:,:ng].reshape(-1,G.shape[1],G.shape[2]),out[:,ng:].copy(),groups,{'original_terms':len(G),'compiled_terms':len(groups),'max_relative_basis_residual':residual,'admission_relative_bound':32*np.finfo(float).eps}

def compile_warp(code,groups,unroll=True):
 # Exact source pattern required: compilation must fail closed on donor drift.
 start=code.index('  for(int port=0;port<17;port++){');end=code.index('  if(stage==1)rhs*=2.;',start)
 chunks=[]
 for group,members in enumerate(groups):
  gterms=[];bterms=[]
  for index,factor in members:
   port,channel=divmod(index,3);term=f'({factor!r}*__shfl_sync(mask,f{channel},{port}))'
   gterms.append(term);bterms.append(f'({term}*ena[{index}])')
  chunks.append('  { double cg='+ '+'.join(gterms)+'; double cb='+ '+'.join(bterms)+f'; rhs+=cb*chanb[{group}*17+i];\n   for(int j=0;j<17;j++)A[j]+=cg*chanG[{group}*289+i*17+j]; }}\n')
 code=code[:start]+''.join(chunks)+code[end:]
 if unroll:
  # Fixed matrix dimensions are part of this adapter; scalarized row arrays
  # eliminate dynamically indexed local-memory accesses in the dense solve.
  for loop in ('for(int j=0;j<17;j++)','for(int k=j+1;k<17;k++)','for(int j=16;j>=0;j--)'):
   code=code.replace(loop,'\n#pragma unroll\n'+loop)
 return code
