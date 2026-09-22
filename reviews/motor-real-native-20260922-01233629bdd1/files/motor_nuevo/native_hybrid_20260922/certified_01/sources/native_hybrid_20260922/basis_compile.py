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

def residual_certificate(G,b,cG,cb,groups,ena):
 """Bound grouping error for factors in [0,1], including arithmetic margin.

 The 256-epsilon margin exceeds the maximum 51-term accumulation and three
 coefficient products of this adapter. It does not authorize basis truncation
 above compile_basis's independent proportionality bound.
 """
 rg=np.zeros_like(G);rb=np.zeros_like(b)
 for k,members in enumerate(groups):
  for i,factor in members:rg[i]=factor*cG[k];rb[i]=factor*cb[k]
 margin=256*np.finfo(float).eps
 dg=np.sum(abs(G-rg)+margin*(abs(G)+abs(rg)),axis=(0,2))
 db=np.sum((abs(b-rb)+margin*(abs(b)+abs(rb)))*abs(ena[:,None]),axis=0)
 return dg,db

def compile_warp(code,groups,unroll=True,certificate=None):
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
 if certificate is not None:
  dg,db=certificate
  constants=''.join('__device__ __constant__ double '+name+'[17]={'+','.join(repr(float(x)) for x in a)+'};\n' for name,a in [('DGB',dg),('DBB',db)])
  code=constants+code
  old='double rmax=fabs(residual),amax=row,bmax=fabs(br),xmax=fabs(x);'
  new='''double vmax=fabs(v0);
  for(int o=16;o>0;o/=2){double z=__shfl_down_sync(mask,vmax,o);if(i+o<17)vmax=fmax(vmax,z);}
  vmax=__shfl_sync(mask,vmax,0);
  double rmax=fabs(residual),amax=row,bmax=fabs(br),xmax=fabs(x);'''
  if old not in code:raise ValueError('Residual source pattern changed')
  code=code.replace(old,new)
  old='double err=__shfl_sync(mask,rmax/fmax(amax*xmax+bmax,1e-300),0);'
  gm=float(np.max(dg));bm=float(np.max(db))
  new=f'''double rhs_bound=(stage==1?2.:1.)*{bm!r}+(stage==1?{gm!r}*vmax:0.);
  double numerator=rmax+rhs_bound+{gm!r}*xmax;
  double denominator=fmax(amax-{gm!r},0.)*xmax+fmax(bmax-rhs_bound,0.);
  double err=__shfl_sync(mask,numerator/fmax(denominator,1e-300),0);'''
  if old not in code:raise ValueError('Residual acceptance source pattern changed')
  code=code.replace(old,new)
 if unroll:
  # Fixed matrix dimensions are part of this adapter; scalarized row arrays
  # eliminate dynamically indexed local-memory accesses in the dense solve.
  for loop in ('for(int j=0;j<17;j++)','for(int k=j+1;k<17;k++)','for(int j=16;j>=0;j--)'):
   code=code.replace(loop,'\n#pragma unroll\n'+loop)
 return code
