"""Independent-set graph contraction; deterministic parallel message accumulation."""
import numpy as np
from scipy.sparse import csr_matrix
from pn_graph_elimination_backend import EliminationPlan

class CyclicPlan(EliminationPlan):
 def __init__(self,G,M,protected):
  self.G=csr_matrix(G);self.M=csr_matrix(M);n=G.shape[0];pattern=self.G.copy();pattern.data.fill(1);p=self.M.copy();p.data.fill(1);pattern=(pattern+p).tocsr();pattern.setdiag(0);pattern.eliminate_zeros();coo=pattern.tocoo()
  pairs=sorted({(min(int(i),int(j)),max(int(i),int(j))) for i,j in zip(coo.row,coo.col)});adj=[set() for _ in range(n)];edge={p:k for k,p in enumerate(pairs)}
  for i,j in pairs:adj[i].add(j);adj[j].add(i)
  blocked=set(map(int,protected));removed=np.zeros(n,dtype=bool);steps=[];self.rounds=[]
  candidates={i for i in range(n) if i not in blocked and len(adj[i])<=2}
  while candidates:
   selected=[];forbidden=set()
   for i in sorted(candidates):
    if i not in forbidden:selected.append(i);forbidden.add(i);forbidden.update(adj[i])
   if not selected:raise RuntimeError('No contraction progress')
   start=len(steps);changed=set()
   for i in selected:
    neighbors=sorted(adj[i]);js=neighbors+[-1]*(2-len(neighbors));es=[edge[tuple(sorted((i,j)))] for j in neighbors]+[-1]*(2-len(neighbors));cross=-1
    for j in neighbors:adj[j].remove(i);changed.add(j)
    if len(neighbors)==2:
     a,b=neighbors
     if b not in adj[a]:edge[a,b]=len(pairs);pairs.append((a,b));adj[a].add(b);adj[b].add(a)
     cross=edge[a,b]
    steps.append([i,*js,*es,cross]);adj[i].clear();removed[i]=True
   self.rounds.append((start,len(steps)-start));candidates.difference_update(selected)
   candidates.update(i for i in changed if not removed[i] and i not in blocked and len(adj[i])<=2)
  self.steps=np.asarray(steps,dtype=np.int64).reshape(-1,6);self.core=np.flatnonzero(~removed);self.pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2)
  if len(self.core)>256:raise ValueError('Cyclic core bound exceeded')
  i,j=self.pairs.T;self.base=np.stack([np.asarray(self.G[i,j]).ravel(),np.asarray(self.G[j,i]).ravel()],axis=1);self.mass=np.stack([np.asarray(self.M[i,j]).ravel(),np.asarray(self.M[j,i]).ravel()],axis=1)
  self.dg=self.G.diagonal();self.dm=self.M.diagonal();ci={int(row):i for i,row in enumerate(self.core)}
  self.core_edges=np.asarray([[edge[i,j],ci[i],ci[j]] for i in self.core for j in sorted(adj[i]) if i<j],dtype=np.int64).reshape(-1,3)
  self.gathers=[]
  for start,count in self.rounds:
   buckets={}
   def add(target,slot):buckets.setdefault(target,[]).append(slot)
   for s in range(start,start+count):
    i,j,k,e,f,c=self.steps[s]
    if j>=0:add(j,6*s);add(n+j,6*s+1)
    if k>=0:add(k,6*s+2);add(n+k,6*s+3);add(2*n+2*c,6*s+4);add(2*n+2*c+1,6*s+5)
   targets=sorted(buckets);offsets=np.r_[0,np.cumsum([len(buckets[t]) for t in targets])];sources=[s for t in targets for s in buckets[t]]
   self.gathers.append((np.asarray(targets,dtype=np.int64),offsets.astype(np.int64),np.asarray(sources,dtype=np.int64)))
