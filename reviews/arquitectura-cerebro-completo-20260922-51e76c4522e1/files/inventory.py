"""Empirical topology counterexamples to treating PN's current solver as generic."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.sparse import diags,eye,csr_matrix,kron
H=Path(__file__).resolve().parent;P=H.parent/'pn_abc_20260922';sys.path[:0]=[str(P),str(P/'vendor')]
from cyclic_plan import CyclicPlan
from cyclic_tree import CyclicTree
rows=[]
for name,A in [('path128',diags([-np.ones(127),2*np.ones(128),-np.ones(127)],[-1,0,1])),('grid16x16',kron(eye(16),diags([-np.ones(15),2*np.ones(16),-np.ones(15)],[-1,0,1]))+kron(diags([-np.ones(15),2*np.ones(16),-np.ones(15)],[-1,0,1]),eye(16))),('clique96',csr_matrix(97*np.eye(96)-np.ones((96,96))))]:
 G=(A+eye(A.shape[0])).tocsr();M=eye(A.shape[0],format='csr');plan=CyclicPlan(G,M,[]);row={'topology':name,'coordinates':G.shape[0],'G_nnz':G.nnz,'retained_core':len(plan.core),'contraction_rounds':len(plan.rounds)}
 try:gpu=CyclicTree(plan,1.);row['current_GPU_constructor']='accepts'
 except ValueError as exc:row['current_GPU_constructor']='rejects';row['reason']=str(exc)
 rows.append(row)
result={'scope':'Solver topology compatibility counterexample, not physiology or speed benchmark','rows':rows,'fixed_kernel_limitations':['NaT/NaP/K equations and3conductances hardcoded','Ca kernel atmost4gates','Small dense electrical core atmost64','PN specialized outer SDIRK and release chemistry adapter'],'conclusion':'Current PN prototypes are reusable experiments, not a generic whole-brain engine.'}
(H/'INVENTORY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
