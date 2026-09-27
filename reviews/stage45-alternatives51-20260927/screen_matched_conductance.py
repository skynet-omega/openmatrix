"""Adviser normalized conductance proposal with corrected baseline shunt control."""
from pathlib import Path
import json,time
import numpy as np
H=Path(__file__).resolve().parent
cpu=time.process_time();a=np.load(H/'local_projection.npz');r=a['sham_record'];q0=r[:,0];E=r[:,2];I=-r[:,3];drive=r[:,4];theta=r[:,5];tau=r[:,9];X0=E+np.maximum(drive,0);Y0=I+theta+np.maximum(-drive,0);S=X0+Y0;gE0=X0/S;gI0=Y0/S;EL=2*q0-gE0;C=2*tau
if not np.all(S>0):raise ValueError('Invalid scale')
# Control uses total basal conductance=2; adviser original control omitted one unit of shunt.
def rhs(v,gE,gI,conductance):
 if conductance:return (EL+gE-(1+gE+gI)*v)/C
 return (EL+gE*(1-q0)-gI*q0-(gE0+gI0)*(v-q0)-v)/C
initialG=rhs(q0,gE0,gI0,True);initialC=rhs(q0,gE0,gI0,False)
# Prespecified numerical algebra tolerance, not scientific outcome cutoff.
if max(np.max(abs(initialG)),np.max(abs(initialC)))>1e-12:raise ValueError('Basal identity')
traces=[]
for label,dx in [('constant',np.zeros(6)),('observed_PN_transplant',a['sham_delta_input']),('balanced_EI_10percent',.1*S)]:
 e=X0+dx;i=Y0+ (dx if label.startswith('balanced') else 0);ge=e/S;gi=i/S
 G=q0.copy();K=q0.copy();dt=.0001;hist=[]
 for j in range(1000):
  G+=dt*rhs(G,ge,gi,True);K+=dt*rhs(K,ge,gi,False);hist.append([G.copy(),K.copy()])
 z=np.array(hist);outside=np.maximum(np.maximum(-z,z-1),0)
 traces.append({'probe':label,'G_final':G.tolist(),'current_final':K.tolist(),'max_difference':float(np.max(abs(z[:,0]-z[:,1]))),'max_outside_0_1':float(outside.max()),'requires_material_clipping':bool(outside.max()>1e-6)})
res={'scope':'Six recorded generic DN local held-input tests, not recurrent model. Normalized engineering voltage output V=q; no mV claim.','proposal':'ASTRA51 initial-S conductance, own baseline preserved. Correct current control freezes total basal shunt=2, so both initial tau equal inherited tau.','EL':EL.tolist(),'EL_outside_reversal_interval_count':int(((EL<0)|(EL>1)).sum()),'baseline_derivative_max':float(max(np.max(abs(initialG)),np.max(abs(initialC)))),'probes':traces,'classification':'LOCAL_ALGEBRA_QUALIFIED_ONLY; EL outside [0,1] is compensating current, not passive physiological leak','CPU_s':time.process_time()-cpu}
(H/'MATCHED_CONDUCTANCE.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n');print(json.dumps(res))
