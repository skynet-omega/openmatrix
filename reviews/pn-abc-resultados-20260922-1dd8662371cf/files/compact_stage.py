"""Stage-local Schur Newton. All full-mass modes are reconstructed and checked."""
import numpy as np
from scipy.sparse import diags
from condensation import Condensation
from pn_coupled_ionic import stage_channels
from pn_fine_ionic import channel_conductance

def compact_stage(self,vbase,xbase,guess,synapse,ca_base,ca_update,threshold_guess,
                  *,shift,q,current,ca,rtol,atol,max_newton,gate_atol,progress):
    nodes=self.active_nodes;cn=np.array([],dtype=int) if ca is None else ca.nodes
    protected=np.union1d(nodes,cn)
    if synapse is not None:protected=np.union1d(protected,synapse[0])
    extra=self.backend.M-diags(self.backend.C);protected=np.union1d(protected,np.unique(extra.nonzero()[0]))
    key=(id(self.backend),shift,protected.tobytes())
    if getattr(self,'_schur_key',None)!=key:
        self._schur=Condensation(self.backend,protected,shift);self._schur_key=key
    a=self._schur;core=a.core;ap=np.searchsorted(core,nodes);cap=np.searchsorted(core,cn)
    sn,sg,se=(None,None,None) if synapse is None else synapse[:3]
    sp=None if sn is None else np.searchsorted(core,sn)
    def nonlinear(vc):
        x,ionic,jac,gate_error=stage_channels(vc[ap]+self.leak_reversal_mV,xbase,q,self.gbar_nS,self.reversal_mV)
        out=np.zeros(len(core));out[ap]+=ionic.sum(axis=1)
        if sn is not None:out[sp]+=sg*(vc[sp]+self.leak_reversal_mV-se)
        cdata=None
        if ca is not None:
            cdata=ca.stage(vc[cap]+self.leak_reversal_mV,ca_base,q)
            out[cap]+=cdata['current'];gate_error=max(gate_error,cdata['gate_error'])
        return out,x,ionic,jac,gate_error,cdata
    def full_residual(v,data):
        r=self.backend.mass_action((v-vbase)*shift)+self.backend.action(v)-current
        r[core]+=data[0]
        return r
    history=[];total=0
    try:
        ref=guess if threshold_guess is None else threshold_guess
        data=nonlinear(ref[core]);threshold=max(atol,rtol*float(np.linalg.norm(full_residual(ref,data))))
        # Delta coordinates preserve M*v' and avoid cancellation in shift*M*vbase.
        rhs=current-self.backend.action(vbase);reduced=a.reduce(rhs);b=reduced[core]
        delta=guess[core]-vbase[core]
        def evaluate(d):
            data=nonlinear(vbase[core]+d);r=a.K@d+data[0]-b
            return r,float(np.linalg.norm(r)),data
        r,norm,data=evaluate(delta)
        for iteration in range(max_newton+1):
            row={'iteration':iteration,'reduced_residual_l2_pA':norm,'gate_residual':data[4]};history.append(row)
            if progress:progress({'stage_iteration':iteration,'reduced_residual_l2_pA':norm})
            if np.isfinite(norm) and norm<=threshold and data[4]<=gate_atol:
                v=vbase+a.reconstruct(reduced,delta)
                # Reduced convergence is only a trigger to check the unchanged equation.
                # Up to three full residual corrections, like the parent's refinement.
                for refine in range(4):
                    data=nonlinear(v[core]);full=full_residual(v,data);fullnorm=float(np.linalg.norm(full))
                    row['residual_l2_pA']=fullnorm;row['full_refinements']=refine
                    if np.isfinite(fullnorm) and fullnorm<=threshold and data[4]<=gate_atol:
                        return (v,data[1],data[2],data[5]),dict(accepted=True,history=history,threshold_pA=threshold,total_iterations=total)
                    if refine==3 or not np.isfinite(fullnorm):break
                    d=np.zeros(len(v));d[nodes]+=data[3]
                    if sn is not None:d[sn]+=sg
                    if ca is not None:d[cn]+=data[5]['jacobian']
                    v+=a.solve(-full,d);total+=1
                return None,dict(accepted=False,reason='original_full_residual_failed',history=history,threshold_pA=threshold,total_iterations=total)
            if not np.isfinite(norm) or iteration==max_newton:break
            diagonal=np.zeros(len(core));diagonal[ap]+=data[3]
            if sn is not None:diagonal[sp]+=sg
            if ca is not None:diagonal[cap]+=data[5]['jacobian']
            signed=bool(np.all(self.backend.C[core]*shift+diagonal>=0))
            if not signed:
                diagonal.fill(0);diagonal[ap]+=channel_conductance(data[1],self.gbar_nS).sum(axis=1)
                if sn is not None:diagonal[sp]+=sg
                if ca is not None:diagonal[cap]+=data[5]['frozen_conductance']
            correction=a.small.solve(-r,0.,diagonal);total+=1
            linear_error=float(np.linalg.norm(a.K@correction+diagonal*correction+r))
            row['linear_residual_l2_pA']=linear_error;row['exact_local_jacobian']=signed
            if not np.isfinite(correction).all() or linear_error>max(.25*threshold,.1*norm):
                return None,dict(accepted=False,reason='compact_linear_failed',history=history,total_iterations=total)
            for backtrack in range(9):
                trial=delta+(2.**(-backtrack))*correction
                try:rr,nn,dd=evaluate(trial)
                except FloatingPointError:continue
                if np.isfinite(nn) and (nn<norm or nn<=threshold):
                    delta=trial;r,norm,data=rr,nn,dd;row['step_scale']=2.**(-backtrack);break
            else:return None,dict(accepted=False,reason='no_nonlinear_decrease',history=history,total_iterations=total)
    except (FloatingPointError,ArithmeticError,np.linalg.LinAlgError) as exc:
        return None,dict(accepted=False,reason='compact_numerical_failure',detail=str(exc),history=history,total_iterations=total)
    return None,dict(accepted=False,reason='stage_residual_failed',history=history,threshold_pA=threshold,total_iterations=total)
