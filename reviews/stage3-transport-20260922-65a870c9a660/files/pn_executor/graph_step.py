from graph_stage import graph_stage
"""Candidate full-mass SDIRK step, derived from the frozen fine step.

Only mass action changes. C retains physical membrane capacitance for support
and lower-bound checks. Backend must certify M-diag(C) positive semidefinite;
otherwise the inherited Jacobian lower-bound check is not justified. Gates,
Ca chemistry, charges, presampled inputs and commit rules are unchanged.
See work/pn_mass_runtime_20260913/derivation.json and step.diff.
"""
import numpy as np
from pn_coupled_ionic import GAMMA, METHOD
from resident import channels as stage_channels, channel_conductance


def advance_graph_resident(self,dt_ns,current_pA,*,rtol=1e-9,atol=2e-12,maxiter=220,
            max_newton=8,gate_atol=1e-12,progress=None,synaptic_stages=None,_local_channel=None,stage_predictor='previous',
            stage_observation_nodes=None):
    self._assert_model();cp=self.cp
    if not isinstance(stage_predictor,str) or stage_predictor not in ('previous','linear'):
        raise ValueError('Unknown SDIRK stage predictor')
    if cp.iscomplexobj(current_pA):raise ValueError('Real current required')
    current=cp.asarray(current_pA,dtype=cp.float64)
    if (type(dt_ns) is not int or dt_ns<=0 or current.shape!=self.voltage.shape
            or not bool(cp.isfinite(current).all()) or not bool(cp.isfinite(self.voltage).all())
            or self.gates.shape!=(len(self.active_nodes),4) or not cp.isfinite(self.gates).all()
            or cp.any(self.gates<0) or cp.any(self.gates>1)
            or self.ionic_charge_pC.shape!=(3,) or not cp.isfinite(self.ionic_charge_pC).all()):
        raise ValueError('Finite full state, probabilities and positive integer timestep required')
    for value,zero_allowed in [(rtol,False),(atol,True),(gate_atol,False)]:
        if (not np.isscalar(value) or np.iscomplexobj(value) or not np.isfinite(value)
                or (value<0 if zero_allowed else value<=0)):
            raise ValueError('Finite numerical tolerances required')
    if type(maxiter) is not int or maxiter<1 or type(max_newton) is not int or max_newton<1:
        raise ValueError('Positive integer iteration budgets required')
    dt=dt_ns*1e-9;q=GAMMA*dt;shift=1/q;C=self.backend.levels[0]['C'];nodes=self._nodes_gpu
    observation_nodes=None
    if stage_observation_nodes is not None:
        raw=np.asarray(stage_observation_nodes)
        if (raw.ndim!=1 or raw.dtype.kind not in 'iu' or not len(raw)
                or len(np.unique(raw))!=len(raw) or np.any(raw<0) or np.any(raw>=len(C))):
            raise ValueError('Unique valid stage observation nodes required')
        observation_nodes=cp.asarray(raw,dtype=cp.int64)
    # Inputs are sampled by the caller BEFORE Newton, at t+gamma*dt and
    # t+dt. They are read-only stage coefficients, never callbacks which
    # could advance upstream history during rejected Newton iterations.
    synapses=[None,None]
    if synaptic_stages is not None:
        if not isinstance(synaptic_stages,(list,tuple)) or len(synaptic_stages)!=2:
            raise ValueError('Two presampled synaptic stages required')
        previous_nodes=None
        for i,spec in enumerate(synaptic_stages):
            if set(spec)!={'nodes','conductance_nS','reversal_mV'}:
                raise ValueError('Explicit anatomical nodes, conductance and reversal required')
            raw=np.asarray(spec['nodes']);g=np.asarray(spec['conductance_nS']);E=np.asarray(spec['reversal_mV'])
            if (raw.ndim!=1 or raw.dtype.kind not in 'iu' or not len(raw)
                    or len(np.unique(raw))!=len(raw) or np.any(raw<0) or np.any(raw>=len(C))
                    or g.shape!=raw.shape or g.dtype.kind not in 'fiu' or not np.isfinite(g).all() or np.any(g<0)
                    or E.dtype.kind not in 'fiu' or E.shape not in [(),raw.shape] or not np.isfinite(E).all()):
                raise ValueError('Finite nonnegative synaptic conductance on unique physical nodes required')
            if previous_nodes is not None and not np.array_equal(previous_nodes,raw):
                raise ValueError('Synaptic anatomical support must match across stages')
            previous_nodes=raw.copy();sn=cp.asarray(raw,dtype=cp.int64)
            if not bool(cp.all(C[sn]>0)):raise ValueError('Synapses require physical membrane capacitance')
            union=np.union1d(self.active_nodes,raw);jn=cp.asarray(union,dtype=cp.int64)
            ap=cp.asarray(np.searchsorted(union,self.active_nodes));sp=cp.asarray(np.searchsorted(union,raw))
            sg=cp.asarray(g,dtype=cp.float64);se=cp.asarray(E,dtype=cp.float64)
            synapses[i]=(sn,sg,se,jn,ap,sp)
    ca=_local_channel;ca_updates=[None,None]
    if ca is not None:
        if getattr(self,'calcium_port',None) is not ca or ca.time_ns!=self.time_ns:
            raise ValueError('Local channel must belong to this session at the same clock')
        ca.assert_state()
        cn=cp.asarray(ca.nodes,dtype=cp.int64)
        if np.any(ca.nodes>=len(C)) or not bool(cp.all(C[cn]>0)):
            raise ValueError('Calcium requires actual membrane nodes')
        for i,syn in enumerate(synapses):
            union=np.union1d(self.active_nodes,ca.nodes)
            if syn is not None:union=np.union1d(union,cp.asnumpy(syn[0]))
            ca_updates[i]=(cp.asarray(union),cp.asarray(np.searchsorted(union,self.active_nodes)),
                cp.asarray(np.searchsorted(union,ca.nodes)),
                None if syn is None else cp.asarray(np.searchsorted(union,cp.asnumpy(syn[0]))))
    def fallback_stage(vbase,xbase,guess,synapse,ca_base,ca_update,threshold_guess=None):
        v=guess;history=[];total=0;threshold=None
        def evaluate(w):
            x,ionic,jac,gate_error=stage_channels(w[nodes]+self.leak_reversal_mV,
                xbase,q,self.gbar_nS,self.reversal_mV)
            residual=self.backend.mass_action((w-vbase)*shift)+self.backend.action(w)-current
            residual[nodes]+=cp.asarray(ionic.sum(axis=1))
            if synapse is not None:
                sn,sg,se,_,_,_=synapse
                residual[sn]+=sg*(w[sn]+self.leak_reversal_mV-se)
            cdata=None
            if ca is not None:
                cdata=ca.stage(w[cn]+self.leak_reversal_mV,ca_base,q)
                residual[cn]+=cp.asarray(cdata['current']);gate_error=max(gate_error,cdata['gate_error'])
            return residual,float(cp.linalg.norm(residual)),x,ionic,jac,gate_error,cdata
        try:residual,norm,x,ionic,jac,gate_error,cdata=evaluate(v)
        except FloatingPointError:return None,dict(accepted=False,reason='invalid_stage_kinetics',history=history)
        # Relative acceptance is anchored to the legacy initial iterate,
        # so changing the predictor never changes the residual threshold.
        try:reference_norm=norm if threshold_guess is None else evaluate(threshold_guess)[1]
        except FloatingPointError:return None,dict(accepted=False,reason='invalid_reference_kinetics',history=history)
        threshold=max(atol,rtol*reference_norm)
        for iteration in range(max_newton+1):
            row=dict(iteration=iteration,residual_l2_pA=norm,gate_residual=gate_error);history.append(row)
            if progress:progress(dict(stage_iteration=iteration,residual_l2_pA=norm))
            if np.isfinite(norm) and norm<=threshold and gate_error<=gate_atol:
                return (v,x,ionic,cdata),dict(accepted=True,history=history,threshold_pA=threshold,total_iterations=total)
            if not np.isfinite(norm) or iteration==max_newton:break
            diagonal=cp.asarray(jac);correction_nodes=nodes
            if synapse is not None:
                sn,sg,se,correction_nodes,ap,sp=synapse
                combined=cp.zeros(len(correction_nodes),dtype=cp.float64)
                combined[ap]+=diagonal;combined[sp]+=sg;diagonal=combined
            if ca is not None:
                correction_nodes,nap,ncp,nsp=ca_update
                diagonal=cp.zeros(len(correction_nodes),dtype=cp.float64)
                diagonal[nap]+=cp.asarray(jac);diagonal[ncp]+=cp.asarray(cdata['jacobian'])
                if synapse is not None:diagonal[nsp]+=synapse[1]
            # If exact Newton cannot use CG safely, a positive frozen-gate
            # slope is a quasi-Newton correction. The full nonlinear
            # residual remains the sole electrical acceptance criterion.
            signed=bool(cp.all(C[correction_nodes]*shift+diagonal>=0))
            if not signed:
                diagonal=cp.asarray(channel_conductance(x,self.gbar_nS).sum(axis=1))
                if synapse is not None and ca is None:
                    combined=cp.zeros(len(correction_nodes),dtype=cp.float64)
                    combined[ap]+=diagonal;combined[sp]+=sg;diagonal=combined
                if ca is not None:
                    diagonal=cp.zeros(len(correction_nodes),dtype=cp.float64)
                    diagonal[nap]+=cp.asarray(channel_conductance(x,self.gbar_nS).sum(axis=1))
                    diagonal[ncp]+=cp.asarray(cdata['frozen_conductance'])
                    if synapse is not None:diagonal[nsp]+=synapse[1]
            correction,report=self.backend.solve(-residual,shift=shift,
                rtol=min(.1,.25*threshold/max(norm,1e-300)),atol=.25*threshold,
                maxiter=maxiter,diagonal_update=(correction_nodes,diagonal),
                diagonal_is_jacobian=signed)
            total+=report['iterations'];row['linear']=report;row['exact_local_jacobian']=signed
            if report['info']!=0 or not report['fine_residual_passed']:
                return None,dict(accepted=False,reason='linear_solve_failed',history=history,total_iterations=total)
            for backtrack in range(9):
                trial=v+(2.**(-backtrack))*correction
                try:trial_result=evaluate(trial)
                except FloatingPointError:continue
                if np.isfinite(trial_result[1]) and (trial_result[1]<norm or trial_result[1]<=threshold):
                    v=trial;residual,norm,x,ionic,jac,gate_error,cdata=trial_result
                    row['step_scale']=2.**(-backtrack);break
            else:return None,dict(accepted=False,reason='no_nonlinear_decrease',history=history,total_iterations=total)
        return None,dict(accepted=False,reason='stage_residual_failed',history=history,threshold_pA=threshold,total_iterations=total)
    def stage(vbase,xbase,guess,synapse,ca_base,ca_update,threshold_guess=None):
        return graph_stage(self,fallback_stage,vbase,xbase,guess,synapse,ca_base,ca_update,threshold_guess,
            shift=shift,q=q,current=current,rtol=rtol,atol=atol,gate_atol=gate_atol)
    first,r1=stage(self.voltage,self.gates,self.voltage,synapses[0],None if ca is None else ca.gates,ca_updates[0])
    if first is None:return dict(accepted=False,time_ns=self.time_ns,method=METHOD,stages=[r1])
    v1,x1,i1,c1=first;ratio=(1-GAMMA)/GAMMA
    vbase=self.voltage+ratio*(v1-self.voltage)
    xbase=self.gates+ratio*(x1-self.gates)
    cbase=None if ca is None else ca.gates+ratio*(c1['gates']-ca.gates)
    # Predictor changes only the starting iterate; the same full residual
    # and physical stages still decide acceptance. No extra history state.
    guess=v1 if stage_predictor=='previous' else self.voltage+(v1-self.voltage)/GAMMA
    second,r2=stage(vbase,xbase,guess,synapses[1],cbase,ca_updates[1],
                    threshold_guess=None if stage_predictor=='previous' else v1)
    if second is None:return dict(accepted=False,time_ns=self.time_ns,method=METHOD,stages=[r1,r2])
    v2,x2,i2,c2=second;dq=dt*((1-GAMMA)*i1.sum(axis=0)+GAMMA*i2.sum(axis=0))
    charge=self.ionic_charge_pC+dq
    if not cp.isfinite(charge).all() or not bool(cp.isfinite(v2).all()):
        return dict(accepted=False,time_ns=self.time_ns,method=METHOD,reason='nonfinite_final_state',stages=[r1,r2])
    synaptic_charge=None
    if synaptic_stages is not None:
        currents=[]
        for v,syn in zip((v1,v2),synapses):
            sn,sg,se,_,_,_=syn
            currents.append(float(cp.sum(sg*(v[sn]+self.leak_reversal_mV-se))))
        synaptic_charge=dt*((1-GAMMA)*currents[0]+GAMMA*currents[1])
        if not np.isfinite(synaptic_charge):
            return dict(accepted=False,time_ns=self.time_ns,method=METHOD,reason='nonfinite_synaptic_charge',stages=[r1,r2])
    # Read accepted stage voltages before any state commits. This optional
    # observation lets callers integrate receptor/axial currents with the
    # same quadrature, instead of inferring them from endpoint voltages.
    # Observation cannot change Newton, physical state or its identity.
    observations=None
    if observation_nodes is not None:
        observations=dict(nodes=cp.asnumpy(observation_nodes).tolist(),
            start_ns=self.time_ns,dt_ns=dt_ns,fractions=[GAMMA,1.],
            voltage_mV=[(cp.asnumpy(v[observation_nodes])+self.leak_reversal_mV).tolist() for v in (v1,v2)])
    ca_proposal=None
    if ca is not None:
        try:ca_proposal=ca.final_proposal(dt_ns,c1,c2)
        except (ValueError,FloatingPointError) as exc:
            return dict(accepted=False,time_ns=self.time_ns,method=METHOD,reason='calcium_chemistry_rejected',
                        detail=str(exc),stages=[r1,r2])
        ca.commit(ca_proposal)
    # Same RK quadrature as the two accepted membrane/gate equations.
    # This is accumulated ionic charge, not a concentration model.
    self.voltage=v2;self.gates=x2;self.ionic_charge_pC=charge;self.time_ns+=dt_ns
    result=dict(accepted=True,time_ns=self.time_ns,method=METHOD,stage_predictor=stage_predictor,stages=[r1,r2],
                total_iterations=r1['total_iterations']+r2['total_iterations'],
                ionic_current_outward_pA=cp.asnumpy(i2.sum(axis=0)).tolist(),ionic_charge_increment_pC=cp.asnumpy(dq).tolist())
    if synaptic_charge is not None:
        result['synaptic_outward_charge_increment_pC']=synaptic_charge
    if observations is not None:result['stage_observations']=observations
    if ca_proposal is not None:
        chem=ca_proposal['chemistry']
        result['calcium']=dict(outward_charge_increment_pC=ca_proposal['dq'].tolist(),
            site_charge_increment_pC=chem['calcium_charge_increment_pC'].tolist(),
            release_increment=chem['release_increment'].tolist(),
            allocation_error_pC=ca_proposal['allocation_error_pC'],
            chemical_balance_max_pC=float(abs(chem['calcium_balance_pC']).max()),
            pool_balance_max=float(abs(chem['pool_balance']).max()),
            activation_stages=chem['output_activation_stages'].tolist())
    return result
