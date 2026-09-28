"""Experimental ionic adapter on the full, unchanged fine PN passive operator.

Units: mV, pA, nS, nF, seconds (integer nanosecond clock). Positive ionic
current is outward. Gates follow transferred Drosophila NaT/NaP/K kinetics;
densities and placement are mandatory inputs, not identified PN physiology.
Rush-Larsen at OLD local voltage followed by implicit membrane voltage is
first-order splitting. No threshold reset, voltage clipping or holding-current
subtraction. Gating equilibrium at the passive reversal is NOT an active rest.
"""
import hashlib
import json
import inspect
import numpy as np
from kc_four_port_active import channel_rates
from neck_gpu_transient import GPUJointPNTransient


def channel_conductance(gates, gbar_nS):
    """NaT, NaP, K columns; gates are m, h, p, n."""
    m,h,p,n=np.asarray(gates).T
    return np.asarray(gbar_nS)*np.column_stack((m**3*h,p,n**4))


class FinePNIonicSession(GPUJointPNTransient):
    def __init__(self,backend,*,node_identity,active_nodes,gbar_nS,
                 reversal_mV,leak_reversal_mV,channel_provenance):
        # Validate before hashing/allocating the multi-GB fine system.
        nodes=np.asarray(active_nodes);gbar=np.asarray(gbar_nS);rev=np.asarray(reversal_mV)
        C=backend.cpu.levels[0].C
        if (nodes.ndim!=1 or nodes.dtype.kind not in 'iu' or not len(nodes)
                or len(np.unique(nodes))!=len(nodes) or np.any(nodes<0) or np.any(nodes>=len(C))):
            raise ValueError('Nonempty unique anatomical node indices required')
        if np.any(C[nodes]<=0):raise ValueError('Channels require physical membrane capacitance')
        if (gbar.shape!=(len(nodes),3) or gbar.dtype.kind not in 'fiu'
                or not np.isfinite(gbar).all() or np.any(gbar<0)
                or rev.shape!=(3,) or rev.dtype.kind not in 'fiu' or not np.isfinite(rev).all()
                or not np.isscalar(leak_reversal_mV) or np.iscomplexobj(leak_reversal_mV)
                or not np.isfinite(leak_reversal_mV)
                or not isinstance(channel_provenance,str) or not channel_provenance):
            raise ValueError('Explicit finite channel densities, reversals and provenance required')
        super().__init__(backend,node_identity=node_identity)
        def frozen(a,dtype):
            a=np.asarray(a,dtype=dtype)
            return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
        self.active_nodes=frozen(nodes,np.int64);self.gbar_nS=frozen(gbar,np.float64)
        self.reversal_mV=frozen(rev,np.float64);self.leak_reversal_mV=float(leak_reversal_mV)
        self._nodes_gpu=self.cp.asarray(self.active_nodes)
        self.channel_provenance=channel_provenance
        h=hashlib.sha256(self.identity.encode())
        h.update(json.dumps(dict(schema='pn_fine_ionic_v1',method='RL_old_voltage_BE_v1',
            leak_reversal_mV=self.leak_reversal_mV,provenance=channel_provenance),sort_keys=True).encode())
        h.update(inspect.getsource(channel_rates).encode())
        h.update(inspect.getsource(channel_conductance).encode())
        for a in (self.active_nodes,self.gbar_nS,self.reversal_mV):h.update(a.tobytes())
        self.identity=h.hexdigest()
        self._ionic_definition=(self.active_nodes,self.gbar_nS,self.reversal_mV,
                                self.leak_reversal_mV,self.channel_provenance)
        self.gates=channel_rates(np.full(len(nodes),self.leak_reversal_mV))[0]
        self.ionic_charge_pC=np.zeros(3)  # accepted-step quadrature, not concentrations

    def _assert_model(self):
        super()._assert_model()
        arrays=(self.active_nodes,self.gbar_nS,self.reversal_mV)
        if (any(a is not b for a,b in zip(arrays,self._ionic_definition[:3]))
                or self.leak_reversal_mV!=self._ionic_definition[3]
                or self.channel_provenance!=self._ionic_definition[4]):
            raise ValueError('Ionic definition changed; construct and migrate explicitly')

    def ionic_current_pA(self):
        v=self.cp.asnumpy(self.voltage[self._nodes_gpu])+self.leak_reversal_mV
        return channel_conductance(self.gates,self.gbar_nS)*(v[:,None]-self.reversal_mV)

    def advance(self,dt_ns,current_pA,*,rtol=1e-9,atol=2e-12,maxiter=220,progress=None):
        self._assert_model();cp=self.cp
        if cp.iscomplexobj(current_pA):raise ValueError('Real current required')
        current=cp.asarray(current_pA,dtype=cp.float64)
        if (type(dt_ns) is not int or dt_ns<=0 or current.shape!=self.voltage.shape
                or not bool(cp.isfinite(current).all()) or not bool(cp.isfinite(self.voltage).all())
                or self.gates.shape!=(len(self.active_nodes),4) or not np.isfinite(self.gates).all()
                or np.any(self.gates<0) or np.any(self.gates>1)):
            raise ValueError('Finite full state, probabilities and positive integer timestep required')
        dt=dt_ns*1e-9;shift=1/dt
        local=cp.asnumpy(self.voltage[self._nodes_gpu])+self.leak_reversal_mV
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            steady,tau=channel_rates(local)
            gates=steady+(self.gates-steady)*np.exp(-dt/tau)
        if (not np.isfinite(tau).all() or np.any(tau<=0) or not np.isfinite(gates).all()
                or np.any(gates<0) or np.any(gates>1)):
            raise FloatingPointError('Invalid gate kinetics; no clipping or step committed')
        conductance=channel_conductance(gates,self.gbar_nS)
        diagonal=cp.asarray(conductance.sum(axis=1))
        drive=cp.asarray(conductance@(self.reversal_mV-self.leak_reversal_mV))
        rhs=current-self.backend.action(self.voltage)
        rhs[self._nodes_gpu]+=drive-diagonal*self.voltage[self._nodes_gpu]
        change,report=self.backend.solve(rhs,shift=shift,rtol=rtol,atol=atol,
            maxiter=maxiter,progress=progress,diagonal_update=(self._nodes_gpu,diagonal))
        if report['info']!=0 or not report['fine_residual_passed']:
            return dict(accepted=False,time_ns=self.time_ns,solver=report)
        candidate=self.voltage+change
        threshold=max(atol,rtol*float(cp.linalg.norm(rhs)))
        corrections=[];used_iterations=report['iterations']
        # Increment convergence need not survive rounding when it is added to
        # the old voltage. Correct the residual of the representable state,
        # with the SAME gates, operator and external physical acceptance gate.
        # The correction solve spends only the remaining iteration budget.
        for attempt in range(4):
            ionic=conductance*((cp.asnumpy(candidate[self._nodes_gpu])+self.leak_reversal_mV)[:,None]-self.reversal_mV)
            residual=self.backend.levels[0]['C']*((candidate-self.voltage)*shift)+self.backend.action(candidate)-current
            residual[self._nodes_gpu]+=cp.asarray(ionic.sum(axis=1))
            norm=float(cp.linalg.norm(residual))
            if (not np.isfinite(norm) or norm<=threshold or attempt==3 or used_iterations>=maxiter):break
            correction,cr=self.backend.solve(-residual,shift=shift,rtol=min(.1,threshold*.25/norm),
                atol=threshold*.25,maxiter=maxiter-used_iterations,
                diagonal_update=(self._nodes_gpu,diagonal),progress=progress)
            corrections.append(cr);used_iterations+=cr['iterations']
            if cr['info']!=0 or not cr['fine_residual_passed']:break
            candidate=candidate+correction
        charge=self.ionic_charge_pC+dt*ionic.sum(axis=0)
        accepted=bool(np.isfinite(norm) and norm<=threshold and bool(cp.isfinite(candidate).all()) and np.isfinite(charge).all())
        result=dict(accepted=accepted,time_ns=self.time_ns+(dt_ns if accepted else 0),solver=report,
            state_corrections=corrections,total_iterations=used_iterations,
            dynamic_residual_l2_pA=norm,threshold_pA=threshold,
            ionic_current_outward_pA=ionic.sum(axis=0).tolist(),
            ionic_charge_increment_pC=(dt*ionic.sum(axis=0)).tolist())
        if accepted:
            self.voltage=candidate;self.gates=gates;self.ionic_charge_pC=charge;self.time_ns+=dt_ns
        return result

    def state_dict(self):
        self._assert_model()
        return dict(schema='pn_fine_ionic_v1',identity=self.identity,time_ns=self.time_ns,
            voltage_delta_mV=self.cp.asnumpy(self.voltage),gates=self.gates.copy(),
            ionic_charge_pC=self.ionic_charge_pC.copy())

    def load_state_dict(self,state):
        self._assert_model()
        if (set(state)!={'schema','identity','time_ns','voltage_delta_mV','gates','ionic_charge_pC'}
                or state['schema']!='pn_fine_ionic_v1' or state['identity']!=self.identity
                or type(state['time_ns']) is not int or state['time_ns']<0):
            raise ValueError('Incompatible ionic state, kinetics, anatomy or clock')
        values=[]
        for key,shape in [('voltage_delta_mV',self.C.shape),('gates',(len(self.active_nodes),4)),('ionic_charge_pC',(3,))]:
            value=np.asarray(state[key])
            if value.shape!=shape or value.dtype.kind!='f' or not np.isfinite(value).all():
                raise ValueError('Complete finite ionic state required')
            values.append(np.array(value,dtype=np.float64,copy=True))
        v,gates,charge=values
        if np.any(gates<0) or np.any(gates>1):raise ValueError('Gates must be probabilities')
        v_gpu=self.cp.asarray(v)  # Allocation must succeed before replacing any live state.
        self.voltage=v_gpu;self.gates=gates;self.ionic_charge_pC=charge;self.time_ns=state['time_ns']
