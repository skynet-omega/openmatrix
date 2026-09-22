"""Full joint RC membrane history using the existing fine multigrid solver.

Implicit Euler is an explicit numerical approximation. No steady-state reset,
static Schur replacement or CNS admission is implied. Every volume, collar
and retained exterior unknown remains in the saved voltage state.
"""
import hashlib
import numpy as np
from compensated_csr import CompensatedCSR


class JointPNTransient:
    def __init__(self, multigrid, *, node_identity):
        if not isinstance(node_identity,str) or not node_identity:
            raise ValueError('Explicit anatomical node-order identity required')
        self.mg=multigrid
        fine=multigrid.levels[0];self.G=fine.G;self.C=fine.C
        h=hashlib.sha256(node_identity.encode())
        for value in [self.G.data,self.G.indices,self.G.indptr,self.C]:
            h.update(str((value.dtype.str,value.shape)).encode())
            # Do not materialize a bytes copy of the multi-GB fine matrix.
            h.update(memoryview(value).cast('B'))
        self.identity=h.hexdigest();self.voltage=np.zeros(len(self.C));self.time_ns=0
        self._model_arrays=(self.G.data,self.G.indices,self.G.indptr,self.C)
        self.action=CompensatedCSR(self.G)

    def _assert_model(self):
        if self.mg.levels[0].G is not self.G or self.mg.levels[0].C is not self.C:
            raise ValueError('Fine operator changed; construct and migrate explicitly')
        for expected,actual in zip(self._model_arrays,(self.G.data,self.G.indices,self.G.indptr,self.C)):
            if actual is not expected or actual.flags.writeable:
                raise ValueError('Fine coefficient buffers replaced or made writable')

    def advance(self,dt_ns,current_pA,*,rtol=1e-9,atol=2e-12,maxiter=220,progress=None):
        self._assert_model()
        current=np.asarray(current_pA,dtype=float)
        if (type(dt_ns) is not int or dt_ns<=0 or current.shape!=self.C.shape
                or not np.isfinite(current).all()):
            raise ValueError('Positive integer timestep and finite full current required')
        shift=1e9/dt_ns
        # Solve for the increment. This retains the complete old capacitive
        # state without subtracting two large C*v/dt terms in the RHS.
        rhs=current-self.action(self.voltage)
        change,report=self.mg.solve(rhs,shift=shift,rtol=rtol,atol=atol,
            maxiter=maxiter,compensated_rows=True,progress=progress)
        if report['info']!=0 or not report['fine_residual_passed']:
            return dict(accepted=False,time_ns=self.time_ns,solver=report)
        candidate=self.voltage+change
        # Audit the actually representable next state, not just CG's change.
        capacitive=self.C*((candidate-self.voltage)*shift)
        residual=capacitive+self.action(candidate)-current
        norm=float(np.linalg.norm(residual))
        threshold=max(atol,rtol*float(np.linalg.norm(rhs)))
        accepted=bool(np.isfinite(candidate).all() and norm<=threshold)
        result=dict(accepted=accepted,time_ns=self.time_ns+(dt_ns if accepted else 0),
            solver=report,dynamic_residual_l2_pA=norm,threshold_pA=threshold)
        if accepted:self.voltage=candidate;self.time_ns+=dt_ns
        return result

    def state_dict(self):
        self._assert_model()
        return dict(schema='joint_pn_transient_v1',identity=self.identity,
                    time_ns=self.time_ns,voltage_delta_mV=self.voltage.copy())

    def load_state_dict(self,state):
        self._assert_model()
        if (set(state)!={'schema','identity','time_ns','voltage_delta_mV'}
                or state['schema']!='joint_pn_transient_v1' or state['identity']!=self.identity
                or type(state['time_ns']) is not int or state['time_ns']<0):
            raise ValueError('Incompatible temporal state or clock')
        v=np.asarray(state['voltage_delta_mV'])
        if v.shape!=self.C.shape or v.dtype.kind!='f' or not np.isfinite(v).all():
            raise ValueError('Complete finite membrane state required')
        self.voltage=np.array(v,dtype=float,copy=True);self.time_ns=state['time_ns']
