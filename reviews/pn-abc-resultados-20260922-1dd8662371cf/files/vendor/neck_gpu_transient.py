"""Resident GPU temporal state; host copies occur only for explicit snapshots."""
import numpy as np
from neck_joint_transient import JointPNTransient


class GPUJointPNTransient(JointPNTransient):
    def __init__(self,backend,*,node_identity):
        backend.assert_model()
        super().__init__(backend.cpu,node_identity=node_identity)
        self.backend=backend;self.cp=backend.cp;self.voltage=self.cp.zeros(len(self.C),dtype=self.cp.float64)

    def _assert_model(self):
        super()._assert_model();self.backend.assert_model()

    def advance(self,dt_ns,current_pA,*,rtol=1e-9,atol=2e-12,maxiter=220,progress=None):
        self._assert_model();cp=self.cp
        if cp.iscomplexobj(current_pA):raise ValueError('Real current required')
        current=cp.asarray(current_pA,dtype=cp.float64)
        if type(dt_ns) is not int or dt_ns<=0 or current.shape!=self.voltage.shape or not bool(cp.isfinite(current).all()):
            raise ValueError('Positive integer timestep and finite full current required')
        shift=1e9/dt_ns;rhs=current-self.backend.action(self.voltage)
        change,report=self.backend.solve(rhs,shift=shift,rtol=rtol,atol=atol,maxiter=maxiter,progress=progress)
        if report['info']!=0 or not report['fine_residual_passed']:return dict(accepted=False,time_ns=self.time_ns,solver=report)
        candidate=self.voltage+change
        residual=self.backend.levels[0]['C']*((candidate-self.voltage)*shift)+self.backend.action(candidate)-current
        norm=float(cp.linalg.norm(residual));threshold=max(atol,rtol*float(cp.linalg.norm(rhs)))
        accepted=bool(np.isfinite(norm) and norm<=threshold and bool(cp.isfinite(candidate).all()))
        result=dict(accepted=accepted,time_ns=self.time_ns+(dt_ns if accepted else 0),solver=report,dynamic_residual_l2_pA=norm,threshold_pA=threshold)
        if accepted:self.voltage=candidate;self.time_ns+=dt_ns
        return result

    def state_dict(self):
        self._assert_model()
        return dict(schema='joint_pn_transient_v1',identity=self.identity,time_ns=self.time_ns,voltage_delta_mV=self.cp.asnumpy(self.voltage))

    def load_state_dict(self,state):
        super().load_state_dict(state);self.voltage=self.cp.asarray(self.voltage)
