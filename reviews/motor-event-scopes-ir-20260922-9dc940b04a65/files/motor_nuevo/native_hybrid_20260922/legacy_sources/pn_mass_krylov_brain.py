"""Versioned numerical policy for the same173 PN/CNS equations and histories."""
import copy
from pn_inhibitory_closure_brain import GpuPnInhibitoryClosureBrain
from pn_mass_krylov_backend import FullMassKrylovBackend,POLICY
from pn_cholinergic_cns_brain import KEYS
from kcgamma_regional_brain import _record_hash

class GpuPnMassKrylovBrain(GpuPnInhibitoryClosureBrain):
    SCHEMA='matrix_pn_mass_krylov_brain_v1'

    def _install_solver(self):
        source=copy.copy(self._online_source);source.pn=copy.copy(source.pn)
        source.pn.backend=FullMassKrylovBackend.adopt(source.pn.backend)
        self._online_source=source

    @classmethod
    def adopt(cls,parent):
        if type(parent) is not GpuPnInhibitoryClosureBrain or 'mass' not in parent.pn_online_manifest:
            raise ValueError('Exact173 full-mass parent required')
        before=parent.state_dict();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__);obj._install_solver()
        m=copy.deepcopy(parent.pn_online_manifest)
        if 'mass_linear_solver' in m:raise ValueError('Numerical policy already migrated')
        m['mass_linear_solver']=dict(policy=POLICY,operator_identity=obj._online_source.pn.backend.identity,
            adoption_time_ns=obj.time_ns,physical_state_preserved=True,cache_is_physical_state=False,
            scope='Same signed Jacobian, full mass and tolerances; passive LU preconditioner, zero initial iterate, true residual and direct fallback.')
        m['record_sha256']=_record_hash(m);obj.pn_online_manifest=m
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online()
        after=obj.state_dict()
        for k in ('schema','pn_online_manifest'):before.pop(k);after.pop(k)
        if _record_hash(before)!=_record_hash(after):raise ValueError('Numerical policy migration changed physical state')
        return obj

    def validate_online(self):
        super().validate_online();m=self.pn_online_manifest['mass_linear_solver'];backend=self._online_source.pn.backend
        if (type(backend) is not FullMassKrylovBackend or m['policy']!=POLICY or backend.solver_policy!=POLICY
            or m['operator_identity']!=backend.identity):raise ValueError('Full-mass numerical policy mismatch')

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete Krylov PN/CNS state')
        parent=dict(saved,schema=GpuPnInhibitoryClosureBrain.SCHEMA)
        base=GpuPnInhibitoryClosureBrain.from_state(brain,parent)
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__);obj._install_solver()
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online();return obj

    @staticmethod
    def backend_identity():
        out=GpuPnInhibitoryClosureBrain.backend_identity();out['PN_mass_linear_solver']=POLICY;return out
