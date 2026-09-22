"""Change only linear factorization of the179 PN/CNS candidate."""
import copy
import numpy as np
from orn_peripheral_terminal_brain import GpuOrnPeripheralTerminalBrain
from orn_peripheral_terminal import POLICY as ORN_POLICY,PARAMETERS
from pn_inhibitory_closure_brain import GpuPnInhibitoryClosureBrain
from pn_mass_krylov_backend import POLICY as KRYLOV_POLICY
from pn_graph_elimination_backend import FullMassGraphBackend,POLICY
from pn_cholinergic_cns_brain import KEYS
from kcgamma_regional_brain import _record_hash


class GpuPnGraphSolverBrain(GpuOrnPeripheralTerminalBrain):
    SCHEMA='matrix_pn_graph_solver_brain_v1'

    def _install_graph(self):
        source=copy.copy(self._online_source);source.pn=copy.copy(source.pn)
        source.pn.backend=FullMassGraphBackend.adopt(source.pn.backend)
        self._online_source=source

    @classmethod
    def adopt(cls,parent):
        if type(parent) is not GpuOrnPeripheralTerminalBrain:
            raise ValueError('Exact179 mass ORN parent required')
        before=parent.state_dict();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__);obj._install_graph()
        m=copy.deepcopy(parent.pn_online_manifest)
        if 'graph_linear_solver' in m:raise ValueError('Graph solver already adopted')
        b=obj._online_source.pn.backend
        m['graph_linear_solver']=dict(policy=POLICY,operator_identity=b.identity,adoption_time_ns=obj.time_ns,
            physical_state_preserved=True,eliminated_coordinates=len(b._graph_plan.steps),
            retained_junctions=len(b._graph_plan.core),all_coordinates_reconstructed=True,
            scope='Same fullM/G and signed ionic diagonal. Exact algebraic elimination, original residual, Krylov/direct fallback. No new biological or dynamic reduction.')
        m['record_sha256']=_record_hash(m);obj.pn_online_manifest=m
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online()
        after=obj.state_dict()
        for k in ('schema','pn_online_manifest'):before.pop(k);after.pop(k)
        if _record_hash(before)!=_record_hash(after):raise ValueError('Graph adoption changed physical history')
        return obj

    def validate_online(self):
        # The frozen176 implementation requires exact backend type. Reapply
        # its identity/policy checks for this explicit subclass, plus179's
        # terminal policy; retain all173 biological/state validations.
        GpuPnInhibitoryClosureBrain.validate_online(self)
        b=self._online_source.pn.backend;m=self.pn_online_manifest
        old=m['mass_linear_solver'];new=m['graph_linear_solver'];orn=m['orn_peripheral_terminal']
        if (type(b) is not FullMassGraphBackend or old['policy']!=KRYLOV_POLICY
                or b.solver_policy!=KRYLOV_POLICY or new['policy']!=POLICY or b.graph_solver_policy!=POLICY
                or old['operator_identity']!=b.identity or new['operator_identity']!=b.identity):
            raise ValueError('Graph factorization identity or fallback policy changed')
        if (orn['policy']!=ORN_POLICY or orn['parameters']!=PARAMETERS or type(orn['recurrent_connected']) is not bool
                or not np.array_equal(orn['source_ids'],self.brain.node_ids[self._orn_rows])):
            raise ValueError('Changed ORN terminal policy')

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete graph-solver CNS state')
        base=GpuOrnPeripheralTerminalBrain.from_state(brain,dict(saved,schema=GpuOrnPeripheralTerminalBrain.SCHEMA))
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__);obj._install_graph()
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online();return obj

    @staticmethod
    def backend_identity():
        out=GpuOrnPeripheralTerminalBrain.backend_identity();out['PN_graph_linear_solver']=POLICY;return out
