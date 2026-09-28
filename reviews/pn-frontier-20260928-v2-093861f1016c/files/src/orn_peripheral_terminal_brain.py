"""Separate peripheral forcing from6242 terminal input routes, once, in CNS."""
import copy
import numpy as np
from pn_mass_krylov_brain import GpuPnMassKrylovBrain
from pn_inhibitory_closure_brain import GpuPnInhibitoryClosureBrain
from pn_cholinergic_cns_brain import KEYS
from kcgamma_regional_brain import _record_hash
from orn_peripheral_terminal import POLICY, PARAMETERS, CUDA_CODE


class OrnPeripheralTerminalMixin:
    @classmethod
    def adopt(cls, parent, *, recurrent_connected=True):
        if type(parent) is not cls.PARENT or type(recurrent_connected) is not bool:
            raise ValueError('Exact continuing parent and explicit recurrent policy required')
        before = parent.state_dict()
        obj = cls.__new__(cls); obj.__dict__.update(parent.__dict__)
        m = copy.deepcopy(parent.pn_online_manifest)
        if 'orn_peripheral_terminal' in m:
            raise ValueError('ORN partition already installed')
        m['orn_peripheral_terminal'] = dict(policy=POLICY, parameters=copy.deepcopy(PARAMETERS),
            adoption_time_ns=obj.time_ns, recurrent_connected=recurrent_connected,
            source_ids=obj.brain.node_ids[obj._orn_rows].copy(), input_pairs=6242,
            recurrent_pairs=3195, other_pairs=3047, local_PN_feedback_pairs=31,
            peripheral_input='Algebraic9+83.667*clip(drive/80,0,1)Hz; amplitude means, not measured dynamics.',
            canonical_ORN_q='Effective filtered terminal activity/rmax; not a reconstructed peripheral voltage or spike train.',
            anatomy_preserved=True, historical_states_preserved=True, new_dynamic_coordinates=0,
            old_somatic_recurrent_target_replaced_once=True, parameter_fitting=False,
            native_biological_identification=False,
            scope='Bounded engineering modulator using inherited signed inputs;0.3bound and40Hz gate are provisional, not per-cell receptor measurements. All6242pairs retained; no direct PN current.')
        m['record_sha256'] = _record_hash(m); obj.pn_online_manifest = m
        obj._bind_online_routes(); obj._rebind_continuing_inputs(); obj.validate_online()
        after = obj.state_dict()
        for k in ('schema', 'pn_online_manifest'):
            before.pop(k); after.pop(k)
        if _record_hash(before) != _record_hash(after):
            raise ValueError('ORN migration changed pre-existing physical state')
        return obj

    def _bind_online_routes(self):
        super()._bind_online_routes()
        if 'orn_peripheral_terminal' not in self.pn_online_manifest:
            return
        import cupy as cp
        rows = np.asarray(self._orn_rows, dtype=np.int64)
        graph = self.brain.W
        positions = np.concatenate([np.arange(graph.indptr[r],graph.indptr[r+1]) for r in rows]).astype(np.int64)
        pres = graph.indices[positions].astype(np.int64)
        counts = np.array([graph.indptr[r+1]-graph.indptr[r] for r in rows],dtype=np.int64)
        indptr = np.r_[0,np.cumsum(counts)].astype(np.int64)
        recurrent = np.isin(pres, rows)
        post = np.repeat(rows,counts)
        pn = pres == self._online_ports.pn_row
        slots = np.full(len(pres),-1,dtype=np.int64)
        target_ids = self._online_source.general_output.targets
        slots[pn] = np.searchsorted(target_ids,self.brain.node_ids[post[pn]])
        np.testing.assert_array_equal(target_ids[slots[pn]],self.brain.node_ids[post[pn]])
        if (len(rows)!=74 or len(pres)!=6242 or recurrent.sum()!=3195 or pn.sum()!=31
                or np.isin(self.brain.node_ids[pres],[10540,10977]).any()):
            raise ValueError('Unexpected ORN input partition or regional APL dependency')
        self._orn_terminal_cpu = dict(rows=rows,indptr=indptr,pres=pres,positions=positions,
                                     pn_slot=slots,recurrent=recurrent)
        self._orn_terminal_cuda = {k:cp.asarray(v) for k,v in self._orn_terminal_cpu.items()}
        self._orn_terminal_kernel = cp.RawKernel(CUDA_CODE,'orn_terminal',options=('--fmad=false',))

    def coefficients_gpu(self,state,drive,light):
        target, rate = super().coefficients_gpu(state,drive,light)
        import cupy as cp
        d=self._orn_terminal_cuda; c=self.cuda
        # Parent restores its temporary PN edge weights before returning. Use
        # the31 actual local releases here, never the retired PN scalar.
        local=cp.asarray(self._online_source.general_transmission())
        self._orn_terminal_kernel((1,),(128,),tuple(d[k] for k in (
            'rows','indptr','pres','positions','pn_slot','recurrent')) + (
            c['weights'],c['caps'],c['gain'],state[self.transmission_start:],local,drive,
            np.bool_(self.pn_online_manifest['orn_peripheral_terminal']['recurrent_connected']),target))
        return target, rate

    def validate_online(self):
        super().validate_online()
        m=self.pn_online_manifest['orn_peripheral_terminal']
        if (m['policy']!=POLICY or m['parameters']!=PARAMETERS or type(m['recurrent_connected']) is not bool
                or not np.array_equal(m['source_ids'],self.brain.node_ids[self._orn_rows])):
            raise ValueError('Changed peripheral/terminal policy')

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:
            raise ValueError('Incomplete ORN peripheral/terminal brain')
        base=cls.PARENT.from_state(brain,dict(saved,schema=cls.PARENT.SCHEMA))
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online();return obj

    @classmethod
    def backend_identity(cls):
        out=cls.PARENT.backend_identity();out['ORN_peripheral_terminal']=POLICY;return out


class GpuOrnPeripheralTerminalBrain(OrnPeripheralTerminalMixin,GpuPnMassKrylovBrain):
    PARENT=GpuPnMassKrylovBrain
    SCHEMA='matrix_orn_peripheral_terminal_mass_brain_v1'


class GpuOrnPeripheralTerminalFineBrain(OrnPeripheralTerminalMixin,GpuPnInhibitoryClosureBrain):
    PARENT=GpuPnInhibitoryClosureBrain
    SCHEMA='matrix_orn_peripheral_terminal_fine_brain_v1'
