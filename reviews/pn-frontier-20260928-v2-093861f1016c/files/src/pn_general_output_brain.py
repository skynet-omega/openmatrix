"""Replace629 general PN edge sources, retaining anatomical W and PN diagnostic.

The temporary buffer spans all coefficient consumers, including three gamma
recomputations. Dynamic KC/APL receivers use their existing local ports.
"""
import copy
import numpy as np
from pn_electrical_refined_session import GpuPnElectricalRefinedBrain
from pn_general_output_source import GeneralOutputSource
from pn_cholinergic_cns_brain import KEYS
from kcgamma_regional_brain import _record_hash


class GpuPnGeneralOutputBrain(GpuPnElectricalRefinedBrain):
    SCHEMA = 'matrix_pn_general_output_brain_v1'

    @classmethod
    def adopt(cls, parent, spec, *, enabled=True, coupling_ns=None):
        if type(parent) is not GpuPnElectricalRefinedBrain or type(enabled) is not bool:
            raise ValueError('Exact refined PN parent and explicit connection required')
        before = parent.state_dict()
        obj = cls.__new__(cls); obj.__dict__.update(parent.__dict__)
        obj._online_source = GeneralOutputSource.adopt(parent._online_source, spec)
        m = copy.deepcopy(parent.pn_online_manifest)
        previous_coupling = m['coupling_ns']
        if coupling_ns is not None:
            if type(coupling_ns) is not int or coupling_ns not in (31250, 15625):
                raise ValueError('Declared general-output comparison coupling required')
            m['coupling_ns'] = coupling_ns
            m['electrical_outputs']['temporal_refinement']['coupling_ns'] = coupling_ns
        m['general_outputs'] = dict(spec=copy.deepcopy(spec), enabled=enabled,
            adoption_time_ns=obj.time_ns, new_pairs=629, total_pairs=1095, previous_coupling_ns=previous_coupling,
            scope='Local Ca-site release normalized to inherited general rate interface; amplitude transfer provisional. PNq remains a legacy diagnostic.202inputs absent.')
        m['scope'] = 'Live291inputs to finePN/Ca;466electrical outputs and629general consumers use local release when enabled.202inputs absent; PNq retained as legacy diagnostic. Transfer scales provisional.'
        m['record_sha256'] = _record_hash(m); obj.pn_online_manifest = m
        obj._bind_online_routes(); obj._rebind_continuing_inputs(); obj.validate_online()
        after = obj.state_dict()
        for field in ('schema', 'pn_online_manifest', 'pn_online_state'):
            before.pop(field); after.pop(field)
        if _record_hash(before) != _record_hash(after):
            raise ValueError('General output adoption changed inherited CNS state')
        return obj

    def _bind_online_routes(self):
        super()._bind_online_routes()
        if not isinstance(self._online_source, GeneralOutputSource):
            return
        import cupy as cp
        if not self.kc_apl_dynamic_manifest['enabled'] or not self.pn_online_manifest['electrical_outputs']['enabled']:
            raise ValueError('All466 dynamic PN receivers must remain connected')
        port = self._online_source.general_output; graph = self.brain.W
        rows = np.searchsorted(self.brain.node_ids, port.targets)
        np.testing.assert_array_equal(self.brain.node_ids[rows], port.targets)
        if len(rows) != 629 or np.intersect1d(rows, self._dynamic_cache['rows']).size:
            raise ValueError('Exact629 nondynamic general targets required')
        pn = self._online_ports.pn_row; positions = []
        for row in rows:
            candidates = np.arange(graph.indptr[row], graph.indptr[row+1])
            edge = candidates[graph.indices[candidates] == pn]
            if len(edge) != 1:
                raise ValueError('Unique anatomical PN edge required')
            positions.append(edge[0])
        positions = np.asarray(positions, dtype=np.int64)
        if (np.any(self.weights64[positions] <= 0)
                or np.intersect1d(positions, self._apl_positions).size
                or self.visual_mask[pn]):
            raise ValueError('Unexpected inhibitory, regional APL or visual PN edge')
        all_positions = np.flatnonzero(graph.indices == pn)
        remaining = np.setdiff1d(all_positions, positions)
        excluded_rows = np.searchsorted(graph.indptr, remaining, side='right') - 1
        if len(all_positions) != 1095 or not np.isin(excluded_rows, self._dynamic_cache['rows']).all():
            raise ValueError('Incomplete partition of all1095 PN consumers')
        self._general_rows = rows
        self._general_positions = cp.asarray(positions)
        self._general_buffer_active = False

    def coefficients_gpu(self, state, drive, light):
        if not hasattr(self, '_general_positions') or not self.pn_online_manifest['general_outputs']['enabled']:
            return super().coefficients_gpu(state, drive, light)
        if self._general_buffer_active:
            raise RuntimeError('General PN edge buffer cannot be reentered')
        import cupy as cp
        pn = self._online_ports.pn_row; transmission = self.transmission_start + pn
        weights = self.cuda['weights']; positions = self._general_positions
        old = weights[positions].copy()
        view = state.copy(); view[transmission] = 1.
        self._general_buffer_active = True
        try:
            weights[positions] = old * cp.asarray(self._online_source.general_transmission())
            target, rate = super().coefficients_gpu(view, drive, light)
            # Diagnostic history remains governed by its former equation and
            # is never substituted for local release on these outgoing edges.
            target[transmission] = state[pn]
            rate[transmission] = 1. / self.parameters['synaptic_tau_s']
            return target, rate
        finally:
            weights[positions] = old
            self._general_buffer_active = False

    def validate_online(self):
        super().validate_online()
        if isinstance(self._online_source, GeneralOutputSource):
            self._online_source.general_transmission()
            if self._online_source.general_output.origin_ns != self.pn_online_manifest['general_outputs']['adoption_time_ns']:
                raise ValueError('Changed general output activation origin')

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != KEYS or saved['schema'] != cls.SCHEMA:
            raise ValueError('Incomplete general-output state')
        parent = dict(saved); parent['schema'] = GpuPnElectricalRefinedBrain.SCHEMA
        parent['pn_online_state'] = saved['pn_online_state']['base']
        base = GpuPnElectricalRefinedBrain.from_state(brain, parent)
        obj = cls.__new__(cls); obj.__dict__.update(base.__dict__)
        obj._online_source = GeneralOutputSource.adopt(base._online_source, obj.pn_online_manifest['general_outputs']['spec'])
        obj._online_source.load_state_dict(saved['pn_online_state'])
        obj._bind_online_routes(); obj._rebind_continuing_inputs(); obj.validate_online()
        return obj

    @staticmethod
    def backend_identity():
        out = GpuPnElectricalRefinedBrain.backend_identity()
        out['general_PN_outputs'] = '629 general local-release interfaces, temporary coefficient buffer including specialized recomputations; fixed anatomical W. PNq diagnostic remains legacy,202inputs absent.'
        return out
