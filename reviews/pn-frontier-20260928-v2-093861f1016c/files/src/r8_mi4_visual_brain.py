"""Experimental postsynaptic interpretation of existing R8p/R8y -> Mi4.

Davis 2020 reports Mi4 nicotinic receptor transcripts and no detected ort or
HisCl1; Xiao 2023 demonstrates R8 histamine/ACh cotransmission. These findings
motivate net excitation as a hypothesis, not an electrical measurement of
these male synapses. Only the derived effective sign changes. Stored signed
weights, their magnitudes, all state and the earlier receptor/reversal policies
are retained. Mi4's own GABAergic outputs are unchanged.
"""
import copy

import numpy as np

from measured_t4_visual_brain import MeasuredT4VisualBrain, MEASURED_T4_KEYS, _valid_hash
from synaptic_visual_brain import _hash_array


PRE_TYPES = ('R8p', 'R8y')
POST_TYPE = 'Mi4'
POLICY = 'r8_main_retina_mi4_candidate_ach_v1'
SOURCES = (
    'https://doi.org/10.7554/eLife.50901',
    'https://doi.org/10.1038/s41586-023-06681-6',
)
R8_MI4_KEYS = MEASURED_T4_KEYS | {'r8_mi4_manifest', 'r8_mi4_migration'}
REFERENCE_SCHEMAS = {
    'matrix_hybrid_visual_brain_measured_t4_fp64_v1',
    'matrix_hybrid_visual_brain_measured_t4_fp64_cuda_v1',
}


def _selected_pairs(brain, types):
    positions = []
    for row in np.flatnonzero(types == POST_TYPE):
        start, stop = brain.W.indptr[row:row+2]
        p = np.arange(start, stop, dtype=np.int64)
        positions.extend(p[np.isin(types[brain.W.indices[start:stop]], PRE_TYPES)].tolist())
    return np.asarray(positions, dtype=np.int64)


class R8Mi4VisualBrain(MeasuredT4VisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_r8_mi4_fp64_v1'

    @classmethod
    def adopt(cls, reference, node_types, enabled=True):
        """Adopt once without a state jump; False is the unchanged control.

        Canonical types must equal those already retained by the measured-T4
        parent. Relabeling cells cannot expand this intervention's scope.
        """
        if not isinstance(reference, MeasuredT4VisualBrain) or reference.SCHEMA not in REFERENCE_SCHEMAS:
            raise ValueError('Adopt a complete measured T4 state exactly once.')
        if type(enabled) is not bool:
            raise ValueError('enabled must be an explicit Python boolean.')
        b, types = reference.brain, np.asarray(node_types)
        if types.shape != (b.n_neurons,) or types.dtype.kind not in 'US':
            raise ValueError('One canonical type string per node_id is required.')
        types = types.astype(str)
        if not np.array_equal(types, reference.measured_t4_manifest['canonical_node_types']):
            raise ValueError('Cell types differ from the adopted canonical type annotation.')
        # Preserve the parent's actual canonical dtype when hashing; Unicode
        # arrays with equal values can have different storage widths.
        types = reference.measured_t4_manifest['canonical_node_types']
        positions = _selected_pairs(b, types)
        if not len(positions):
            raise ValueError('No existing R8p/R8y -> Mi4 connection is present.')
        pres = b.W.indices[positions]
        posts = np.searchsorted(b.W.indptr, positions, side='right')-1
        manifest = dict(policy=POLICY, enabled=enabled, positions=positions,
            pre_ids=b.node_ids[pres].copy(), post_ids=b.node_ids[posts].copy(),
            pre_types=types[pres].copy(), post_types=types[posts].copy(),
            canonical_types_source='measured_t4_manifest.canonical_node_types',
            canonical_types_sha256=_hash_array(types), node_ids_sha256=_hash_array(b.node_ids),
            anatomical_indptr_sha256=_hash_array(b.W.indptr),
            anatomical_indices_sha256=_hash_array(b.W.indices),
            adoption_signed_weights_sha256=_hash_array(b.W.data[positions]),
            current_signed_weights_sha256=_hash_array(b.W.data[positions]),
            sources=list(SOURCES),
            evidence='Davis 2020 Fig7A/Fig8C-D: R8 postsynaptic partner Mi4 expresses nicotinic ACh receptor subunits with no detected ort or HisCl1 RNA. Xiao 2023 demonstrates R8 histamine/ACh cotransmission.',
            uncertainty='RNA nondetection does not establish protein absence. Net excitation of these particular male R8p/R8y -> Mi4 pairs is a hypothesis; cotransmission does not identify their receptor conductance, magnitude, reversal or time constant.',
            exclusions='R8d, R8_unclear, R7R8_unclear, all other destinations and all Mi4 outputs retain their prior interpretation.',
            efficacy='If enabled, effective weight = abs(stored signed weight) only for selected pairs; otherwise their original signed value. No magnitude fitting or new edges.',
            preserved='Prior R8-to-Mi1 interpretation and Mi9-to-T4 GluCl reversal/calibration, all graph weights, gains, state and clock.')
        saved = reference.state_dict()
        saved['r8_mi4_manifest'] = manifest
        saved['r8_mi4_migration'] = dict(source_schema=saved['schema'], enabled=enabled,
            time_ns=int(reference.time_ns), source_state_sha256=_hash_array(reference.state),
            changes='Enabled: selected postsynaptic effective sign only; derivatives may change. Disabled: unchanged measured T4 dynamics.',
            preserved='Complete neural/retinal/transmission state and clock without jump; anatomical IDs, signed weights and magnitudes; all prior receptor policies and numerical parameters. This is a model revision, not learning.')
        saved['schema'] = cls.SCHEMA
        return cls.from_state(b, saved)

    def _build(self):
        super()._build()
        self._validate_r8_mi4_manifest()
        self._apply_r8_mi4_cache()

    def _validate_r8_mi4_manifest(self):
        m, b = self.r8_mi4_manifest, self.brain
        if (not isinstance(m, dict) or m.get('policy') != POLICY or
                tuple(m.get('sources', ())) != SOURCES or type(m.get('enabled')) is not bool or
                m.get('canonical_types_source') != 'measured_t4_manifest.canonical_node_types'):
            raise ValueError('Unknown R8-to-Mi4 interpretation, enabled flag or provenance.')
        types = self.measured_t4_manifest['canonical_node_types']
        for key, array in [('node_ids_sha256', b.node_ids), ('canonical_types_sha256', types),
                           ('anatomical_indptr_sha256', b.W.indptr), ('anatomical_indices_sha256', b.W.indices)]:
            if m.get(key) != _hash_array(array):
                raise ValueError('R8-to-Mi4 manifest disagrees with canonical anatomy/types.')
        p = m.get('positions')
        if (not isinstance(p, np.ndarray) or p.dtype != np.int64 or p.ndim != 1 or not len(p)
                or not np.array_equal(p, _selected_pairs(b, types))):
            raise ValueError('Selection must contain exactly the existing R8p/R8y -> Mi4 pairs.')
        pres = b.W.indices[p]
        posts = np.searchsorted(b.W.indptr, p, side='right')-1
        for key, values in [('pre_ids', b.node_ids[pres]), ('post_ids', b.node_ids[posts]),
                            ('pre_types', types[pres]), ('post_types', types[posts])]:
            if not np.array_equal(m.get(key), values):
                raise ValueError('R8-to-Mi4 pair identity/type disagrees with its position.')
        if (not np.all(self.visual_mask[pres] & self.visual_mask[posts]) or
                np.any(b.W.data[p] > 0) or not np.isfinite(b.W.data[p]).all()):
            raise ValueError('R8-to-Mi4 interpretation requires existing inhibitory graded-cell pairs.')
        if (m.get('current_signed_weights_sha256') != _hash_array(b.W.data[p]) or
                not _valid_hash(m.get('adoption_signed_weights_sha256'))):
            raise ValueError('R8-to-Mi4 signed weight record differs from saved anatomy.')

    def _apply_r8_mi4_cache(self):
        positions = self.r8_mi4_manifest['positions']
        values = self.brain.W.data[positions].astype(np.float64)
        self.weights64[positions] = np.abs(values) if self.r8_mi4_manifest['enabled'] else values
        # Existing deterministic CPU/CUDA reductions consume this cache. No
        # kernel or canonical signed weight is modified by this interpretation.
        if hasattr(self, 'cuda'):
            import cupy as cp
            self.cuda['weights'][cp.asarray(positions)] = cp.asarray(self.weights64[positions])

    def sync_plastic_weights(self, plasticity):
        p = self.r8_mi4_manifest['positions']
        values = self.brain.W.data[p]
        if np.any(values > 0) or not np.isfinite(values).all():
            raise ValueError('Plastic updates may change magnitude, not the canonical R8 sign.')
        super().sync_plastic_weights(plasticity)
        self._apply_r8_mi4_cache()
        self.r8_mi4_manifest['current_signed_weights_sha256'] = _hash_array(values)

    def state_dict(self):
        saved = super().state_dict()
        saved['r8_mi4_manifest'] = copy.deepcopy(self.r8_mi4_manifest)
        saved['r8_mi4_manifest']['current_signed_weights_sha256'] = _hash_array(
            self.brain.W.data[self.r8_mi4_manifest['positions']])
        saved['r8_mi4_migration'] = copy.deepcopy(self.r8_mi4_migration)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != R8_MI4_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete R8-to-Mi4 state or wrong neural family.')
        parent = {k: saved[k] for k in MEASURED_T4_KEYS}
        parent['schema'] = MeasuredT4VisualBrain.SCHEMA
        base = MeasuredT4VisualBrain.from_state(brain, parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.r8_mi4_manifest = copy.deepcopy(saved['r8_mi4_manifest'])
        obj.r8_mi4_migration = copy.deepcopy(saved['r8_mi4_migration'])
        m = obj.r8_mi4_migration
        if (not isinstance(m, dict) or type(m.get('time_ns')) is not int or
                not 0 <= m['time_ns'] <= obj.time_ns or m.get('source_schema') not in REFERENCE_SCHEMAS or
                not _valid_hash(m.get('source_state_sha256')) or type(m.get('enabled')) is not bool or
                not isinstance(obj.r8_mi4_manifest, dict) or m['enabled'] != obj.r8_mi4_manifest.get('enabled')):
            raise ValueError('Invalid R8-to-Mi4 migration history.')
        if m['time_ns'] == obj.time_ns and m['source_state_sha256'] != _hash_array(obj.state):
            raise ValueError('R8-to-Mi4 adoption must preserve the complete state exactly.')
        obj._build()
        return obj
