"""Candidate postsynaptic receptor interpretation of R8p/R8y -> Mi1.

R8 cotransmission and Mi1 receptor RNA support cholinergic excitation here,
but these particular male synapses have no identified electrical calibration.
Only the effective sign of existing pairs changes. Anatomy, stored signed W,
magnitudes, all neural history, and all other pathways remain intact.
"""
import copy
import numpy as np

from synaptic_visual_brain import SynapticVisualBrain, STATE_KEYS, _hash_array

PRE_TYPES = ('R8p', 'R8y')
POST_TYPE = 'Mi1'
POLICY = 'r8_main_retina_mi1_candidate_ach_v1'
SOURCES = (
    'https://doi.org/10.7554/eLife.50901',
    'https://doi.org/10.1038/s41586-023-06681-6',
)
RECEPTOR_KEYS = STATE_KEYS | {'receptor_manifest', 'receptor_migration'}


class ReceptorVisualBrain(SynapticVisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_receptor_fp64_v1'

    @classmethod
    def adopt(cls, reference, node_types):
        if isinstance(reference, ReceptorVisualBrain):
            raise ValueError('Receptor interpretation already present.')
        if not isinstance(reference, SynapticVisualBrain):
            raise TypeError('Adopt a complete synaptic hybrid state first.')
        if reference.SCHEMA not in {
            'matrix_hybrid_visual_brain_synaptic_fp64_v1',
            'matrix_hybrid_visual_brain_synaptic_fp64_cuda_v1'}:
            raise ValueError('Unsupported source neural family.')
        types = np.asarray(node_types)
        if types.shape != (reference.brain.n_neurons,) or types.dtype.kind not in 'US':
            raise ValueError('One canonical type string per brain.node_ids entry is required.')
        b = reference.brain
        positions, pres, posts = [], [], []
        for row in np.flatnonzero(types == POST_TYPE):
            start, end = b.W.indptr[row:row+2]
            pos = np.arange(start, end, dtype=np.int64)
            pre = b.W.indices[start:end]
            chosen = np.isin(types[pre], PRE_TYPES)
            positions.extend(pos[chosen].tolist())
            pres.extend(pre[chosen].tolist())
            posts.extend([row]*int(chosen.sum()))
        positions = np.asarray(positions, dtype=np.int64)
        pres, posts = np.asarray(pres, dtype=np.int64), np.asarray(posts, dtype=np.int64)
        if len(positions) == 0:
            raise ValueError('No existing R8p/R8y -> Mi1 pair exists in this preparation.')
        if np.any(b.W.data[positions] > 0):
            raise ValueError('Existing source signs differ from the documented histamine-only model.')
        if not np.all(reference.visual_mask[pres] & reference.visual_mask[posts]):
            raise ValueError('Selected receptor intervention requires graded pre/post cells.')
        manifest = dict(policy=POLICY, positions=positions,
            pre_ids=b.node_ids[pres].copy(), post_ids=b.node_ids[posts].copy(),
            pre_types=types[pres].astype(str), post_types=types[posts].astype(str),
            node_ids_sha256=_hash_array(b.node_ids),
            canonical_types_sha256=_hash_array(types.astype(str)),
            anatomical_indptr_sha256=_hash_array(b.W.indptr),
            anatomical_indices_sha256=_hash_array(b.W.indices),
            adoption_signed_weights_sha256=_hash_array(b.W.data[positions]),
            sources=list(SOURCES),
            evidence='R8-Rh5/Rh6 cholinergic markers and Mi1 nicotinic receptor RNA, no detected ort/HisCl1 (Davis 2020 Fig8); R8 histamine/ACh cotransmission demonstrated (Xiao 2023).',
            uncertainty='Candidate net excitation of these existing R8p/R8y -> Mi1 pairs; not electrical measurement of their receptor, strength or time constant. RNA nondetection does not prove protein absence.',
            exclusions='R8d, R8_unclear, R7R8_unclear and every destination other than Mi1 are unchanged.',
            efficacy='effective weight = abs(stored signed weight) for these pairs only; no fitted magnitude')
        saved = reference.state_dict()
        saved['receptor_manifest'] = manifest
        saved['receptor_migration'] = dict(source_schema=saved['schema'],
            time_ns=int(reference.time_ns), source_state_sha256=_hash_array(reference.state),
            changes='Postsynaptic effective sign only; state and release are continuous, but derivatives may change at adoption.',
            preserved='Full neural/retinal/transmission state, clock, anatomical IDs/topology/signed weights, gains, thresholds and RNG.')
        saved['schema'] = cls.SCHEMA
        return cls.from_state(b, saved)

    def _build(self):
        super()._build()
        self._validate_manifest()
        self._apply_receptor_cache()

    def _validate_manifest(self):
        m, b = self.receptor_manifest, self.brain
        if m.get('policy') != POLICY or tuple(m.get('sources', ())) != SOURCES:
            raise ValueError('Unknown receptor interpretation/provenance.')
        for key, array in [('node_ids_sha256', b.node_ids),
                           ('anatomical_indptr_sha256', b.W.indptr),
                           ('anatomical_indices_sha256', b.W.indices)]:
            if m.get(key) != _hash_array(array):
                raise ValueError('Receptor manifest does not match the canonical graph.')
        p = m.get('positions')
        if (not isinstance(p, np.ndarray) or p.dtype != np.int64 or p.ndim != 1
                or len(p) == 0 or np.any(np.diff(p) <= 0) or p[0] < 0 or p[-1] >= b.W.nnz):
            raise ValueError('Invalid receptor pair positions.')
        pres = b.W.indices[p]
        posts = np.searchsorted(b.W.indptr, p, side='right')-1
        for key, values in [('pre_ids', b.node_ids[pres]), ('post_ids', b.node_ids[posts])]:
            if not np.array_equal(m.get(key), values):
                raise ValueError('Receptor pair IDs disagree with its stored positions.')
        if (np.shape(m.get('pre_types')) != p.shape or np.shape(m.get('post_types')) != p.shape
                or not np.isin(m['pre_types'], PRE_TYPES).all()
                or not np.all(m['post_types'] == POST_TYPE)
                or not np.all(self.visual_mask[pres] & self.visual_mask[posts])):
            raise ValueError('Receptor selection exceeds its documented cell types.')
        if np.any(b.W.data[p] > 0) or not np.isfinite(b.W.data[p]).all():
            raise ValueError('Canonical receptor-pair weights changed sign or became invalid.')

    def _apply_receptor_cache(self):
        positions = self.receptor_manifest['positions']
        self.weights64[positions] = np.abs(self.brain.W.data[positions].astype(np.float64))
        # CUDA is optional and owned by the existing GPU superclass. Update the
        # selected elements only; the authoritative brain.W is never mutated.
        if hasattr(self, 'cuda'):
            import cupy as cp
            self.cuda['weights'][cp.asarray(positions)] = cp.asarray(self.weights64[positions])

    def sync_plastic_weights(self, plasticity):
        super().sync_plastic_weights(plasticity)
        self._apply_receptor_cache()

    def state_dict(self):
        saved = super().state_dict()
        saved['receptor_manifest'] = copy.deepcopy(self.receptor_manifest)
        saved['receptor_migration'] = copy.deepcopy(self.receptor_migration)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != RECEPTOR_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete receptor state or wrong neural family.')
        # Delegate legacy kinetic validation without interpreting its history.
        kinetic = {k: saved[k] for k in STATE_KEYS}
        kinetic['schema'] = SynapticVisualBrain.SCHEMA
        base = SynapticVisualBrain.from_state(brain, kinetic)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.receptor_manifest = copy.deepcopy(saved['receptor_manifest'])
        obj.receptor_migration = copy.deepcopy(saved['receptor_migration'])
        migration = obj.receptor_migration
        if (type(migration.get('time_ns')) is not int or
                not 0 <= migration['time_ns'] <= obj.time_ns or
                migration.get('source_schema') not in {
                    'matrix_hybrid_visual_brain_synaptic_fp64_v1',
                    'matrix_hybrid_visual_brain_synaptic_fp64_cuda_v1'}):
            raise ValueError('Invalid receptor adoption provenance.')
        obj._build()
        return obj
