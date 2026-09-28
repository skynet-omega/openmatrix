"""Experimental graded DNp20/DNOVS1 on the unchanged canonical graph.

Suver 2016 Fig3C reports purely graded DNOVS1 whole-cell responses. The
canonical DNp20 alias identifies two candidate homologues, not their measured
male physiology. Reuse of generic conductances/release/tau is an explicit
hypothesis. A one-time coordinate conversion preserves release algebraically
and chemical history exactly; its FP64 rounding is recorded, not corrected by
an ongoing clamp. No electrical synapse or measured initial Vm is invented.
"""
import copy

import numpy as np

from r8_mi4_visual_brain import R8Mi4VisualBrain, R8_MI4_KEYS
from measured_t4_visual_brain import _valid_hash
from synaptic_visual_brain import _hash_array


TARGET_IDS = np.array([10059, 10162], dtype=np.int64)
TARGET_TYPE = 'DNp20'
POLICY = 'dnp20_dnovs1_graded_candidate_v1'
SOURCE_URL = 'https://doi.org/10.1523/JNEUROSCI.2277-16.2016'
SOURCE_HTML_SHA256 = '32ba15a124fdc4997cfbc6e07a0b14f5c2e9e1bf52d1daca78f9e97adbf73713'
CANONICAL_ANNOTATION_SHA256 = 'b3ed715bc6bb6a703cabc84a372ac448b9f036956ced31cdcf6f80f890d4e01a'
GRADED_DESCENDING_KEYS = R8_MI4_KEYS | {'graded_descending_manifest', 'graded_descending_migration'}
REFERENCE_SCHEMAS = {
    'matrix_hybrid_visual_brain_r8_mi4_fp64_v1',
    'matrix_hybrid_visual_brain_r8_mi4_fp64_cuda_v1',
}


def _rows(brain):
    rows = np.searchsorted(brain.node_ids, TARGET_IDS)
    if np.any(rows >= brain.n_neurons) or not np.array_equal(brain.node_ids[rows], TARGET_IDS):
        raise ValueError('Both canonical DNp20 identities 10059/10162 are required.')
    return rows.astype(np.int64)


def _vs_pairs(brain, types, rows):
    positions = []
    for row in rows:
        start, stop = brain.W.indptr[row:row+2]
        p = np.arange(start, stop, dtype=np.int64)
        positions.extend(p[types[brain.W.indices[start:stop]] == 'VS'].tolist())
    return np.asarray(positions, dtype=np.int64)


def _release(state, visual_rows, n):
    result = state[:n].copy()
    result[visual_rows] = np.clip((80.*result[visual_rows]-15.)/40., 0., 1.)
    return result


def _valid_ids(brain, value):
    if (not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.ndim != 1
            or np.any(np.diff(value) <= 0)):
        raise ValueError('Graded identity list must be a sorted unique int64 vector.')
    indices = np.searchsorted(brain.node_ids, value)
    if np.any(indices >= brain.n_neurons) or not np.array_equal(brain.node_ids[indices], value):
        raise ValueError('Unknown graded identity.')
    return indices


def _validate_record(brain, saved):
    """Validate migration before delegating the old family's coordinate codec."""
    m = saved.get('graded_descending_manifest')
    h = saved.get('graded_descending_migration')
    if (not isinstance(m, dict) or not isinstance(h, dict) or m.get('policy') != POLICY
            or m.get('source_url') != SOURCE_URL or m.get('source_html_sha256') != SOURCE_HTML_SHA256
            or m.get('canonical_annotation_sha256') != CANONICAL_ANNOTATION_SHA256
            or type(m.get('enabled')) is not bool or type(m.get('vs_input_cut')) is not bool):
        raise ValueError('Unknown graded DNp20 policy, flags or primary source.')
    rows = _rows(brain)
    types = saved['measured_t4_manifest'].get('canonical_node_types')
    if (not isinstance(types, np.ndarray) or types.shape != (brain.n_neurons,) or
            types.dtype.kind != 'U' or not np.all(types[rows] == TARGET_TYPE)):
        raise ValueError('Target identities are not canonically annotated DNp20.')
    old_ids = m.get('source_visual_ids')
    old_vi = _valid_ids(brain, old_ids)
    if np.intersect1d(old_ids, TARGET_IDS).size:
        raise ValueError('DNp20 targets must belong to the source rate population.')
    expected = np.union1d(old_ids, TARGET_IDS) if m['enabled'] else old_ids
    vi = _valid_ids(brain, saved['visual_ids'])
    if not np.array_equal(saved['visual_ids'], expected):
        raise ValueError('Graded scope differs from exactly the two DNp20 targets.')
    old_mask = np.zeros(brain.n_neurons, dtype=bool)
    old_mask[old_vi] = True
    new_mask = old_mask.copy()
    new_mask[rows] = m['enabled']
    arrays = [('node_ids_sha256', brain.node_ids), ('canonical_types_sha256', types),
              ('anatomical_indptr_sha256', brain.W.indptr), ('anatomical_indices_sha256', brain.W.indices),
              ('source_visual_ids_sha256', old_ids), ('source_visual_mask_sha256', old_mask),
              ('new_visual_ids_sha256', expected), ('new_visual_mask_sha256', new_mask)]
    if any(m.get(key) != _hash_array(array) for key, array in arrays):
        raise ValueError('Graded DNp20 manifest disagrees with anatomy, annotation or masks.')
    if not np.array_equal(m.get('target_ids'), TARGET_IDS) or not np.array_equal(m.get('target_rows'), rows):
        raise ValueError('Changed graded DNp20 identities or positions.')
    if (not np.array_equal(m.get('target_types'), types[rows]) or
            m.get('target_instances') != ['DNp20_R', 'DNp20_L'] or m.get('target_soma_sides') != ['R', 'L'] or
            m.get('canonical_types_source') != 'measured_t4_manifest.canonical_node_types'):
        raise ValueError('Invalid target type provenance.')
    p = m.get('vs_positions')
    if (not isinstance(p, np.ndarray) or p.dtype != np.int64 or p.ndim != 1
            or not np.array_equal(p, _vs_pairs(brain, types, rows))):
        raise ValueError('VS diagnostic scope must equal existing VS -> DNp20 pairs.')
    if m['vs_input_cut'] and not len(p):
        raise ValueError('Cannot request a VS input cut without an existing VS pair.')
    pre = brain.W.indices[p]
    post = np.searchsorted(brain.W.indptr, p, side='right')-1
    for key, a in [('vs_pre_ids', brain.node_ids[pre]), ('vs_post_ids', brain.node_ids[post]),
                   ('vs_pre_types', types[pre]), ('vs_post_types', types[post])]:
        if not np.array_equal(m.get(key), a):
            raise ValueError('VS diagnostic pair identity differs from its CSR position.')
    if (not np.isfinite(brain.W.data[p]).all() or
            m.get('current_vs_signed_weights_sha256') != _hash_array(brain.W.data[p]) or
            not _valid_hash(m.get('adoption_signed_weights_sha256'))):
        raise ValueError('Invalid signed weight record for graded DNp20.')
    if (type(h.get('time_ns')) is not int or type(saved.get('time_ns')) is not int
            or not 0 <= h['time_ns'] <= saved['time_ns'] or h.get('source_schema') not in REFERENCE_SCHEMAS
            or type(h.get('enabled')) is not bool or h['enabled'] != m['enabled']
            or type(h.get('vs_input_cut')) is not bool or h['vs_input_cut'] != m['vs_input_cut']):
        raise ValueError('Invalid graded DNp20 migration history.')
    for key in ('source_state_sha256', 'migrated_state_sha256', 'initial_transmission_sha256'):
        if not _valid_hash(h.get(key)):
            raise ValueError('Missing graded DNp20 migration hash.')
    for key in ('old_coordinates', 'new_coordinates', 'initial_release_before',
                'initial_release_after', 'release_roundoff', 'initial_target_transmission'):
        a = h.get(key)
        if not isinstance(a, np.ndarray) or a.dtype != np.float64 or a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError('Invalid graded DNp20 coordinate migration record.')
    old = h['old_coordinates']
    new = (15.+40.*old)/80. if m['enabled'] else old
    after = np.clip((80.*new-15.)/40., 0., 1.) if m['enabled'] else old
    if (np.any((old < 0.) | (old > 1.)) or not np.array_equal(h['new_coordinates'], new)
            or not np.array_equal(h['initial_release_before'], old)
            or not np.array_equal(h['initial_release_after'], after)
            or not np.array_equal(h['release_roundoff'], after-old)):
        raise ValueError('Migration is not the declared release-preserving affine conversion.')
    for key, values in [('source_published_view', (old*brain.r_max[rows].astype(float)).astype(np.float32)),
                        ('migrated_published_view', (after*brain.r_max[rows].astype(float)).astype(np.float32))]:
        stored = h.get(key)
        if (not isinstance(stored, np.ndarray) or stored.dtype != np.float32 or stored.shape != (2,)
                or not np.array_equal(stored, values)):
            raise ValueError('Published coordinate migration differs from its recorded release.')
    state = saved['state']
    n, photo_n = brain.n_neurons, len(saved['photo_ids'])
    if (not isinstance(state, np.ndarray) or state.dtype != np.float64
            or state.shape != (2*n+2*photo_n,) or not np.isfinite(state).all()
            or np.any((state < 0.) | (state > 1.))):
        raise ValueError('Invalid complete migrated neural state.')
    if h['time_ns'] == saved['time_ns']:
        origin = state.copy()
        origin[rows] = old
        if (not np.array_equal(state[rows], new) or _hash_array(state) != h['migrated_state_sha256']
                or _hash_array(origin) != h['source_state_sha256']
                or _hash_array(state[n+2*photo_n:]) != h['initial_transmission_sha256']
                or not np.array_equal(state[n+2*photo_n+rows], h['initial_target_transmission'])):
            raise ValueError('Adoption changed more than the declared two coordinates.')
    return rows, old_vi, vi


class GradedDescendingBrain(R8Mi4VisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_graded_descending_fp64_v1'

    @classmethod
    def adopt(cls, reference, node_types, enabled=True, vs_input_cut=False):
        """Convert two coordinates once; disabled/uncut is the exact control.

        vs_input_cut is a separate, persistent diagnostic of existing chemical
        VS inputs, not a missing electrical coupling model or an active policy.

        This replaces/consumes the source model on the same brain object.
        The source state array is unchanged, but the shared published float32
        view may change by the recorded rounding of the two converted cells;
        do not continue the old model afterward. For an independent reference,
        load a separate checkpoint as with other shared-brain adoptions.
        """
        if not isinstance(reference, R8Mi4VisualBrain) or reference.SCHEMA not in REFERENCE_SCHEMAS:
            raise ValueError('Adopt the complete R8-to-Mi4 family exactly once.')
        if type(enabled) is not bool or type(vs_input_cut) is not bool:
            raise ValueError('enabled and vs_input_cut must be explicit booleans.')
        b, rows = reference.brain, _rows(reference.brain)
        types = np.asarray(node_types)
        canonical = reference.measured_t4_manifest['canonical_node_types']
        if (types.shape != (b.n_neurons,) or types.dtype.kind not in 'US'
                or not np.array_equal(types.astype(str), canonical)):
            raise ValueError('Cell types differ from the canonical parent annotation.')
        types = canonical
        if not np.all(types[rows] == TARGET_TYPE) or np.any(reference.visual_mask[rows]):
            raise ValueError('Only the two canonical rate-mode DNp20 cells can be converted.')
        saved = reference.state_dict()
        old = reference.state[rows].copy()
        converted = (15.+40.*old)/80. if enabled else old.copy()
        before_release = reference.release()
        if (b.time_ns != reference.time_ns or
                not np.array_equal((before_release*reference.caps).astype(np.float32), b.rates)):
            raise ValueError('Source published view or clock is inconsistent before adoption.')
        old_ids = reference.visual_ids.copy()
        saved['visual_ids'] = np.union1d(old_ids, TARGET_IDS) if enabled else old_ids.copy()
        saved['state'][rows] = converted
        new_vi = np.searchsorted(b.node_ids, saved['visual_ids'])
        after_release = _release(saved['state'], new_vi, b.n_neurons)
        new_mask = reference.visual_mask.copy()
        new_mask[rows] = enabled
        p = _vs_pairs(b, types, rows)
        pre, post = b.W.indices[p], np.searchsorted(b.W.indptr, p, side='right')-1
        saved['graded_descending_manifest'] = dict(policy=POLICY, enabled=enabled, vs_input_cut=vs_input_cut,
            target_ids=TARGET_IDS.copy(), target_rows=rows.copy(), target_types=types[rows].copy(),
            target_instances=['DNp20_R', 'DNp20_L'], target_soma_sides=['R', 'L'],
            canonical_alias='Dorkenwald 2024: DNOVS1; canonical MaleCNS v10 nodes.parquet synonyms',
            canonical_types_source='measured_t4_manifest.canonical_node_types',
            canonical_types_sha256=_hash_array(types), node_ids_sha256=_hash_array(b.node_ids),
            source_visual_ids=old_ids, source_visual_ids_sha256=_hash_array(old_ids),
            new_visual_ids_sha256=_hash_array(saved['visual_ids']),
            source_visual_mask_sha256=_hash_array(reference.visual_mask), new_visual_mask_sha256=_hash_array(new_mask),
            anatomical_indptr_sha256=_hash_array(b.W.indptr), anatomical_indices_sha256=_hash_array(b.W.indices),
            adoption_signed_weights_sha256=_hash_array(b.W.data),
            vs_positions=p, vs_pre_ids=b.node_ids[pre].copy(), vs_post_ids=b.node_ids[post].copy(),
            vs_pre_types=types[pre].copy(), vs_post_types=types[post].copy(),
            current_vs_signed_weights_sha256=_hash_array(b.W.data[p]),
            source_url=SOURCE_URL, source_html_sha256=SOURCE_HTML_SHA256,
            canonical_annotation_sha256=CANONICAL_ANNOTATION_SHA256,
            evidence='Suver et al. 2016 Fig3C: whole-cell DNOVS1 responses are purely graded, unlike the small spikes in DNOVS2 and DNHS1. Canonical DNp20 aliases identify these two candidate homologues.',
            limitations='The canonical crosswalk does not measure these male cells. Generic leak/reversals, conductance per weight, tau and release curve remain uncalibrated. Electrical coupling to VS is not represented or invented; chemical VS cuts cannot remove absent gap-junction paths.',
            coordinate='For enabled targets only: y=(15+40*r)/80; Vm=-80+80*y is a new model coordinate, not a measured initial potential. Chemical transmission is unchanged.',
            cut_scope='If vs_input_cut, zero the derived cache of existing canonical VS -> DNp20 pairs only; signed W remains intact. This diagnostic is not a candidate for active continuation.',
            global_cut_limit='Inherited visual_output_cut uses the expanded graded mask; it is not a specific VS/DNp20 control. Use the explicit VS-pair diagnostic for that question.',
            preserved='All other state coordinates, chemical history, graph, signed weights, gains, numerical settings, earlier receptor policies and clock.')
        saved['graded_descending_migration'] = dict(source_schema=saved['schema'], time_ns=int(reference.time_ns),
            enabled=enabled, vs_input_cut=vs_input_cut,
            source_state_sha256=_hash_array(reference.state), migrated_state_sha256=_hash_array(saved['state']),
            initial_transmission_sha256=_hash_array(reference.transmission_release()),
            old_coordinates=old, new_coordinates=converted.copy(), initial_release_before=before_release[rows].copy(),
            initial_release_after=after_release[rows].copy(), release_roundoff=(after_release-before_release)[rows].copy(),
            initial_target_transmission=reference.transmission_release()[rows].copy(),
            source_published_view=b.rates[rows].copy(),
            migrated_published_view=(after_release[rows]*reference.caps[rows]).astype(np.float32),
            interpretation='One-time coordinate/model revision, not learning, measured basal Vm, state reset or ongoing clamp. Affine release continuity is algebraic; actual FP64 roundoff is retained explicitly.',
            source_ownership='Consumes/replaces the old model on its shared brain. Source state is retained, but its float32 published view may change at the two targets as recorded; load a separate checkpoint to continue the reference.')
        saved['schema'] = cls.SCHEMA
        # A coordinate roundoff can rarely change the float32 published view.
        # Validate with a temporary shallow view first; commit only after the
        # complete new codec succeeds, without mutating any canonical weight.
        view = copy.copy(b)
        view.rates = (after_release*reference.caps).astype(np.float32)
        obj = cls.from_state(view, saved)
        obj.brain = b
        obj.publish_rates()
        return obj

    def _build(self):
        super()._build()
        _validate_record(self.brain, self.state_dict())
        self._apply_vs_cut()

    def _apply_vs_cut(self):
        if not self.graded_descending_manifest['vs_input_cut']:
            return
        p = self.graded_descending_manifest['vs_positions']
        self.weights64[p] = 0.
        if hasattr(self, 'cuda'):
            import cupy as cp
            self.cuda['weights'][cp.asarray(p)] = 0.

    def sync_plastic_weights(self, plasticity):
        p = self.graded_descending_manifest['vs_positions']
        if not np.isfinite(self.brain.W.data[p]).all():
            raise ValueError('Nonfinite VS-pair weights cannot be hidden by a diagnostic cut.')
        super().sync_plastic_weights(plasticity)
        self._apply_vs_cut()
        self.graded_descending_manifest['current_vs_signed_weights_sha256'] = _hash_array(self.brain.W.data[p])

    def state_dict(self):
        saved = super().state_dict()
        saved['graded_descending_manifest'] = copy.deepcopy(self.graded_descending_manifest)
        p = self.graded_descending_manifest['vs_positions']
        saved['graded_descending_manifest']['current_vs_signed_weights_sha256'] = _hash_array(self.brain.W.data[p])
        saved['graded_descending_migration'] = copy.deepcopy(self.graded_descending_migration)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != GRADED_DESCENDING_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete graded descending state or wrong neural family.')
        rows, old_vi, vi = _validate_record(brain, saved)
        release = _release(saved['state'], vi, brain.n_neurons)
        if (brain.time_ns != saved['time_ns'] or
                not np.array_equal((release*brain.r_max.astype(float)).astype(np.float32), brain.rates)):
            raise ValueError('Migrated published view or clock differs from authoritative state.')
        # Validate all historical families using a virtual rate-coordinate
        # view. Their immutable adoption hashes remain meaningful even when
        # several migrations occurred at this same clock. The virtual view is
        # never advanced, published to the real brain or returned to callers.
        parent = {k: copy.deepcopy(saved[k]) for k in R8_MI4_KEYS}
        parent['schema'] = R8Mi4VisualBrain.SCHEMA
        parent['visual_ids'] = saved['graded_descending_manifest']['source_visual_ids'].copy()
        migration = saved['graded_descending_migration']
        parent['state'][rows] = (migration['old_coordinates'] if saved['time_ns'] == migration['time_ns']
                                else release[rows])
        virtual = copy.copy(brain)
        virtual.rates = (_release(parent['state'], old_vi, brain.n_neurons)*brain.r_max.astype(float)).astype(np.float32)
        base = R8Mi4VisualBrain.from_state(virtual, parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.brain = brain
        obj.state = saved['state'].copy()
        obj.visual_ids = saved['visual_ids'].copy()
        obj.graded_descending_manifest = copy.deepcopy(saved['graded_descending_manifest'])
        obj.graded_descending_migration = copy.deepcopy(saved['graded_descending_migration'])
        obj._build()
        return obj
