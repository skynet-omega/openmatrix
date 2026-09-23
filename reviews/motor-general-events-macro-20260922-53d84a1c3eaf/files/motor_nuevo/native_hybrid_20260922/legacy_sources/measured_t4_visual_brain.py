"""Experimental Mi9 -> T4 GluCl reversal constrained by somatic I-V.

Only the reversal of existing inhibitory Mi9 -> T4a/b/c/d conductances
changes. Stored weights, total conductance, graph and every state are retained.
The training estimate is an uncertain somatic constraint, not a validated
synaptic constant or an identification of conductance per anatomical contact.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numba
import numpy as np

from receptor_visual_brain import ReceptorVisualBrain, RECEPTOR_KEYS
from synaptic_visual_brain import _hash_array


PRE_TYPE = 'Mi9'
POST_TYPES = ('T4a', 'T4b', 'T4c', 'T4d')
POLICY = 'mi9_t4_glucl_somatic_iv_reversal_candidate_v1'
CALIBRATION_SCHEMA = 'matrix_t4_glucl_iv_calibration_v1'
ARTICLE_DOI = '10.1038/s41586-022-04428-3'
DATASET_DOI = '10.17617/3.8G'
PROTEIN_SOURCE_DOI = '10.1016/j.neuron.2023.12.014'
MEASURED_T4_KEYS = RECEPTOR_KEYS | {'measured_t4_manifest', 'measured_t4_migration'}
REFERENCE_SCHEMAS = {
    'matrix_hybrid_visual_brain_receptor_fp64_v1',
    'matrix_hybrid_visual_brain_receptor_fp64_cuda_v1',
}


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record_hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _validate_calibration(calibration):
    if not isinstance(calibration, dict) or calibration.get('schema') != CALIBRATION_SCHEMA:
        raise ValueError('An explicit measured T4 calibration record is required.')
    for key in ('E_GluCl_mV', 'train_E_GluCl_mV'):
        value = calibration.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not -80. <= value <= 0.:
            raise ValueError('GluCl reversal must be finite and within [-80, 0] mV.')
    if (calibration.get('reference_E_GluCl_mV') != -80. or
            calibration.get('source_article_doi') != ARTICLE_DOI or
            calibration.get('source_dataset_doi') != DATASET_DOI or
            calibration.get('fit_voltage_range_mV') != [-100., -40.] or
            calibration.get('estimator') != 'negative_mean_intercept_pA_over_mean_slope_nS'):
        raise ValueError('Unknown electrical reference, fitting protocol or source.')
    hashes = calibration.get('hashes')
    expected_hashes = {'target_manifest', 'iv_cells', 'cell_splits', 'electrical_targets', 'extractor', 'source_iv'}
    if not isinstance(hashes, dict) or set(hashes) != expected_hashes or not all(_valid_hash(v) for v in hashes.values()):
        raise ValueError('Incomplete calibration hashes.')
    train, heldout = calibration.get('training_fits'), calibration.get('heldout_cell_ids')
    if not isinstance(train, list) or not train or not isinstance(heldout, list):
        raise ValueError('Calibration must retain its training cells and heldout identities.')
    ids, slopes, offsets = [], [], []
    for row in train:
        if not isinstance(row, dict) or set(row) != {'cell_id', 'slope_nS', 'intercept_pA'}:
            raise ValueError('Invalid training I-V record.')
        if not isinstance(row['cell_id'], str) or not row['cell_id']:
            raise ValueError('Training cell identity is absent.')
        for key in ('slope_nS', 'intercept_pA'):
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not np.isfinite(row[key]):
                raise ValueError('Nonfinite or nonnumeric I-V fit.')
        if row['slope_nS'] <= 0.:
            raise ValueError('The selected glutamate conductance must have positive fitted slope.')
        ids.append(row['cell_id'])
        slopes.append(row['slope_nS'])
        offsets.append(row['intercept_pA'])
    if (len(set(ids)) != len(ids) or any(not isinstance(c, str) or not c for c in heldout)
            or len(set(heldout)) != len(heldout) or set(ids) & set(heldout)):
        raise ValueError('Training and heldout identities must be unique and disjoint.')
    estimate = float(-np.mean(offsets)/np.mean(slopes))
    if calibration['train_E_GluCl_mV'] != estimate:
        raise ValueError('Training reversal differs from its embedded I-V fits.')
    value = calibration['E_GluCl_mV']
    expected_selection = 'reference' if value == -80. else 'training_estimate' if value == estimate else 'declared_sensitivity'
    if calibration.get('selection') != expected_selection:
        raise ValueError('Reversal selection status is inconsistent.')
    if type(calibration.get('split_seed')) is not int:
        raise ValueError('Cell partition seed is missing.')
    return float(value)


def load_calibration(target_directory, E_GluCl_mV=None):
    """Read small, hashed target products; no heldout amplitudes enter the fit.

    Omit E_GluCl_mV for the TRAIN equal-cell mean I-V estimate; -80 reproduces
    the reference. Other values in [-80,0] are declared sensitivity candidates.
    Restored states carry this record and do not depend on external files.
    """
    root = Path(target_directory)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    names = ('fig2_iv_cells.csv', 'cell_splits.json', 'fig2_electrical_targets.json')
    for name in names:
        if _file_hash(root/name) != manifest['files'][name]:
            raise ValueError('Electrical target checksum differs: '+name)
    if (manifest.get('schema') != 'matrix_visual_physiology_targets_v1' or
            manifest.get('source_article_doi') != ARTICLE_DOI or
            manifest.get('source_dataset_doi') != DATASET_DOI):
        raise ValueError('Unsupported target source.')
    splits = json.loads((root/'cell_splits.json').read_text(encoding='utf-8'))
    targets = json.loads((root/'fig2_electrical_targets.json').read_text(encoding='utf-8'))
    with (root/'fig2_iv_cells.csv').open(encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['transmitter'] == 'Glu']
    for row in rows:
        if row['role'] != splits['fig2'][row['cell_id']]:
            raise ValueError('I-V condition disagrees with cell split.')
    train = [dict(cell_id=r['cell_id'], slope_nS=float(r['slope_nS']),
                  intercept_pA=float(r['intercept_pA'])) for r in rows if r['role'] == 'calibration']
    heldout = [r['cell_id'] for r in rows if r['role'] == 'heldout']
    estimate = float(-np.mean([r['intercept_pA'] for r in train])/np.mean([r['slope_nS'] for r in train]))
    if estimate != targets['iv']['Glu__calibration']['equal_cell_mean_curve_reversal_mv']:
        raise ValueError('Training target and I-V records disagree.')
    if E_GluCl_mV is not None and (isinstance(E_GluCl_mV, bool) or not isinstance(E_GluCl_mV, (int, float))):
        raise ValueError('Use a numeric reversal in mV.')
    value = estimate if E_GluCl_mV is None else float(E_GluCl_mV)
    record = dict(schema=CALIBRATION_SCHEMA, E_GluCl_mV=value,
        train_E_GluCl_mV=estimate, reference_E_GluCl_mV=-80.,
        selection='reference' if value == -80. else 'training_estimate' if value == estimate else 'declared_sensitivity',
        source_article_doi=ARTICLE_DOI, source_dataset_doi=DATASET_DOI,
        source_iv_url=manifest['inputs']['fig2fedfig4b_iv.csv']['url'],
        split_seed=splits['seed'], training_fits=train, heldout_cell_ids=heldout,
        fit_voltage_range_mV=[-100., -40.], estimator='negative_mean_intercept_pA_over_mean_slope_nS',
        hashes=dict(target_manifest=_file_hash(root/'manifest.json'),
            iv_cells=_file_hash(root/names[0]), cell_splits=_file_hash(root/names[1]),
            electrical_targets=_file_hash(root/names[2]), extractor=manifest['extractor_sha256'],
            source_iv=manifest['inputs']['fig2fedfig4b_iv.csv']['sha256']),
        interpretation='Somatic T4 glutamate I-V constraint applied as an uncertain Mi9-to-T4 GluCl reversal; not a measured synaptic constant, per-edge strength, or global visual calibration.')
    _validate_calibration(record)
    return record


def _selected_pairs(brain, types):
    positions = []
    for row in np.flatnonzero(np.isin(types, POST_TYPES)):
        start, stop = brain.W.indptr[row:row+2]
        p = np.arange(start, stop, dtype=np.int64)
        positions.extend(p[types[brain.W.indices[start:stop]] == PRE_TYPE].tolist())
    return np.asarray(positions, dtype=np.int64)


@numba.njit(parallel=True, fastmath=False, cache=True)
def _correct_targets(rows, ptr, positions, pres, weights, transmission, scale, alpha, tau, target, rate):
    for i in numba.prange(len(rows)):
        g = 0.
        for e in range(ptr[i], ptr[i+1]):
            g -= (weights[positions[e]]*scale)*transmission[pres[e]]
        row = rows[i]
        target[row] += alpha*g/(rate[row]*tau[row])


class MeasuredT4VisualBrain(ReceptorVisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_measured_t4_fp64_v1'

    @classmethod
    def adopt(cls, reference, node_types, calibration):
        if not isinstance(reference, ReceptorVisualBrain) or reference.SCHEMA not in REFERENCE_SCHEMAS:
            raise ValueError('Adopt a complete, unmodified receptor-family state exactly once.')
        _validate_calibration(calibration)
        b, types = reference.brain, np.asarray(node_types)
        if types.shape != (b.n_neurons,) or types.dtype.kind not in 'US':
            raise ValueError('One canonical type string per node_id is required.')
        types = types.astype(str)
        positions = _selected_pairs(b, types)
        if not len(positions):
            raise ValueError('No existing Mi9 -> T4a/b/c/d connection is present.')
        pres = b.W.indices[positions]
        posts = np.searchsorted(b.W.indptr, positions, side='right')-1
        manifest = dict(policy=POLICY, positions=positions,
            receptor_localization_source_doi=PROTEIN_SOURCE_DOI,
            receptor_localization_evidence='Sanfilippo et al. Fig6: tagged GluClalpha localizes postsynaptically at Mi9 contacts onto distal T4 dendrites. This establishes a receptor-associated connection type, not its reversal in this male preparation.',
            pre_ids=b.node_ids[pres].copy(), post_ids=b.node_ids[posts].copy(),
            pre_types=types[pres].copy(), post_types=types[posts].copy(),
            canonical_node_types=types.copy(), canonical_types_sha256=_hash_array(types),
            node_ids_sha256=_hash_array(b.node_ids),
            anatomical_indptr_sha256=_hash_array(b.W.indptr),
            anatomical_indices_sha256=_hash_array(b.W.indices),
            adoption_signed_weights_sha256=_hash_array(b.W.data[positions]),
            current_signed_weights_sha256=_hash_array(b.W.data[positions]),
            calibration=copy.deepcopy(calibration), calibration_sha256=_record_hash(calibration),
            equation='Add ((E_GluCl_mV+80)/80)*g_Mi9 to the normalized voltage numerator; total conductance and rate unchanged.',
            scope='Existing Mi9 -> graded T4a/b/c/d pairs only; no new edges, sign changes, gain fitting, input currents or direction decoder.')
        saved = reference.state_dict()
        saved['measured_t4_manifest'] = manifest
        saved['measured_t4_migration'] = dict(source_schema=saved['schema'], time_ns=int(reference.time_ns),
            source_state_sha256=_hash_array(reference.state),
            preserved='All neural, retinal, transmission states, release, clock, weights, graph, gains and numerical parameters.',
            changes='Only the derivative of selected T4 cells may change at adoption; no state jump.')
        saved['schema'] = cls.SCHEMA
        return cls.from_state(b, saved)

    @property
    def E_GluCl_mV(self):
        return float(self.measured_t4_manifest['calibration']['E_GluCl_mV'])

    def _build(self):
        super()._build()
        self._validate_measured_manifest()
        p = self.measured_t4_manifest['positions']
        posts = np.searchsorted(self.brain.W.indptr, p, side='right')-1
        self._mi9_rows, counts = np.unique(posts, return_counts=True)
        self._mi9_rows = self._mi9_rows.astype(np.int64)
        self._mi9_indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self._mi9_positions = p.copy()
        self._mi9_pres = self.brain.W.indices[p].astype(np.int64)
        self._mi9_alpha = (self.E_GluCl_mV+80.)/80.

    def _validate_measured_manifest(self):
        m, b = self.measured_t4_manifest, self.brain
        if (not isinstance(m, dict) or m.get('policy') != POLICY or
                m.get('receptor_localization_source_doi') != PROTEIN_SOURCE_DOI):
            raise ValueError('Unknown measured T4 policy.')
        types = m.get('canonical_node_types')
        if not isinstance(types, np.ndarray) or types.shape != (b.n_neurons,) or types.dtype.kind != 'U':
            raise ValueError('Stored canonical cell types are incomplete.')
        for key, array in [('node_ids_sha256', b.node_ids), ('canonical_types_sha256', types),
                           ('anatomical_indptr_sha256', b.W.indptr), ('anatomical_indices_sha256', b.W.indices)]:
            if m.get(key) != _hash_array(array):
                raise ValueError('Measured T4 manifest disagrees with canonical anatomy/types.')
        p = m.get('positions')
        if (not isinstance(p, np.ndarray) or p.dtype != np.int64 or p.ndim != 1 or not len(p)
                or not np.array_equal(p, _selected_pairs(b, types))):
            raise ValueError('Measured T4 selection must contain exactly the existing Mi9 -> T4 pairs.')
        pres = b.W.indices[p]
        posts = np.searchsorted(b.W.indptr, p, side='right')-1
        for key, values in [('pre_ids', b.node_ids[pres]), ('post_ids', b.node_ids[posts]),
                            ('pre_types', types[pres]), ('post_types', types[posts])]:
            if not np.array_equal(m.get(key), values):
                raise ValueError('Measured T4 pair identity/type disagrees with its position.')
        if (not np.all(self.visual_mask[pres] & self.visual_mask[posts]) or
                np.any(b.W.data[p] > 0) or not np.isfinite(b.W.data[p]).all()):
            raise ValueError('Mi9 -> T4 reversal requires existing inhibitory graded-cell pairs.')
        if (m.get('current_signed_weights_sha256') != _hash_array(b.W.data[p]) or
                not _valid_hash(m.get('adoption_signed_weights_sha256'))):
            raise ValueError('Measured T4 signed weight record differs from saved anatomy.')
        _validate_calibration(m.get('calibration'))
        if m.get('calibration_sha256') != _record_hash(m['calibration']):
            raise ValueError('Electrical calibration record checksum differs.')

    def _coefficients(self, state, drive, light):
        target, rate = super()._coefficients(state, drive, light)
        if self._mi9_alpha != 0.:
            _correct_targets(self._mi9_rows, self._mi9_indptr, self._mi9_positions,
                self._mi9_pres, self.weights64, state[self.transmission_start:],
                self.parameters['conductance_per_stored_weight'], self._mi9_alpha,
                self.tau, target, rate)
        return target, rate

    def sync_plastic_weights(self, plasticity):
        p = self.measured_t4_manifest['positions']
        values = self.brain.W.data[p]
        if np.any(values > 0) or not np.isfinite(values).all():
            raise ValueError('Plastic updates may change magnitude, not Mi9 receptor sign.')
        super().sync_plastic_weights(plasticity)
        self.measured_t4_manifest['current_signed_weights_sha256'] = _hash_array(values)

    def state_dict(self):
        saved = super().state_dict()
        saved['measured_t4_manifest'] = copy.deepcopy(self.measured_t4_manifest)
        saved['measured_t4_manifest']['current_signed_weights_sha256'] = _hash_array(
            self.brain.W.data[self.measured_t4_manifest['positions']])
        saved['measured_t4_migration'] = copy.deepcopy(self.measured_t4_migration)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != MEASURED_T4_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete measured T4 state or wrong neural family.')
        parent = {k: saved[k] for k in RECEPTOR_KEYS}
        parent['schema'] = ReceptorVisualBrain.SCHEMA
        base = ReceptorVisualBrain.from_state(brain, parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.measured_t4_manifest = copy.deepcopy(saved['measured_t4_manifest'])
        obj.measured_t4_migration = copy.deepcopy(saved['measured_t4_migration'])
        m = obj.measured_t4_migration
        if (not isinstance(m, dict) or type(m.get('time_ns')) is not int or
                not 0 <= m['time_ns'] <= obj.time_ns or m.get('source_schema') not in REFERENCE_SCHEMAS
                or not _valid_hash(m.get('source_state_sha256'))):
            raise ValueError('Invalid measured T4 migration history.')
        if m['time_ns'] == obj.time_ns and m['source_state_sha256'] != _hash_array(obj.state):
            raise ValueError('Measured T4 adoption must preserve the complete state exactly.')
        obj._build()
        return obj
