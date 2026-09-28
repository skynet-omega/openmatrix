"""Explicit external retinal ports at uncovered canonical L1/L2 cells.

One four-state sensor is shared by the L1/L2 targets of an optical column.
The existing 166700 neurons and their graph remain intact. Neither the sensor
nor its added histamine conductance is an identified missing biological edge.
Retinal parameters are inherited approximations; the 20 ms sensor membrane
is the mean of the previous membrane prior, not a physiological measurement.
"""
from pathlib import Path
import copy
import json

import numpy as np

from cvn7_conductance_brain import (
    Cvn7ConductanceBrain, GpuCvn7ConductanceBrain, CVN_KEYS)
from kcgamma_regional_brain import _record_hash
from retinal_phototransduction import (
    PARAMETERS as RETINAL_PARAMETERS, coefficients as retinal_coefficients,
    initial_state as retinal_initial_state, provenance as retinal_provenance)
from session_io import sha256
from synaptic_visual_brain import _hash_array


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / 'data/retinal_boundary_20260910/selection.npz'
POLICY = 'external_column_retina_to_uncovered_l1_l2_histamine_v1'
BOUNDARY_KEYS = CVN_KEYS | {'boundary_manifest', 'held_boundary_light'}
SELECTION_KEYS = {'target_rows', 'target_ids', 'target_sensor',
    'target_relative_gmax', 'sensor_rays', 'sensor_sides'}
STATE_ORDER = ['microvilli_available_fraction',
    'filtered_successful_absorption_fraction', 'normalized_voltage',
    'normalized_transmission']
SENSOR_MEMBRANE_TAU_S = .020
SENSOR_TRANSMISSION_TAU_S = .005
HISTAMINE_REVERSAL_MV = -80.
PARENT_STATE_SIZE = 355470


def parameters():
    return dict(retinal=copy.deepcopy(RETINAL_PARAMETERS),
        sensor_membrane_tau_s=SENSOR_MEMBRANE_TAU_S,
        sensor_transmission_tau_s=SENSOR_TRANSMISSION_TAU_S,
        histamine_reversal_mv=HISTAMINE_REVERSAL_MV,
        leak_mv=-60., voltage_low_mv=-80., voltage_high_mv=0.,
        release_low_mv=-65., release_high_mv=-25.)


def _light(value, n):
    value = np.asarray(value)
    if (value.dtype != np.float64 or value.shape != (n,)
            or not np.isfinite(value).all() or np.any((value < 0.) | (value > 1.))):
        raise ValueError('Boundary light requires one finite normalized float64 value per sensor')
    return value.copy()


def initial_boundary_state(light):
    """Declared new sensor history: equilibrium at the current optical image."""
    light = np.asarray(light)
    light = _light(light, len(light))
    available, bump = retinal_initial_state(light).reshape(2, -1)
    photo = RETINAL_PARAMETERS['maximum_relative_photoconductance'] * bump
    ephoto = (RETINAL_PARAMETERS['photo_reversal_mv'] + 80.) / 80.
    voltage = (.25 + ephoto * photo) / (1. + photo)
    transmission = np.clip((80. * voltage - 15.) / 40., 0., 1.)
    return np.concatenate((available, bump, voltage, transmission))


def _read_selection(path):
    path = DEFAULT_SELECTION if path is None else Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        path = path / 'selection.npz'
    path = path.resolve()
    manifest_path = path.parent / 'manifest.json'
    source = json.loads(manifest_path.read_text(encoding='utf-8'))
    digest = sha256(path)
    declared = source.get('selection_sha256', source.get('files', {}).get(path.name))
    if isinstance(declared, dict):
        declared = declared.get('sha256')
    if declared is None or declared != digest:
        raise ValueError('Retinal boundary selection checksum differs from its source manifest')
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != SELECTION_KEYS:
            raise ValueError('Unknown or incomplete retinal boundary selection arrays')
        selection = {key: arrays[key].copy() for key in arrays.files}
    receipt = dict(selection_path=str(path), selection_sha256=digest,
        manifest_path=str(manifest_path), manifest_file_sha256=sha256(manifest_path),
        source_manifest=source, source_manifest_record_sha256=_record_hash(source),
        selection_arrays_sha256=_record_hash(selection))
    return selection, receipt


def _check_selection(reference, selection):
    b = reference.brain
    if b.n_neurons != 166700 or set(selection) != SELECTION_KEYS:
        raise ValueError('Boundary ports require the complete canonical CNS and exact selection fields')
    rows, ids, sensors = (selection[key] for key in ('target_rows', 'target_ids', 'target_sensor'))
    gains, rays, sides = (selection[key] for key in ('target_relative_gmax', 'sensor_rays', 'sensor_sides'))
    if (any(not isinstance(v, np.ndarray) for v in selection.values())
            or rows.dtype != np.int64 or rows.ndim != 1 or not len(rows)
            or np.any(np.diff(rows) <= 0) or rows[0] < 0 or rows[-1] >= b.n_neurons
            or ids.dtype != np.int64 or ids.shape != rows.shape
            or not np.array_equal(ids, b.node_ids[rows])
            or sensors.dtype != np.int64 or sensors.shape != rows.shape
            or gains.dtype != np.float64 or gains.shape != rows.shape
            or not np.isfinite(gains).all() or np.any(gains <= 0.)
            or rays.dtype != np.float64 or rays.ndim != 2 or rays.shape[1] != 3
            or not len(rays) or not np.isfinite(rays).all()
            or not np.allclose(np.linalg.norm(rays, axis=1), 1., rtol=0., atol=1.e-10)
            or sides.shape != (len(rays),) or sides.dtype.kind != 'U'
            or not np.isin(sides, ['L', 'R']).all()
            or np.any(sensors < 0) or np.any(sensors >= len(rays))
            or not np.array_equal(np.unique(sensors), np.arange(len(rays)))):
        raise ValueError('Invalid boundary IDs, column sensor mapping, conductance or optical geometry')
    types = reference.measured_t4_manifest['canonical_node_types']
    if (not np.isin(types[rows], ['L1', 'L2']).all()
            or not reference.visual_mask[rows].all()
            or np.intersect1d(rows, reference.pi).size):
        raise ValueError('Only uncovered graded L1/L2 can receive external boundary conductance')
    for row in rows:
        pres = b.W.indices[b.W.indptr[row]:b.W.indptr[row + 1]]
        if np.any(types[pres] == 'R1-R6'):
            raise ValueError('An external boundary target already has a canonical R1-R6 pair')
    for name, key in (('retinal_transduction_manifest', 'target_rows'),
                      ('regional_manifest', 'target_rows'),
                      ('orn_pn_synaptic_manifest', 'post_indices'),
                      ('cvn7_manifest', 'target_rows')):
        if np.intersect1d(rows, getattr(reference, name).get(key, [])).size:
            raise ValueError('Boundary targets overlap another local component: ' + name)


def _histamine_targets(target, rate, rows, sensor_index, gmax, transmission, tau):
    """Add g_His*(E_His-V); E_His=-80 mV is normalized zero."""
    gh = gmax * transmission[sensor_index]
    old_total = rate[rows] * tau[rows]
    total = old_total + gh
    target[rows] = target[rows] * old_total / total
    rate[rows] = total / tau[rows]


class LaminaBoundaryBrain(Cvn7ConductanceBrain):
    SCHEMA = 'matrix_lamina_boundary_brain_fp64_v1'
    BOUNDARY_PARENT_CLASS = Cvn7ConductanceBrain

    @classmethod
    def adopt(cls, reference, *, initial_light, enabled=True, selection_path=None):
        if (type(enabled) is not bool or type(reference) is not cls.BOUNDARY_PARENT_CLASS
                or reference.SCHEMA != cls.BOUNDARY_PARENT_CLASS.SCHEMA
                or len(reference.state) != PARENT_STATE_SIZE):
            raise ValueError('Boundary adoption requires the exact complete CvN7 parent and a boolean connection')
        selection, receipt = _read_selection(selection_path)
        _check_selection(reference, selection)
        light = _light(initial_light, len(selection['sensor_rays']))
        saved = reference.state_dict()
        old_record = {key: value for key, value in saved.items() if key != 'schema'}
        new = initial_boundary_state(light)
        b = reference.brain
        m = dict(policy=POLICY, enabled=enabled, source_schema=reference.SCHEMA,
            parent_state_size=PARENT_STATE_SIZE, adoption_time_ns=int(reference.time_ns),
            initial_parent_state_sha256=_hash_array(reference.state),
            initial_parent_record_sha256=_record_hash(old_record),
            initial_light=light.copy(), initial_new_state_sha256=_hash_array(new),
            parameters=parameters(), state_order=STATE_ORDER.copy(), source=receipt,
            **selection,
            node_ids_sha256=_hash_array(b.node_ids),
            canonical_types_sha256=_hash_array(reference.measured_t4_manifest['canonical_node_types']),
            anatomical_indptr_sha256=_hash_array(b.W.indptr),
            anatomical_indices_sha256=_hash_array(b.W.indices),
            anatomical_weights_sha256=_hash_array(b.W.data),
            inherited_retinal_manifest_sha256=_record_hash(reference.retinal_transduction_manifest),
            retinal_provenance=retinal_provenance(),
            sensor_equation='dA/dt=(1-A)/tau_rec-(phi/N)*A; dz/dt=(L*A-z)/5ms; du/dt=((.25+1.25*gphoto)/(1+gphoto)-u)*(1+gphoto)/20ms; ds/dt=(clip((80*u-15)/40,0,1)-s)/5ms.',
            histamine_equation='gh=target_relative_gmax*s[target_sensor]; oldtotal=parent_rate*tau; target=parent_target*oldtotal/(oldtotal+gh); rate=(oldtotal+gh)/tau; EHis=-80mV.',
            conductance_status='Median anatomical contacts for the target type with exactly six canonical R1-R6 inputs, multiplied by inherited .001 relative conductance/contact. Not measured nS or reconstructed missing contacts.',
            membrane_tau_status='20ms mean of the inherited random membrane prior; no photoreceptor electrical calibration.',
            initialization='New external sensor A,z,u,s at equilibrium of current pending optical image; all pre-existing neural and body history preserved. Earlier external sensor history is unknown.',
            scope='External achromatic retinal apparatus at canonical L1/L2 without any canonical R1-R6 pair; shared sensor per column. No new brain neuron IDs or anatomical edges.',
            boundary_direction='Feedforward apparatus; no invented feedback from L1/L2 to the missing receptor.',
            bypass='When disabled, parent coefficients and error norm are unchanged; four added states per sensor are frozen.',
            new_canonical_neurons=0, new_anatomical_pairs=0,
            biological_validation=False, parameter_fitting=False)
        m['record_sha256'] = _record_hash(m)
        saved.update(schema=cls.SCHEMA, state=np.concatenate((reference.state, new)),
            boundary_manifest=m, held_boundary_light=light)
        return cls.from_state(b, saved)

    @property
    def boundary_parent_state_size(self):
        return int(self.boundary_manifest['parent_state_size'])

    @property
    def boundary_state(self):
        return self.state[self.boundary_parent_state_size:].reshape(4, -1).copy()

    @property
    def retinal_transduction_state(self):
        return self.state[self.retinal_parent_state_size:self.boundary_parent_state_size].reshape(2, -1).copy()

    def set_boundary_light(self, value):
        light = _light(value, len(self.boundary_manifest['sensor_rays']))
        self.held_boundary_light = light
        if hasattr(self, '_boundary_cuda'):
            import cupy as cp
            self._boundary_cuda['light'] = cp.asarray(light)

    def _norm_size(self):
        if self.boundary_manifest['enabled']:
            return len(self.state)
        if self.retinal_transduction_manifest['enabled']:
            return self.boundary_parent_state_size
        if self.regional_manifest['enabled']:
            return self.retinal_parent_state_size
        if self.orn_synaptic_enabled:
            return self.regional_parent_state_size
        return self.parent_state_size if self.adaptation_active else self.inherited_state_size

    def _build_boundary_cache(self):
        m, b = self.boundary_manifest, self.brain
        selection = {key: m[key] for key in SELECTION_KEYS}
        _check_selection(self, selection)
        source = m['source']
        if (m.get('policy') != POLICY or type(m.get('enabled')) is not bool
                or m.get('source_schema') != self.BOUNDARY_PARENT_CLASS.SCHEMA
                or m.get('parent_state_size') != PARENT_STATE_SIZE
                or m.get('parameters') != parameters() or m.get('state_order') != STATE_ORDER
                or m.get('record_sha256') != _record_hash(m)
                or m.get('biological_validation') is not False
                or m.get('parameter_fitting') is not False
                or m.get('new_canonical_neurons') != 0 or m.get('new_anatomical_pairs') != 0
                or m.get('retinal_provenance') != retinal_provenance()
                or source.get('selection_arrays_sha256') != _record_hash(selection)
                or source.get('source_manifest_record_sha256') != _record_hash(source['source_manifest'])
                or type(m.get('adoption_time_ns')) is not int
                or not 0 <= m['adoption_time_ns'] <= self.time_ns):
            raise ValueError('Changed external retinal mechanism, parameter or source contract')
        for key, value in [('node_ids', b.node_ids),
                           ('canonical_types', self.measured_t4_manifest['canonical_node_types']),
                           ('anatomical_indptr', b.W.indptr), ('anatomical_indices', b.W.indices),
                           ('anatomical_weights', b.W.data)]:
            if m.get(key + '_sha256') != _hash_array(value):
                raise ValueError('External retinal selection differs from canonical data: ' + key)
        if m['inherited_retinal_manifest_sha256'] != _record_hash(self.retinal_transduction_manifest):
            raise ValueError('External apparatus changed the inherited retinal component')
        n = len(selection['sensor_rays'])
        if (self.state.dtype != np.float64 or self.state.shape != (PARENT_STATE_SIZE + 4 * n,)
                or not np.isfinite(self.state).all() or np.any((self.state < 0.) | (self.state > 1.))):
            raise ValueError('Invalid extended retinal boundary state')
        self.held_boundary_light = _light(self.held_boundary_light, n)
        initial_light = _light(m['initial_light'], n)
        if self.time_ns == m['adoption_time_ns']:
            initial = initial_boundary_state(initial_light)
            parent = {key: value for key, value in self.state_dict().items()
                      if key not in ('schema', 'boundary_manifest', 'held_boundary_light')}
            parent['state'] = self.state[:PARENT_STATE_SIZE].copy()
            if (m['initial_parent_state_sha256'] != _hash_array(parent['state'])
                    or m['initial_parent_record_sha256'] != _record_hash(parent)
                    or m['initial_new_state_sha256'] != _hash_array(initial)
                    or not np.array_equal(self.state[PARENT_STATE_SIZE:], initial)):
                raise ValueError('Boundary adoption changed inherited state or its initial sensor history')
        self._boundary_rows = m['target_rows']
        self._boundary_sensor = m['target_sensor']
        self._boundary_gmax = m['target_relative_gmax']

    def _coefficients(self, state, drive, light):
        size = self.boundary_parent_state_size
        target, rate = Cvn7ConductanceBrain._coefficients(self, state[:size], drive, light)
        if not self.boundary_manifest['enabled']:
            return np.concatenate((target, state[size:])), np.concatenate((rate, np.zeros(len(state) - size)))
        available, bump, voltage, transmission = state[size:].reshape(4, -1)
        at, ar, bt = retinal_coefficients(available, self.held_boundary_light)
        photo = RETINAL_PARAMETERS['maximum_relative_photoconductance'] * bump
        total = 1. + photo
        ephoto = (RETINAL_PARAMETERS['photo_reversal_mv'] + 80.) / 80.
        ut = (.25 + ephoto * photo) / total
        st = np.clip((80. * voltage - 15.) / 40., 0., 1.)
        _histamine_targets(target, rate, self._boundary_rows, self._boundary_sensor,
                           self._boundary_gmax, transmission, self.tau)
        n = len(available)
        return (np.concatenate((target, at, bt, ut, st)),
                np.concatenate((rate, ar, np.full(n, 1. / RETINAL_PARAMETERS['bump_filter_tau_s']),
                    total / SENSOR_MEMBRANE_TAU_S, np.full(n, 1. / SENSOR_TRANSMISSION_TAU_S))))

    def state_dict(self):
        saved = super().state_dict()
        saved['boundary_manifest'] = copy.deepcopy(self.boundary_manifest)
        saved['held_boundary_light'] = self.held_boundary_light.copy()
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != BOUNDARY_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete external retinal boundary state or wrong backend')
        m, state = saved['boundary_manifest'], saved['state']
        if (not isinstance(m, dict) or m.get('parent_state_size') != PARENT_STATE_SIZE
                or not isinstance(state, np.ndarray) or state.dtype != np.float64
                or state.ndim != 1 or len(state) <= PARENT_STATE_SIZE):
            raise ValueError('Invalid boundary state layout')
        parent = {key: value for key, value in saved.items()
                  if key not in ('boundary_manifest', 'held_boundary_light')}
        parent.update(schema=cls.BOUNDARY_PARENT_CLASS.SCHEMA,
                      state=state[:PARENT_STATE_SIZE].copy())
        base = cls.BOUNDARY_PARENT_CLASS.from_state(brain, parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.state = state.copy()
        obj.boundary_manifest = copy.deepcopy(m)
        obj.held_boundary_light = copy.deepcopy(saved['held_boundary_light'])
        obj._build_boundary_cache()
        return obj


class GpuLaminaBoundaryBrain(LaminaBoundaryBrain, GpuCvn7ConductanceBrain):
    SCHEMA = 'matrix_lamina_boundary_brain_fp64_cuda_v1'
    BOUNDARY_PARENT_CLASS = GpuCvn7ConductanceBrain

    def _build_boundary_cache(self):
        super()._build_boundary_cache()
        import cupy as cp
        self._boundary_cuda = dict(rows=cp.asarray(self._boundary_rows),
            sensor=cp.asarray(self._boundary_sensor), gmax=cp.asarray(self._boundary_gmax),
            light=cp.asarray(self.held_boundary_light))

    def coefficients_gpu(self, state, drive, light):
        import cupy as cp
        size = self.boundary_parent_state_size
        target, rate = GpuCvn7ConductanceBrain.coefficients_gpu(self, state[:size], drive, light)
        if not self.boundary_manifest['enabled']:
            return cp.concatenate((target, state[size:])), cp.concatenate((rate, cp.zeros(len(state) - size)))
        available, bump, voltage, transmission = state[size:].reshape(4, -1)
        c = self._boundary_cuda
        at, ar, bt = retinal_coefficients(available, c['light'])
        photo = RETINAL_PARAMETERS['maximum_relative_photoconductance'] * bump
        total = 1. + photo
        ephoto = (RETINAL_PARAMETERS['photo_reversal_mv'] + 80.) / 80.
        ut = (.25 + ephoto * photo) / total
        st = cp.clip((80. * voltage - 15.) / 40., 0., 1.)
        _histamine_targets(target, rate, c['rows'], c['sensor'], c['gmax'], transmission, self.cuda['tau'])
        n = len(available)
        return (cp.concatenate((target, at, bt, ut, st)),
                cp.concatenate((rate, ar, cp.full(n, 1. / RETINAL_PARAMETERS['bump_filter_tau_s']),
                    total / SENSOR_MEMBRANE_TAU_S, cp.full(n, 1. / SENSOR_TRANSMISSION_TAU_S))))

    @staticmethod
    def backend_identity():
        result = GpuCvn7ConductanceBrain.backend_identity()
        result['lamina_boundary'] = ('External four-state column sensors feed histamine conductance to uncovered canonical L1/L2 rows; '
            'FP64 stage-consistent coefficients; no added graph pairs or canonical neurons; uncalibrated retinal/membrane priors.')
        return result
