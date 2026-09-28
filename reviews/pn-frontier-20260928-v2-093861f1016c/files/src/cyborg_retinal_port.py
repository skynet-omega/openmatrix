"""Local engineering luminance-to-voltage ports on mapped R1-R6 identities.

Each port sees only its own existing optical input. A declared 5 ms first-order
voltage device replaces that row's phototransduction and recurrent drive. The
old phototransduction state remains stored and evolves, but no longer drives
these selected voltage rows while the device is enabled. Chemical transmission
and every other neural equation remain inherited. This is NOT fly physiology.
"""
import copy
import hashlib
import json

import numpy as np

from graded_descending_brain import GradedDescendingBrain, GRADED_DESCENDING_KEYS
from synaptic_visual_brain import _hash_array


POLICY = 'cyborg_local_mapped_r1_r6_voltage_port_v1'
RETINAL_PORT_KEYS = GRADED_DESCENDING_KEYS | {'cyborg_retinal_port_manifest'}
PARENT_SCHEMAS = {
    'matrix_hybrid_visual_brain_graded_descending_fp64_v1',
    'matrix_hybrid_visual_brain_graded_descending_fp64_cuda_v1',
}
DEVICE_PARAMETERS = dict(tau_s=.005, luminance_low=0., luminance_high=1.,
    voltage_numerator_offset=15., voltage_numerator_scale=40., voltage_denominator=80.)


def _record_hash(record):
    encoded = {key: {'array_sha256': _hash_array(value)} if isinstance(value, np.ndarray) else value
               for key,value in record.items() if key != 'record_sha256'}
    return hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _selection(brain, photo_ids, canonical_types, mapped_ids):
    raw = np.asarray(mapped_ids)
    if raw.ndim != 1 or raw.dtype.kind not in 'iu' or not len(raw):
        raise ValueError('Optical mapping must provide a nonempty integer ID vector.')
    if raw.dtype.kind == 'u' and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError('Mapped identity does not fit canonical int64 IDs.')
    mapped = np.sort(raw.astype(np.int64))
    if np.any(np.diff(mapped) <= 0):
        raise ValueError('Optical mapping must not duplicate canonical identities.')
    mr = np.searchsorted(brain.node_ids,mapped)
    if np.any(mr >= brain.n_neurons) or not np.array_equal(brain.node_ids[mr],mapped):
        raise ValueError('Optical mapping contains a noncanonical identity.')
    ids = np.intersect1d(mapped,photo_ids)
    rows = np.searchsorted(brain.node_ids,ids)
    keep = canonical_types[rows] == 'R1-R6'
    ids,rows = ids[keep],rows[keep].astype(np.int64)
    if not len(ids):
        raise ValueError('No R1-R6 has both a real optical mapping and an inherited light-input port.')
    photo_positions = np.searchsorted(photo_ids,ids).astype(np.int64)
    return mapped,ids,rows,photo_positions


def _replace_coefficients(target, rate, rows, photo_positions, light):
    """One scalar ray input per row; no population statistic or visual feature."""
    target[rows] = (15.+40.*light[photo_positions])/80.
    rate[rows] = 1./.005


class CyborgRetinalPortBrain(GradedDescendingBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_cyborg_retinal_port_fp64_v1'

    @classmethod
    def adopt(cls, reference, mapped_ids, enabled=True):
        if (not isinstance(reference,GradedDescendingBrain)
                or reference.SCHEMA not in PARENT_SCHEMAS):
            raise ValueError('Adopt the unchanged graded-descending family exactly once.')
        if type(enabled) is not bool:
            raise ValueError('enabled must be an explicit boolean.')
        b = reference.brain
        types = reference.measured_t4_manifest['canonical_node_types']
        mapped,ids,rows,photo_positions = _selection(b,reference.photo_ids,types,mapped_ids)
        if not np.all(reference.visual_mask[rows]):
            raise ValueError('Retinal device rows must already use graded voltage.')
        if (b.time_ns != reference.time_ns
                or not np.array_equal((reference.release()*reference.caps).astype(np.float32),b.rates)):
            raise ValueError('Parent published view or neural clock is inconsistent.')
        initial_voltage = reference.state[rows].copy()
        manifest = dict(policy=POLICY,enabled=enabled,engineering_device=True,
            biological_calibration=False,animal_ability=False,source_schema=reference.SCHEMA,
            adoption_time_ns=int(reference.time_ns),source_state_sha256=_hash_array(reference.state),
            source_transmission_sha256=_hash_array(reference.transmission_release()),
            adoption_signed_weights_sha256=_hash_array(b.W.data),
            node_ids_sha256=_hash_array(b.node_ids),canonical_types_sha256=_hash_array(types),
            photo_ids_sha256=_hash_array(reference.photo_ids),
            anatomical_indptr_sha256=_hash_array(b.W.indptr),anatomical_indices_sha256=_hash_array(b.W.indices),
            mapped_ids=mapped,target_ids=ids,target_rows=rows,photo_positions=photo_positions,
            initial_target_voltage=initial_voltage,
            initial_targets_outside_affine_release_range=int(np.count_nonzero(
                (initial_voltage < 15./80.) | (initial_voltage > 55./80.))),
            device_parameters=copy.deepcopy(DEVICE_PARAMETERS),
            selection='Intersection of supplied real optical mapping, inherited photo_ids, and canonical R1-R6 type. No response selection.',
            voltage_equation='target=(15+40*own_light)/80; dVnorm/dt=(target-Vnorm)/0.005',
            release_equation='Inherited clip((80*Vnorm-15)/40,0,1). First-order luminance tracking is exact while Vnorm lies in [15/80,55/80].',
            interface_units='Normalized light 0..1 to normalized release 0..1 via existing voltage coordinate; neither measured voltage nor measured firing rate.',
            parameter_provenance='Fixed engineering choice before behavior, using the existing interface range; no behavioral gain fitting or physiological calibration.',
            replaced='Only selected voltage-row coefficients: recurrent inputs and native phototransduction no longer drive those rows when enabled.',
            retained='All state coordinates, old fast/adaptation variables and their evolution, canonical graph/weights/signs, gains, stored taus, existing chemical transmission and all other neural-row equations.',
            claim_scope='Tests the candidate CNS behind a prosthetic retina. Does not reconstruct retinal physiology or attribute the device transfer function to the animal.',
            disabled_scope='Disabled leaves all inherited coefficients bit-for-bit unchanged.')
        manifest['record_sha256'] = _record_hash(manifest)
        saved = reference.state_dict()
        saved['cyborg_retinal_port_manifest'] = manifest
        saved['schema'] = cls.SCHEMA
        return cls.from_state(b,saved)

    def _build(self):
        super()._build()
        self._validate_port()

    def _validate_port(self):
        m,b = self.cyborg_retinal_port_manifest,self.brain
        types = self.measured_t4_manifest['canonical_node_types']
        if (m.get('policy') != POLICY or type(m.get('enabled')) is not bool
                or m.get('engineering_device') is not True or m.get('biological_calibration') is not False
                or m.get('animal_ability') is not False or m.get('source_schema') not in PARENT_SCHEMAS
                or type(m.get('adoption_time_ns')) is not int
                or not 0 <= m['adoption_time_ns'] <= self.time_ns
                or m.get('device_parameters') != DEVICE_PARAMETERS
                or m.get('record_sha256') != _record_hash(m)):
            raise ValueError('Invalid retinal-port policy, engineering constants or history.')
        for key,array in [('node_ids_sha256',b.node_ids),('canonical_types_sha256',types),
                          ('photo_ids_sha256',self.photo_ids),('anatomical_indptr_sha256',b.W.indptr),
                          ('anatomical_indices_sha256',b.W.indices)]:
            if m.get(key) != _hash_array(array):
                raise ValueError('Retinal-port anatomical or optical identity changed: '+key)
        mapped,ids,rows,photo_positions = _selection(b,self.photo_ids,types,m.get('mapped_ids'))
        for key,expected in [('mapped_ids',mapped),('target_ids',ids),('target_rows',rows),
                             ('photo_positions',photo_positions)]:
            value=m.get(key)
            if (not isinstance(value,np.ndarray) or value.dtype != np.int64
                    or not np.array_equal(value,expected)):
                raise ValueError('Retinal-port mapping differs from its canonical intersection: '+key)
        if not np.all(self.visual_mask[rows]):
            raise ValueError('Retinal-port voltage rows are not all graded.')
        initial = m.get('initial_target_voltage')
        if (not isinstance(initial,np.ndarray) or initial.dtype != np.float64
                or initial.shape != rows.shape or not np.isfinite(initial).all()
                or np.any((initial<0.) | (initial>1.))
                or m.get('initial_targets_outside_affine_release_range') != int(np.count_nonzero(
                    (initial<15./80.) | (initial>55./80.)))):
            raise ValueError('Invalid recorded initial retinal voltage coordinates.')
        if self.time_ns == m['adoption_time_ns']:
            if (m.get('source_state_sha256') != _hash_array(self.state)
                    or m.get('source_transmission_sha256') != _hash_array(self.transmission_release())
                    or m.get('adoption_signed_weights_sha256') != _hash_array(b.W.data)
                    or not np.array_equal(initial,self.state[rows])):
                raise ValueError('Retinal-port adoption changed inherited state or signed weights.')
        self._retinal_port_rows = rows
        self._retinal_port_photo_positions = photo_positions

    @property
    def retinal_port_ids(self):
        return self.cyborg_retinal_port_manifest['target_ids'].copy()

    @property
    def retinal_port_rows(self):
        return self._retinal_port_rows.copy()

    def _coefficients(self,state,drive,light):
        target,rate = super()._coefficients(state,drive,light)
        if self.cyborg_retinal_port_manifest['enabled']:
            _replace_coefficients(target,rate,self._retinal_port_rows,
                                  self._retinal_port_photo_positions,light)
        return target,rate

    def state_dict(self):
        saved = super().state_dict()
        saved['cyborg_retinal_port_manifest'] = copy.deepcopy(self.cyborg_retinal_port_manifest)
        return saved

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved) != RETINAL_PORT_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete retinal-port state or wrong neural family.')
        parent = {key:copy.deepcopy(saved[key]) for key in GRADED_DESCENDING_KEYS}
        parent['schema'] = GradedDescendingBrain.SCHEMA
        base = GradedDescendingBrain.from_state(brain,parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.cyborg_retinal_port_manifest = copy.deepcopy(saved['cyborg_retinal_port_manifest'])
        obj._build()
        return obj


# Importing the class does not initialize CUDA; constructing this GPU subclass
# remains the owning session's responsibility.
from gpu_graded_descending_brain import GpuGradedDescendingBrain


class GpuCyborgRetinalPortBrain(CyborgRetinalPortBrain,GpuGradedDescendingBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_cyborg_retinal_port_fp64_cuda_v1'

    def _build(self):
        super()._build()
        import cupy as cp
        self._retinal_port_cuda_rows = cp.asarray(self._retinal_port_rows)
        self._retinal_port_cuda_photo_positions = cp.asarray(self._retinal_port_photo_positions)

    def coefficients_gpu(self,state,drive,light):
        target,rate = super().coefficients_gpu(state,drive,light)
        if self.cyborg_retinal_port_manifest['enabled']:
            _replace_coefficients(target,rate,self._retinal_port_cuda_rows,
                                  self._retinal_port_cuda_photo_positions,light)
        return target,rate

    @staticmethod
    def backend_identity():
        info=GpuGradedDescendingBrain.backend_identity()
        info['retinal_port']='Own mapped light only; fixed 5 ms voltage device on R1-R6; inherited chemical kinetics; engineering prosthesis, not retinal physiology.'
        return info
