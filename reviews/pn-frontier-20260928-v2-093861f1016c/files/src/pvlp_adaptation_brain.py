"""Local intrinsic adaptation hypothesis on the unchanged retinal-port CNS.

For selected nonvisual PVLP024 cells, q_target=max(0,tanh(x-beta*a)) and
tau_a da/dt=q-a, where x is the parent's complete pre-rectification input.
The memory is integrated with the neural/transmission state. It starts at
zero as a declared new component; no pre-adoption history is reconstructed.
Neither beta nor tau_a has a biological default or measured PVLP provenance.
"""
import copy
import math

import numba
import numpy as np

from cyborg_retinal_port import (CyborgRetinalPortBrain, GpuCyborgRetinalPortBrain,
                                RETINAL_PORT_KEYS, _record_hash)
from hybrid_visual_brain import exponential_midpoint
from synaptic_visual_brain import _hash_array


POLICY = 'pvlp024_local_intrinsic_adaptation_v1'
PVLP_ADAPTATION_KEYS = RETINAL_PORT_KEYS | {'pvlp_adaptation_manifest'}
DEFAULT_TARGET_IDS = (16459, 16475, 19454)


def _parameters(beta, tau_s, enabled):
    if type(enabled) is not bool:
        raise ValueError('enabled must be an explicit Python boolean.')
    for name, value in [('beta', beta), ('tau_s', tau_s)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(name+' must be an explicit finite scalar.')
        if not np.isfinite(value) or value < 0 or (name == 'tau_s' and value == 0):
            raise ValueError('beta must be nonnegative and tau_s strictly positive.')
    return float(beta), float(tau_s)


def _selection(brain, types, visual_ids, target_ids):
    ids = np.asarray(target_ids)
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu':
        raise ValueError('Target identities must be a nonempty integer vector.')
    if ids.dtype.kind == 'u' and np.any(ids > np.iinfo(np.int64).max):
        raise ValueError('Target identity exceeds the canonical integer domain.')
    ids = np.sort(ids.astype(np.int64))
    if np.any(np.diff(ids) <= 0):
        raise ValueError('Target identities must not repeat.')
    rows = np.searchsorted(brain.node_ids, ids).astype(np.int64)
    if np.any(rows >= brain.n_neurons) or not np.array_equal(brain.node_ids[rows], ids):
        raise ValueError('Unknown canonical adaptation target.')
    if not np.all(types[rows] == 'PVLP024') or np.intersect1d(ids, visual_ids).size:
        raise ValueError('Adaptation requires canonically annotated, rate-mode PVLP024 cells.')
    return ids, rows


@numba.njit(fastmath=False, cache=True)
def _adapt_targets(rows, indptr, indices, weights, transmission, caps, visual,
                   gain, theta, drive, connected, beta, memory, target):
    # Same per-row CSR order and arithmetic as visual_sparse_kernel.py.
    # Reconstruct x; arctanh of the parent's rectified/saturated target loses it.
    for j in range(len(rows)):
        row = rows[j]
        current = 0.
        for edge in range(indptr[row], indptr[row+1]):
            col = indices[edge]
            if connected or not visual[col]:
                current += weights[edge]*(transmission[col]*caps[col])
        x = gain[row]*(current+drive[row]-theta[row])
        target[row] = max(0., math.tanh(x-beta*memory[j]))


ADAPTATION_CUDA_SOURCE = r'''
extern "C" __global__ void pvlp_adaptation(
 int k, const long long* rows, const long long* ptr, const int* idx,
 const double* w, const double* s, const double* caps, const bool* visual,
 const double* gain, const double* theta, const double* drive, bool connected,
 double beta, const double* memory, double* target) {
 int j=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 if(j>=k) return;
 long long row=rows[j]; double current=0.;
 for(long long e=ptr[row]+lane;e<ptr[row+1];e+=32) {
   int col=idx[e];
   if(connected || !visual[col]) current+=w[e]*(s[col]*caps[col]);
 }
 for(int d=16;d>0;d/=2) current+=__shfl_down_sync(0xffffffff,current,d);
 if(lane==0) {
   double x=gain[row]*(current+drive[row]-theta[row]);
   target[row]=fmax(0.,tanh(x-beta*memory[j]));
 }
}
'''


class PvlpAdaptationBrain(CyborgRetinalPortBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_pvlp_adaptation_fp64_v1'
    PARENT_CLASS = CyborgRetinalPortBrain

    @classmethod
    def adopt(cls, reference, target_ids=DEFAULT_TARGET_IDS, *, beta, tau_s, enabled=True):
        """Replace the parent on its shared brain; preserve every inherited value."""
        beta, tau_s = _parameters(beta, tau_s, enabled)
        if not isinstance(reference, cls.PARENT_CLASS) or reference.SCHEMA != cls.PARENT_CLASS.SCHEMA:
            raise ValueError('Adopt a retinal-port parent of the same numerical backend exactly once.')
        types = reference.measured_t4_manifest['canonical_node_types']
        ids, rows = _selection(reference.brain, types, reference.visual_ids, target_ids)
        saved = reference.state_dict()
        prefix = saved['state']
        record = dict(policy=POLICY, enabled=enabled, beta=beta, tau_s=tau_s,
            target_ids=ids, target_rows=rows, target_type='PVLP024',
            source_schema=reference.SCHEMA, adoption_time_ns=int(reference.time_ns),
            inherited_state_size=int(len(prefix)), initial_inherited_state_sha256=_hash_array(prefix),
            initial_transmission_sha256=_hash_array(reference.transmission_release()),
            adoption_signed_weights_sha256=_hash_array(reference.brain.W.data),
            node_ids_sha256=_hash_array(reference.brain.node_ids), canonical_types_sha256=_hash_array(types),
            anatomical_indptr_sha256=_hash_array(reference.brain.W.indptr),
            anatomical_indices_sha256=_hash_array(reference.brain.W.indices),
            initialization='New memory a=0; inherited state unchanged; pre-adoption adaptation history unavailable.',
            equation='q_target=max(0,tanh((gain/r_max)*(I+drive-theta)-beta*a)); tau_s*da/dt=q-a',
            parameter_provenance='Explicit phenomenological hypothesis; beta and tau_s are not measured PVLP024 parameters.',
            scope='Intrinsic local feedback on each selected cell and all its existing outputs; no direction or motor signal.',
            inactive_error_policy='Memory evolves but is excluded from error norm when disabled or beta=0; inherited trajectory and solver decisions remain exact.',
            biological_validation=False)
        record['record_sha256'] = _record_hash(record)
        saved['pvlp_adaptation_manifest'] = record
        saved['state'] = np.concatenate((prefix, np.zeros(len(rows), dtype=np.float64)))
        saved['schema'] = cls.SCHEMA
        return cls.from_state(reference.brain, saved)

    @property
    def inherited_state_size(self):
        return 2*self.brain.n_neurons+2*len(self.pi)

    @property
    def adaptation_rows(self):
        return self.pvlp_adaptation_manifest['target_rows'].copy()

    @property
    def adaptation_ids(self):
        return self.pvlp_adaptation_manifest['target_ids'].copy()

    @property
    def adaptation_state(self):
        return self.state[self.inherited_state_size:].copy()

    @property
    def adaptation_active(self):
        m = self.pvlp_adaptation_manifest
        return m['enabled'] and m['beta'] != 0.

    def transmission_release(self, state=None):
        state = self.state if state is None else state
        return state[self.transmission_start:self.inherited_state_size].copy()

    def _coefficients(self, state, drive, light):
        size, m = self.inherited_state_size, self.pvlp_adaptation_manifest
        target, rate = CyborgRetinalPortBrain._coefficients(self, state[:size], drive, light)
        rows = m['target_rows']
        if self.adaptation_active:
            b = self.brain
            _adapt_targets(rows, b.W.indptr, b.W.indices, self.weights64,
                state[self.transmission_start:size], self.caps, self.visual_mask,
                self.rate_gain, self.rate_theta, drive, self.visual_output_connected,
                m['beta'], state[size:], target)
        return (np.concatenate((target, state[rows])),
                np.concatenate((rate, np.full(len(rows), 1./m['tau_s']))))

    def _validated_inputs(self, dt_ns, drive, light):
        drive, light = np.asarray(drive, dtype=float), np.asarray(light, dtype=float)
        if type(dt_ns) is not int or dt_ns <= 0 or self.brain.time_ns != self.time_ns:
            raise ValueError('Invalid neural duration/clock.')
        if (drive.shape != (self.brain.n_neurons,) or light.shape != (len(self.pi),)
                or not np.isfinite(drive).all() or not np.isfinite(light).all()
                or np.any((light < 0) | (light > 1)) or np.any(drive[self.vi])):
            raise ValueError('Invalid physical sensory input.')
        return drive, light

    def advance(self, dt_ns, drive, light):
        """Inherited exponential midpoint/doubling; active memory joins its norm."""
        drive, light = self._validated_inputs(dt_ns, drive, light)
        p, remaining, attempts = self.parameters, dt_ns, 0
        coefficients = lambda y: self._coefficients(y, drive, light)
        norm_size = len(self.state) if self.adaptation_active else self.inherited_state_size
        while remaining:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError('Numerical work bound exceeded.')
            h_ns = min(remaining, self.next_step_ns, p['maximum_step_ns'])
            h = h_ns*1e-9
            full = exponential_midpoint(self.state, h, coefficients)
            half = exponential_midpoint(self.state, .5*h, coefficients)
            half = exponential_midpoint(half, .5*h, coefficients)
            scale = p['atol']+p['rtol']*np.maximum(np.abs(full[:norm_size]), np.abs(half[:norm_size]))
            error = float(np.max(np.abs(half[:norm_size]-full[:norm_size])/(3.*scale)))
            if not np.isfinite(error) or not np.isfinite(half).all():
                raise FloatingPointError('Nonfinite adaptive neural integration.')
            if error <= 1.:
                if np.any((half < 0.) | (half > 1.)):
                    raise FloatingPointError('Neural/adaptation state left its domain.')
                self.state = half
                remaining -= h_ns
                self.statistics['accepted'] += 1
                self.statistics['minimum_accepted_ns'] = min(self.statistics['minimum_accepted_ns'], h_ns)
                self.statistics['maximum_accepted_error'] = max(self.statistics['maximum_accepted_error'], error)
                self.next_step_ns = min(p['maximum_step_ns'], h_ns*2 if error < .1 else h_ns)
            else:
                self.statistics['rejected'] += 1
                if h_ns//2 < p['minimum_step_ns']:
                    raise FloatingPointError('Requested neural accuracy unattainable.')
                self.next_step_ns = h_ns//2
        self.time_ns += dt_ns
        self.publish_rates()

    def state_dict(self):
        saved = CyborgRetinalPortBrain.state_dict(self)
        saved['pvlp_adaptation_manifest'] = copy.deepcopy(self.pvlp_adaptation_manifest)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if not isinstance(saved, dict) or set(saved) != PVLP_ADAPTATION_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete adaptation state or wrong neural family.')
        m = saved['pvlp_adaptation_manifest']
        if not isinstance(m, dict):
            raise ValueError('Missing local adaptation manifest.')
        _parameters(m.get('beta'), m.get('tau_s'), m.get('enabled'))
        if (m.get('policy') != POLICY or m.get('target_type') != 'PVLP024'
                or m.get('source_schema') != cls.PARENT_CLASS.SCHEMA
                or m.get('biological_validation') is not False
                or m.get('record_sha256') != _record_hash(m)):
            raise ValueError('Changed adaptation policy or parameter record.')
        types = saved['measured_t4_manifest']['canonical_node_types']
        ids, rows = _selection(brain, types, saved['visual_ids'], m.get('target_ids'))
        for key, expected in [('target_ids', ids), ('target_rows', rows)]:
            actual = m.get(key)
            if not isinstance(actual, np.ndarray) or actual.dtype != np.int64 or not np.array_equal(actual, expected):
                raise ValueError('Changed canonical adaptation selection.')
        size = 2*brain.n_neurons+2*len(saved['photo_ids'])
        state = saved['state']
        if (not isinstance(state, np.ndarray) or state.dtype != np.float64
                or state.shape != (size+len(rows),) or not np.isfinite(state).all()
                or np.any((state < 0) | (state > 1)) or m.get('inherited_state_size') != size):
            raise ValueError('Invalid complete neural/transmission/adaptation state.')
        if type(m.get('adoption_time_ns')) is not int or not 0 <= m['adoption_time_ns'] <= saved['time_ns']:
            raise ValueError('Invalid adaptation adoption clock.')
        for key, value in [('node_ids_sha256', brain.node_ids), ('canonical_types_sha256', types),
                           ('anatomical_indptr_sha256', brain.W.indptr), ('anatomical_indices_sha256', brain.W.indices)]:
            if m.get(key) != _hash_array(value):
                raise ValueError('Changed adaptation anatomy: '+key)
        if m['adoption_time_ns'] == saved['time_ns']:
            start = brain.n_neurons+2*len(saved['photo_ids'])
            if (np.any(state[size:] != 0.) or m['initial_inherited_state_sha256'] != _hash_array(state[:size])
                    or m['initial_transmission_sha256'] != _hash_array(state[start:size])
                    or m['adoption_signed_weights_sha256'] != _hash_array(brain.W.data)):
                raise ValueError('Adaptation adoption changed the parent or its new zero memory.')
        parent = {key: copy.deepcopy(saved[key]) for key in RETINAL_PORT_KEYS}
        parent['schema'] = cls.PARENT_CLASS.SCHEMA
        parent['state'] = state[:size].copy()
        # Validate historical hashes against their exact original prefix. The
        # parent builds all caches before the extra state/manifest is attached.
        base = cls.PARENT_CLASS.from_state(brain, parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.state = state.copy()
        obj.pvlp_adaptation_manifest = copy.deepcopy(m)
        obj._build_adaptation_cache()
        return obj

    def _build_adaptation_cache(self):
        pass


class GpuPvlpAdaptationBrain(PvlpAdaptationBrain, GpuCyborgRetinalPortBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_pvlp_adaptation_fp64_cuda_v1'
    PARENT_CLASS = GpuCyborgRetinalPortBrain

    def _build_adaptation_cache(self):
        import cupy as cp
        self._pvlp_cuda_rows = cp.asarray(self.adaptation_rows, dtype=cp.int64)
        self._pvlp_kernel = cp.RawKernel(ADAPTATION_CUDA_SOURCE, 'pvlp_adaptation',
            options=('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'))

    def coefficients_gpu(self, state, drive, light):
        import cupy as cp
        size, m = self.inherited_state_size, self.pvlp_adaptation_manifest
        target, rate = GpuCyborgRetinalPortBrain.coefficients_gpu(self, state[:size], drive, light)
        rows = self._pvlp_cuda_rows
        if self.adaptation_active:
            c, k = self.cuda, len(rows)
            args = (np.int32(k), rows, c['indptr'], c['indices'], c['weights'],
                state[self.transmission_start:size], c['caps'], c['visual'], c['gain'], c['theta'],
                drive, np.bool_(self.visual_output_connected), np.float64(m['beta']), state[size:], target)
            self._pvlp_kernel(((k*32+255)//256,), (256,), args)
        return (cp.concatenate((target, state[rows])),
                cp.concatenate((rate, cp.full(len(rows), 1./m['tau_s'], dtype=cp.float64))))

    def advance(self, dt_ns, drive, light):
        import cupy as cp
        drive, light = self._validated_inputs(dt_ns, drive, light)
        p, y = self.parameters, cp.asarray(self.state)
        dg, lg = cp.asarray(drive), cp.asarray(light)
        def midpoint(z, h):
            target, rate = self.coefficients_gpu(z, dg, lg)
            middle = z+(-cp.expm1(-.5*h*rate))*(target-z)
            target, rate = self.coefficients_gpu(middle, dg, lg)
            return z+(-cp.expm1(-h*rate))*(target-z)
        remaining, attempts = dt_ns, 0
        norm_size = len(self.state) if self.adaptation_active else self.inherited_state_size
        while remaining:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError('Numerical work bound exceeded.')
            h_ns = min(remaining, self.next_step_ns, p['maximum_step_ns'])
            h = h_ns*1e-9
            full = midpoint(y, h)
            half = midpoint(midpoint(y, .5*h), .5*h)
            scale = p['atol']+p['rtol']*cp.maximum(cp.abs(full[:norm_size]), cp.abs(half[:norm_size]))
            error = float(cp.max(cp.abs(half[:norm_size]-full[:norm_size])/(3.*scale)))
            if not np.isfinite(error) or not bool(cp.isfinite(half).all()):
                raise FloatingPointError('Nonfinite adaptive neural integration.')
            if error <= 1.:
                if bool(cp.any((half < 0.) | (half > 1.))):
                    raise FloatingPointError('Neural/adaptation state left its domain.')
                y = half
                remaining -= h_ns
                self.statistics['accepted'] += 1
                self.statistics['minimum_accepted_ns'] = min(self.statistics['minimum_accepted_ns'], h_ns)
                self.statistics['maximum_accepted_error'] = max(self.statistics['maximum_accepted_error'], error)
                self.next_step_ns = min(p['maximum_step_ns'], h_ns*2 if error < .1 else h_ns)
            else:
                self.statistics['rejected'] += 1
                if h_ns//2 < p['minimum_step_ns']:
                    raise FloatingPointError('Requested neural accuracy unattainable.')
                self.next_step_ns = h_ns//2
        self.state = cp.asnumpy(y)
        self.time_ns += dt_ns
        self.publish_rates()

    @staticmethod
    def backend_identity():
        info = GpuCyborgRetinalPortBrain.backend_identity()
        info['pvlp_adaptation'] = 'Selected nonvisual PVLP024 intrinsic feedback; deterministic FP64 warp reduction; memory integrated by exponential midpoint and step doubling.'
        return info
