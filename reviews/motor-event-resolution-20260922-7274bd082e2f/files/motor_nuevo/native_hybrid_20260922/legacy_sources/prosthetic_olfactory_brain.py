"""A fixed external afferent-rate prosthesis on selected ORN DM1 -> PN terms.

The device supplies rate only to the four-state synaptic filters replacing
selected ORN-to-PN terms. It bypasses endogenous ORN activity and terminal
feedback on that branch, while other ORN outputs remain in the inherited CNS.
The output dynamics, topology, signed weights and neuronal parameters remain
as in the local synaptic bridge. This is an engineering sensory substitution,
not reconstruction of peripheral transduction or endogenous axonal modulation.
"""
import copy
import math
from pathlib import Path

import numba
import numpy as np

from cyborg_retinal_port import _record_hash
from hybrid_visual_brain import exponential_midpoint
from olfactory_synapse_candidate import FAST, SLOW, SOURCE_DOI
from pvlp_adaptation_brain import (PvlpAdaptationBrain, GpuPvlpAdaptationBrain,
                                   PVLP_ADAPTATION_KEYS)
from synaptic_visual_brain import _hash_array


POLICY = 'prosthetic_olfactory_dm1_pn_rate_port_v1'
ORN_PN_KEYS = PVLP_ADAPTATION_KEYS | {'orn_pn_synaptic_manifest', 'held_afferent_rate_hz'}
FAST_AREA = FAST.k_ns_per_spike * FAST.tau_g_s
SLOW_AREA = SLOW.k_ns_per_spike * SLOW.tau_g_s
NORMALIZATION = FAST_AREA + SLOW_AREA
SELECTION_KEYS = ('pre_ids', 'post_ids', 'pre_indices', 'post_indices',
                  'w_csr_data_positions', 'weights_post_pre_current',
                  'anatomical_contact_counts', 'source_orn_ids',
                  'source_orn_indices', 'source_rmax_hz')


def _integers(value, name):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a):
        raise ValueError(name+' must be a nonempty integer vector.')
    if a.dtype.kind == 'u' and np.any(a > np.iinfo(np.int64).max):
        raise ValueError(name+' exceeds int64.')
    return a.astype(np.int64, copy=True)


def validate_selection(reference, selection):
    """Validate anatomical identities and oriented CSR terms, never responses."""
    if isinstance(selection, (str, Path)):
        with np.load(selection, allow_pickle=False) as archive:
            selection = {key: archive[key].copy() for key in SELECTION_KEYS}
    try:
        result = {key: _integers(selection[key], key) for key in SELECTION_KEYS
                  if key not in ('weights_post_pre_current', 'source_rmax_hz')}
        result['weights_post_pre_current'] = np.asarray(selection['weights_post_pre_current']).copy()
        result['source_rmax_hz'] = np.asarray(selection['source_rmax_hz'], dtype=np.float64).copy()
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete ORN/PN anatomical selection.') from exc
    b = reference.brain
    pre, post, positions = (result[key] for key in
                           ('pre_indices', 'post_indices', 'w_csr_data_positions'))
    n = len(pre)
    edge_keys = SELECTION_KEYS[:7]
    if any(result[key].shape != (n,) for key in edge_keys):
        raise ValueError('Inconsistent selected-edge arrays.')
    if (np.any(pre < 0) or np.any(pre >= b.n_neurons)
            or np.any(post < 0) or np.any(post >= b.n_neurons)
            or np.any(positions < 0) or np.any(positions >= b.W.nnz)
            or len(np.unique(positions)) != n):
        raise ValueError('Unknown or duplicated selected CSR term.')
    if (not np.array_equal(b.node_ids[pre], result['pre_ids'])
            or not np.array_equal(b.node_ids[post], result['post_ids'])
            or not np.array_equal(b.W.indices[positions], pre)
            or np.any(positions < b.W.indptr[post])
            or np.any(positions >= b.W.indptr[post+1])):
        raise ValueError('Selected CSR positions do not identify post <- pre.')
    weights = result['weights_post_pre_current']
    if (weights.dtype.kind != 'f' or not np.isfinite(weights).all()
            or np.any(weights <= 0.) or np.any(result['anatomical_contact_counts'] <= 0)
            or not np.array_equal(weights, b.W.data[positions])
            or not np.array_equal(reference.weights64[positions], weights.astype(np.float64))):
        raise ValueError('Selected positive weights differ from inherited current terms.')
    types = reference.measured_t4_manifest['canonical_node_types']
    if (not np.all(types[pre] == 'ORN_DM1') or not np.all(types[post] == 'DM1_lPN')
            or np.any(reference.visual_mask[pre]) or np.any(reference.visual_mask[post])):
        raise ValueError('This bridge requires rate-mode ORN_DM1 -> DM1_lPN identities.')
    sources = np.unique(pre).astype(np.int64)
    if (not np.array_equal(result['source_orn_indices'], sources)
            or not np.array_equal(result['source_orn_ids'], b.node_ids[sources])
            or result['source_rmax_hz'].shape != sources.shape
            or not np.isfinite(result['source_rmax_hz']).all()
            or np.any(result['source_rmax_hz'] <= 0.)
            or not np.array_equal(result['source_rmax_hz'], reference.caps[sources])):
        raise ValueError('Source identities or rate-proxy caps disagree with the CNS.')
    order = np.argsort(positions)
    for key in edge_keys:
        result[key] = result[key][order].copy()
    return result


def _initial_states(rate_hz, caps, initialization):
    q = rate_hz/caps
    if initialization == 'rested':
        return np.concatenate((np.ones(2*len(q)), np.zeros(2*len(q))))
    if initialization == 'steady_baseline':
        af = 1./(1.+FAST.r_per_spike*FAST.tau_A_s*rate_hz)
        ass = 1./(1.+SLOW.r_per_spike*SLOW.tau_A_s*rate_hz)
        return np.concatenate((af, ass, q*af, q*ass))
    raise ValueError("initialization must be 'rested' or 'steady_baseline'.")


def _validated_afferent_rate(value, caps):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.ndim not in (0, 1):
        raise ValueError('Afferent rate must be a real scalar or source-sized vector.')
    if raw.ndim == 0:
        result = np.full(len(caps), float(raw), dtype=np.float64)
    elif raw.shape == caps.shape:
        result = raw.astype(np.float64, copy=True)
    else:
        raise ValueError('Afferent rate vector differs from selected source count.')
    if not np.isfinite(result).all() or np.any(result < 0.) or np.any(result > caps):
        raise ValueError('Afferent rate must be finite and between zero and source caps.')
    return result


def _component_parameters():
    return dict(fast=FAST.as_dict(), slow=SLOW.as_dict(), source_doi=SOURCE_DOI,
                normalization_nS_per_Hz=NORMALIZATION,
                state_order=['A_fast', 'A_slow', 'z_fast', 'z_slow'])


@numba.njit(fastmath=False, cache=True)
def _replace_pn_targets(rows, local_ptr, slots, indptr, indices, weights,
                        transmission, caps, visual, gain, theta, drive,
                        connected, z_fast, z_slow, target):
    for j in range(len(rows)):
        row = rows[j]
        current = 0.
        for edge in range(indptr[row], indptr[row+1]):
            col = indices[edge]
            if connected or not visual[col]:
                slot = slots[local_ptr[j]+edge-indptr[row]]
                if slot >= 0:
                    q = (FAST_AREA*z_fast[slot]+SLOW_AREA*z_slow[slot])/NORMALIZATION
                else:
                    q = transmission[col]
                current += weights[edge]*(q*caps[col])
        target[row] = max(0., math.tanh(gain[row]*(current+drive[row]-theta[row])))


ORN_PN_CUDA_SOURCE = r'''
extern "C" __global__ void orn_pn_synaptic(
 int k, const long long* rows, const long long* local_ptr, const int* slots,
 const long long* ptr, const int* idx, const double* w, const double* s,
 const double* caps, const bool* visual, const double* gain,
 const double* theta, const double* drive, bool connected,
 const double* z_fast, const double* z_slow, double fast_area,
 double slow_area, double normalization, double* target) {
 int j=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 if(j>=k) return;
 long long row=rows[j]; double current=0.;
 for(long long e=ptr[row]+lane;e<ptr[row+1];e+=32) {
   int col=idx[e];
   if(connected || !visual[col]) {
     int slot=slots[local_ptr[j]+e-ptr[row]];
     double q=slot>=0 ? (fast_area*z_fast[slot]+slow_area*z_slow[slot])/normalization : s[col];
     current+=w[e]*(q*caps[col]);
   }
 }
 for(int d=16;d>0;d/=2) current+=__shfl_down_sync(0xffffffff,current,d);
 if(lane==0) target[row]=fmax(0.,tanh(gain[row]*(current+drive[row]-theta[row])));
}
'''


class ProstheticOlfactoryBrain(PvlpAdaptationBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_prosthetic_olfactory_fp64_v1'
    ORN_PARENT_CLASS = PvlpAdaptationBrain

    @classmethod
    def adopt(cls, reference, selection, initialization, enabled=True, baseline_hz=7.0):
        if (not isinstance(reference, cls.ORN_PARENT_CLASS)
                or reference.SCHEMA != cls.ORN_PARENT_CLASS.SCHEMA):
            raise ValueError('Adopt the same-backend PVLP parent exactly once.')
        if type(enabled) is not bool:
            raise ValueError('enabled must be an explicit Python boolean.')
        selected = validate_selection(reference, selection)
        rows, caps = selected['source_orn_indices'], selected['source_rmax_hz']
        if np.asarray(baseline_hz).ndim != 0:
            raise ValueError('baseline_hz must be one scalar device setting.')
        held = _validated_afferent_rate(baseline_hz, caps)
        initial = _initial_states(held, caps, initialization)
        saved = reference.state_dict()
        prefix, b = saved['state'], reference.brain
        manifest = dict(policy=POLICY, enabled=enabled, source_schema=reference.SCHEMA,
            adoption_time_ns=int(reference.time_ns), parent_state_size=len(prefix),
            initialization=initialization, baseline_hz=float(held[0]),
            initial_afferent_rate_hz=held.copy(),
            device_scope='External rate input replaces only selected ORN-to-PN source terms; endogenous ORN outputs elsewhere and all other CNS equations remain active.',
            control_scope='Rate-to-current sensory engineering with no stimulus identity, direction, reward or behavioral policy in the effector path.',
            rate_reference='7 Hz baseline and 20/50 Hz 500 ms electrical nerve stimuli in VM2: Kazama and Wilson 2008, doi:10.1016/j.neuron.2008.02.030, Figure 8D-E. Engineering transfer to DM1; not measured DM1 odor transduction.',
            initialization_scope='New synaptic-history assumption; neither rested nor stationary states reconstruct unknown pre-adoption history.',
            parameters=_component_parameters(), biological_validation=False,
            input='Held external afferent rate in spike/s units, supplied locally by the device; not endogenous ORN q, old transmission or a concentration-response measurement.',
            equation='dA_x/dt=(1-A_x)/tau_Ax-r_x*rate_external_Hz*A_x; dz_x/dt=((rate_external_Hz/r_max)*A_x-z_x)/tau_gx',
            output='Replace selected W*(s_old*r_max) by W*r_max*(k_f*tau_gf*z_f+k_s*tau_gs*z_s)/sum(k*tau_g).',
            gain_scope='Low-rate undepressed DC normalization, not a measured conductance/current transfer or gain preservation at finite depressed rates.',
            inactive_error_policy='New states evolve but do not enter the error norm when disabled; parent state and solver decisions remain exact.',
            initial_parent_state_sha256=_hash_array(prefix), initial_new_state_sha256=_hash_array(initial),
            initial_transmission_sha256=_hash_array(reference.transmission_release()),
            adoption_signed_weights_sha256=_hash_array(b.W.data),
            node_ids_sha256=_hash_array(b.node_ids),
            canonical_types_sha256=_hash_array(reference.measured_t4_manifest['canonical_node_types']),
            anatomical_indptr_sha256=_hash_array(b.W.indptr),
            anatomical_indices_sha256=_hash_array(b.W.indices), **selected)
        manifest['record_sha256'] = _record_hash(manifest)
        saved['orn_pn_synaptic_manifest'] = manifest
        saved['held_afferent_rate_hz'] = held.copy()
        saved['state'] = np.concatenate((prefix, initial))
        saved['schema'] = cls.SCHEMA
        return cls.from_state(b, saved)

    @property
    def parent_state_size(self):
        return self.inherited_state_size+len(self.pvlp_adaptation_manifest['target_rows'])

    @property
    def adaptation_state(self):
        return self.state[self.inherited_state_size:self.parent_state_size].copy()

    @property
    def orn_source_ids(self):
        return self.orn_pn_synaptic_manifest['source_orn_ids'].copy()

    @property
    def orn_source_rows(self):
        return self.orn_pn_synaptic_manifest['source_orn_indices'].copy()

    @property
    def source_caps(self):
        return self.orn_pn_synaptic_manifest['source_rmax_hz'].copy()

    @property
    def orn_synaptic_state(self):
        return self.state[self.parent_state_size:].reshape(4, -1).copy()

    @property
    def orn_synaptic_enabled(self):
        return self.orn_pn_synaptic_manifest['enabled']

    @property
    def held_afferent_rate_hz(self):
        return self._held_afferent_rate_hz.copy()

    def hold_afferent_rate(self, rate_hz):
        """Hold a device input until replaced; do not reset any neural state."""
        held = _validated_afferent_rate(rate_hz, self._orn_caps)
        self._held_afferent_rate_hz = held
        if hasattr(self, '_orn_pn_cuda'):
            import cupy as cp
            self._orn_pn_cuda['held_rate'] = cp.asarray(held)

    def _norm_size(self):
        if self.orn_synaptic_enabled:
            return len(self.state)
        return self.parent_state_size if self.adaptation_active else self.inherited_state_size

    def _build_orn_pn_cache(self):
        m, b = self.orn_pn_synaptic_manifest, self.brain
        self._orn_rows = m['source_orn_indices'].copy()
        self._orn_caps = m['source_rmax_hz'].copy()
        self._pn_rows = np.unique(m['post_indices']).astype(np.int64)
        self._pn_local_ptr = np.r_[0, np.cumsum(b.W.indptr[self._pn_rows+1]-b.W.indptr[self._pn_rows])].astype(np.int64)
        self._pn_slots = np.full(self._pn_local_ptr[-1], -1, dtype=np.int32)
        for pre, post, pos in zip(m['pre_indices'], m['post_indices'], m['w_csr_data_positions']):
            j = np.searchsorted(self._pn_rows, post)
            slot = np.searchsorted(self._orn_rows, pre)
            self._pn_slots[self._pn_local_ptr[j]+pos-b.W.indptr[post]] = slot

    def _coefficients(self, state, drive, light):
        size, rows = self.parent_state_size, self._orn_rows
        target, rate = PvlpAdaptationBrain._coefficients(self, state[:size], drive, light)
        af, ass, zf, zs = state[size:].reshape(4, -1)
        held = self._held_afferent_rate_hz
        q = held/self._orn_caps
        rf = 1./FAST.tau_A_s+FAST.r_per_spike*held
        rs = 1./SLOW.tau_A_s+SLOW.r_per_spike*held
        if self.orn_synaptic_enabled:
            b = self.brain
            _replace_pn_targets(self._pn_rows, self._pn_local_ptr, self._pn_slots,
                b.W.indptr, b.W.indices, self.weights64,
                state[self.transmission_start:self.inherited_state_size], self.caps,
                self.visual_mask, self.rate_gain, self.rate_theta, drive,
                self.visual_output_connected, zf, zs, target)
        return (np.concatenate((target, (1./FAST.tau_A_s)/rf, (1./SLOW.tau_A_s)/rs, q*af, q*ass)),
                np.concatenate((rate, rf, rs, np.full(len(rows), 1./FAST.tau_g_s),
                                np.full(len(rows), 1./SLOW.tau_g_s))))

    def advance(self, dt_ns, drive, light):
        drive, light = self._validated_inputs(dt_ns, drive, light)
        p, remaining, attempts = self.parameters, dt_ns, 0
        coefficients = lambda y: self._coefficients(y, drive, light)
        norm_size = self._norm_size()
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
                    raise FloatingPointError('Neural/synaptic state left its domain.')
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
        saved = PvlpAdaptationBrain.state_dict(self)
        saved['orn_pn_synaptic_manifest'] = copy.deepcopy(self.orn_pn_synaptic_manifest)
        saved['held_afferent_rate_hz'] = self.held_afferent_rate_hz
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if not isinstance(saved, dict) or set(saved) != ORN_PN_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete ORN/PN state or wrong neural family.')
        m = saved['orn_pn_synaptic_manifest']
        if (not isinstance(m, dict) or m.get('policy') != POLICY
                or type(m.get('enabled')) is not bool
                or m.get('source_schema') != cls.ORN_PARENT_CLASS.SCHEMA
                or m.get('parameters') != _component_parameters()
                or m.get('initialization') not in ('rested', 'steady_baseline')
                or m.get('biological_validation') is not False
                or m.get('record_sha256') != _record_hash(m)):
            raise ValueError('Changed ORN/PN policy, parameters or provenance.')
        size = 2*brain.n_neurons+2*len(saved['photo_ids'])+len(saved['pvlp_adaptation_manifest']['target_rows'])
        rows = _integers(m.get('source_orn_indices'), 'source_orn_indices')
        state = saved['state']
        if (m.get('parent_state_size') != size or not isinstance(state, np.ndarray)
                or state.dtype != np.float64 or state.shape != (size+4*len(rows),)
                or not np.isfinite(state).all() or np.any((state < 0.) | (state > 1.))
                or type(m.get('adoption_time_ns')) is not int
                or not 0 <= m['adoption_time_ns'] <= saved['time_ns']):
            raise ValueError('Invalid complete neural/synaptic state or adoption clock.')
        parent = {key: copy.deepcopy(saved[key]) for key in PVLP_ADAPTATION_KEYS}
        parent['schema'] = cls.ORN_PARENT_CLASS.SCHEMA
        parent['state'] = state[:size].copy()
        base = cls.ORN_PARENT_CLASS.from_state(brain, parent)
        selected = validate_selection(base, m)
        baseline = m.get('baseline_hz')
        if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
            raise ValueError('Invalid baseline device rate.')
        initial_rate = _validated_afferent_rate(baseline, selected['source_rmax_hz'])
        initial_saved = m.get('initial_afferent_rate_hz')
        if (not isinstance(initial_saved, np.ndarray) or initial_saved.dtype != np.float64
                or not np.array_equal(initial_saved, initial_rate)):
            raise ValueError('Changed baseline afferent initialization.')
        held = saved['held_afferent_rate_hz']
        if not isinstance(held, np.ndarray) or held.dtype != np.float64 or held.shape != initial_rate.shape:
            raise ValueError('Held afferent input must be a saved FP64 source vector.')
        held = _validated_afferent_rate(held, selected['source_rmax_hz'])
        for key, expected in selected.items():
            actual = m.get(key)
            if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError('Changed canonical ORN/PN selection: '+key)
        for key, array in [('node_ids_sha256', brain.node_ids),
                           ('canonical_types_sha256', base.measured_t4_manifest['canonical_node_types']),
                           ('anatomical_indptr_sha256', brain.W.indptr), ('anatomical_indices_sha256', brain.W.indices)]:
            if m.get(key) != _hash_array(array):
                raise ValueError('Changed ORN/PN anatomy: '+key)
        if m['adoption_time_ns'] == saved['time_ns']:
            initial = _initial_states(initial_rate, base.caps[rows], m['initialization'])
            if (not np.array_equal(state[size:], initial)
                    or m.get('initial_parent_state_sha256') != _hash_array(state[:size])
                    or m.get('initial_new_state_sha256') != _hash_array(initial)
                    or m.get('initial_transmission_sha256') != _hash_array(base.transmission_release())
                    or m.get('adoption_signed_weights_sha256') != _hash_array(brain.W.data)):
                raise ValueError('ORN/PN adoption changed inherited state or initialization.')
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.state = state.copy()
        obj.orn_pn_synaptic_manifest = copy.deepcopy(m)
        obj._held_afferent_rate_hz = held.copy()
        obj._build_orn_pn_cache()
        return obj


class GpuProstheticOlfactoryBrain(ProstheticOlfactoryBrain, GpuPvlpAdaptationBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_prosthetic_olfactory_fp64_cuda_v1'
    ORN_PARENT_CLASS = GpuPvlpAdaptationBrain

    def _build_orn_pn_cache(self):
        super()._build_orn_pn_cache()
        import cupy as cp
        self._orn_pn_cuda = dict(rows=cp.asarray(self._orn_rows), caps=cp.asarray(self._orn_caps),
            pn_rows=cp.asarray(self._pn_rows), ptr=cp.asarray(self._pn_local_ptr), slots=cp.asarray(self._pn_slots),
            held_rate=cp.asarray(self._held_afferent_rate_hz))
        self._orn_pn_kernel = cp.RawKernel(ORN_PN_CUDA_SOURCE, 'orn_pn_synaptic',
            options=('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'))

    def coefficients_gpu(self, state, drive, light):
        import cupy as cp
        size, o = self.parent_state_size, self._orn_pn_cuda
        target, rate = GpuPvlpAdaptationBrain.coefficients_gpu(self, state[:size], drive, light)
        af, ass, zf, zs = state[size:].reshape(4, -1)
        q = o['held_rate']/o['caps']
        rf = 1./FAST.tau_A_s+FAST.r_per_spike*o['held_rate']
        rs = 1./SLOW.tau_A_s+SLOW.r_per_spike*o['held_rate']
        if self.orn_synaptic_enabled:
            c, k = self.cuda, len(self._pn_rows)
            args = (np.int32(k), o['pn_rows'], o['ptr'], o['slots'], c['indptr'], c['indices'],
                c['weights'], state[self.transmission_start:self.inherited_state_size],
                c['caps'], c['visual'], c['gain'], c['theta'], drive,
                np.bool_(self.visual_output_connected), zf, zs,
                np.float64(FAST_AREA), np.float64(SLOW_AREA), np.float64(NORMALIZATION), target)
            self._orn_pn_kernel(((k*32+255)//256,), (256,), args)
        return (cp.concatenate((target, (1./FAST.tau_A_s)/rf, (1./SLOW.tau_A_s)/rs, q*af, q*ass)),
                cp.concatenate((rate, rf, rs, cp.full(len(q), 1./FAST.tau_g_s, dtype=cp.float64),
                                cp.full(len(q), 1./SLOW.tau_g_s, dtype=cp.float64))))

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
        remaining, attempts, norm_size = dt_ns, 0, self._norm_size()
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
                    raise FloatingPointError('Neural/synaptic state left its domain.')
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
        info = GpuPvlpAdaptationBrain.backend_identity()
        info['prosthetic_olfactory'] = 'External held local afferent rate into four synaptic states per selected source; only selected ORN_DM1 -> PN terms replaced; all other ORN outputs remain endogenous; FP64 fixed warp reductions and inherited adaptive integrator.'
        return info
