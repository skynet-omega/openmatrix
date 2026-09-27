"""Generated coefficient layout; arithmetic/order retained. See generator provenance."""
import orn_peripheral_terminal_brain as _m0
import pn_general_output_brain as _m1
import olfactory_endogenous_brain as _m2
import kc_axonal_brain as _m3
import kc_apl_dynamic_brain as _m4
import kcgamma_output_brain as _m5
import pnkc_receptor_brain as _m6
import lamina_boundary_brain as _m7
import cvn7_conductance_brain as _m8
import retinal_transduction_brain as _m9
import t4_gaba_brain as _m10
import kcgamma_regional_brain as _m11
import prosthetic_olfactory_brain as _m12
import pvlp_adaptation_brain as _m13
import cyborg_retinal_port as _m14
import gpu_measured_t4_visual_brain as _m15
import gpu_synaptic_visual_brain as _m16

def _OrnPeripheralTerminalMixin(self, state, drive, light, *, frame):
    (target, rate) = _GpuPnGeneralOutputBrain(self, state, drive, light, frame=frame)
    import cupy as cp
    d = self._orn_terminal_cuda
    c = self.cuda
    local = cp.asarray(self._online_source.general_transmission())
    self._orn_terminal_kernel((1,), (128,), tuple((d[k] for k in ('rows', 'indptr', 'pres', 'positions', 'pn_slot', 'recurrent'))) + (c['weights'], c['caps'], c['gain'], state[self.transmission_start:], local, drive, _m0.np.bool_(self.pn_online_manifest['orn_peripheral_terminal']['recurrent_connected']), target))
    return (target, rate)

def _GpuPnGeneralOutputBrain(self, state, drive, light, *, frame):
    if not hasattr(self, '_general_positions') or not self.pn_online_manifest['general_outputs']['enabled']:
        return _GpuOlfactoryEndogenousBrain(self, state, drive, light, frame=frame)
    if self._general_buffer_active:
        raise RuntimeError('General PN edge buffer cannot be reentered')
    import cupy as cp
    pn = self._online_ports.pn_row
    transmission = self.transmission_start + pn
    weights = self.cuda['weights']
    positions = self._general_positions
    old = weights[positions].copy()
    view = state.copy()
    view[transmission] = 1.0
    self._general_buffer_active = True
    try:
        weights[positions] = old * cp.asarray(self._online_source.general_transmission())
        (target, rate) = _GpuOlfactoryEndogenousBrain(self, view, drive, light, frame=frame)
        target[transmission] = state[pn]
        rate[transmission] = 1.0 / self.parameters['synaptic_tau_s']
        return (target, rate)
    finally:
        weights[positions] = old
        self._general_buffer_active = False

def _GpuOlfactoryEndogenousBrain(self, state, drive, light, *, frame):
    import cupy as cp
    (target, rate) = _GpuKcAxonalBrain(self, state, drive, light, frame=frame)
    if self.olfactory_endogenous_manifest['enabled']:
        (start, end) = (self.parent_state_size, self.regional_parent_state_size)
        o = self._orn_pn_cuda
        (t, r) = _m2.endogenous_coefficients(state, o['rows'], o['caps'], start, end, xp=cp)
        target[start:end] = t
        rate[start:end] = r
    return (target, rate)

def _GpuKcAxonalBrain(self, state, drive, light, *, frame):
    (target, rate) = _GpuKcAplDynamicBrain(self, state, drive, light, frame=frame)
    if self.kc_axonal_manifest['enabled']:
        start = self.regional_parent_state_size
        n = len(self.regional_rows)
        for sl in [slice(start, start + n), slice(start + 2 * n, start + 3 * n)]:
            target[sl] = state[sl]
            rate[sl] = 0.0
    return (target, rate)

def _GpuKcAplDynamicBrain(self, state, drive, light, *, frame):
    if not self.kc_apl_dynamic_manifest['enabled']:
        return _GpuKcGammaOutputBrain(self, state, drive, light, frame=frame)
    if not self._edge_buffer_active:
        raise RuntimeError('Regional release buffer must be installed during coefficient evaluation')
    view = state.copy()
    view[self.transmission_start + self._apl_gpu_rows] = 1.0
    (target, rate) = _GpuKcGammaOutputBrain(self, view, drive, light, frame=frame)
    target[self._dynamic_gpu_rows] = state[self._dynamic_gpu_rows]
    rate[self._dynamic_gpu_rows] = 0.0
    target[self.transmission_start + self._apl_gpu_rows] = state[self.transmission_start + self._apl_gpu_rows]
    rate[self.transmission_start + self._apl_gpu_rows] = 0.0
    return (target, rate)

def _GpuKcGammaOutputBrain(self, state, drive, light, *, frame):
    (target, rate) = _GpuPnkcReceptorBrain(self, state, drive, light, frame=frame)
    if not (self.kcgamma_output_manifest['enabled'] and self.regional_manifest['enabled']):
        return (target, rate)
    r = self._output_cuda
    c = self.cuda
    n = len(self._output_cache['rows'])
    sax = state[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3, -1)[2]
    args = (_m5.np.int32(n), r['rows'], r['local_ptr'], r['mode'], r['fraction'], r['slot'], c['indptr'], c['indices'], c['weights'], state[self.transmission_start:self.inherited_state_size], sax, c['caps'], c['visual'], c['gain'], c['theta'], drive, _m5.np.bool_(self.visual_output_connected), target)
    self._regional_current_kernel(((n * 32 + 255) // 256,), (256,), args)
    return (target, rate)

def _GpuPnkcReceptorBrain(self, state, drive, light, *, frame):
    import cupy as cp
    (target, rate) = _GpuLaminaBoundaryBrain(self, state[:_m6.PARENT_STATE_SIZE], drive, light, frame=frame)
    if not self.pnkc_receptor_manifest['enabled']:
        return (frame.append('target', target, state[_m6.PARENT_STATE_SIZE:]), frame.append('rate', rate, cp.zeros(len(state) - _m6.PARENT_STATE_SIZE)))
    r = self._pnkc_cuda
    c = self.cuda
    rows = r['source_rows']
    gate = state[_m6.PARENT_STATE_SIZE:]
    (rt, rr) = _m6.receptor_coefficients(state[rows] * c['caps'][rows])
    release = state[self.transmission_start:self.inherited_state_size].copy()
    release[rows] = _m6.BETA_PER_S / _m6.K_PER_HZ * gate / c['caps'][rows]
    sax = state[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3, -1)[2]
    n = len(self._pnkc_cache['rows'])
    args = (_m6.np.int32(n), r['rows'], r['local_ptr'], r['mode'], r['fraction'], r['slot'], c['indptr'], c['indices'], c['weights'], release, sax, c['caps'], c['visual'], c['gain'], c['theta'], drive, _m6.np.bool_(self.visual_output_connected), target)
    self._regional_current_kernel(((n * 32 + 255) // 256,), (256,), args)
    return (frame.append('target', target, rt), frame.append('rate', rate, rr))

def _GpuLaminaBoundaryBrain(self, state, drive, light, *, frame):
    import cupy as cp
    size = self.boundary_parent_state_size
    (target, rate) = _GpuCvn7ConductanceBrain(self, state[:size], drive, light, frame=frame)
    if not self.boundary_manifest['enabled']:
        return (frame.append('target', target, state[size:]), frame.append('rate', rate, cp.zeros(len(state) - size)))
    (available, bump, voltage, transmission) = state[size:].reshape(4, -1)
    c = self._boundary_cuda
    (at, ar, bt) = _m7.retinal_coefficients(available, c['light'])
    photo = _m7.RETINAL_PARAMETERS['maximum_relative_photoconductance'] * bump
    total = 1.0 + photo
    ephoto = (_m7.RETINAL_PARAMETERS['photo_reversal_mv'] + 80.0) / 80.0
    ut = (0.25 + ephoto * photo) / total
    st = cp.clip((80.0 * voltage - 15.0) / 40.0, 0.0, 1.0)
    _m7._histamine_targets(target, rate, c['rows'], c['sensor'], c['gmax'], transmission, self.cuda['tau'])
    n = len(available)
    return (frame.append('target', target, at, bt, ut, st), frame.append('rate', rate, ar, cp.full(n, 1.0 / _m7.RETINAL_PARAMETERS['bump_filter_tau_s']), total / _m7.SENSOR_MEMBRANE_TAU_S, cp.full(n, 1.0 / _m7.SENSOR_TRANSMISSION_TAU_S)))

def _GpuCvn7ConductanceBrain(self, state, drive, light, *, frame):
    (target, rate) = _GpuRetinalTransductionBrain(self, state, drive, light, frame=frame)
    if self.cvn7_manifest['enabled']:
        (c, p) = (self.cuda, _m8.PARAMETERS)
        self._cvn7_kernel((1,), (256,), (_m8.np.int32(len(self._cvn7_rows)), self._cvn7_cuda_rows, c['indptr'], c['indices'], c['weights'], state[self.transmission_start:self.inherited_state_size], c['caps'], c['visual'], c['tau'], _m8.np.float64(self.parameters['conductance_per_stored_weight']), _m8.np.bool_(self.visual_output_connected), drive, *[_m8.np.float64(p[key]) for key in ('rest_mV', 'reset_mV', 'threshold_mV', 'membrane_tau_s', 'refractory_s', 'excitatory_reversal_mV', 'inhibitory_reversal_mV')], target, rate))
    return (target, rate)

def _GpuRetinalTransductionBrain(self, state, drive, light, *, frame):
    import cupy as cp
    size = self.retinal_parent_state_size
    (target, rate) = _GpuT4GabaBrain(self, state[:size], drive, light, frame=frame)
    if not self.retinal_transduction_manifest['enabled']:
        return (frame.append('target', target, state[size:]), frame.append('rate', rate, cp.zeros(len(state) - size)))
    (c, r, p) = (self.cuda, self._retinal_cuda, _m9.PARAMETERS)
    (available, bump) = state[size:].reshape(2, -1)
    (at, ar, bt) = _m9.coefficients(available, light[r['pp']])
    k = len(available)
    self._retinal_kernel(((k * 32 + 255) // 256,), (256,), (_m9.np.int32(k), r['rows'], r['ptr'], r['lai'], c['indptr'], c['indices'], c['weights'], state[self.transmission_start:self.inherited_state_size], bump, c['tau'], _m9.np.float64(self.parameters['conductance_per_stored_weight']), _m9.np.float64(p['maximum_relative_photoconductance']), _m9.np.float64((p['photo_reversal_mv'] + 80.0) / 80.0), target, rate))
    return (frame.append('target', target, at, bt), frame.append('rate', rate, ar, cp.full(k, 1.0 / p['bump_filter_tau_s'])))

def _GpuT4GabaBrain(self, state, drive, light, *, frame):
    (target, rate) = _GpuKcGammaRegionalBrain(self, state, drive, light, frame=frame)
    if self.t4_gaba_manifest['enabled']:
        (c, g) = (self.cuda, self._gaba_cuda)
        n = len(self._gaba_rows)
        self._gaba_kernel(((n * 32 + 255) // 256,), (256,), (_m10.np.int32(n), g['rows'], g['ptr'], g['positions'], g['pres'], c['weights'], state[self.transmission_start:self.inherited_state_size], _m10.np.float64(self.parameters['conductance_per_stored_weight']), _m10.np.float64((_m10.E_GABA_MV + 80.0) / 80.0), c['tau'], target, rate))
    return (target, rate)

def _GpuKcGammaRegionalBrain(self, state, drive, light, *, frame):
    import cupy as cp
    size = self.regional_parent_state_size
    m = self.regional_manifest
    (target, rate) = _GpuProstheticOlfactoryBrain(self, state[:size], drive, light, frame=frame)
    if not m['enabled']:
        return (frame.append('target', target, state[size:]), frame.append('rate', rate, cp.zeros(len(state) - size)))
    r = self._regional_cuda
    c = self.cuda
    p = m['parameters']
    k = len(self.regional_rows)
    (x, b, sax) = state[size:].reshape(3, -1)
    n = len(self._regional_cache['rows'])
    if n:
        args = (_m11.np.int32(n), r['rows'], r['local_ptr'], r['mode'], r['fraction'], r['slot'], c['indptr'], c['indices'], c['weights'], state[self.transmission_start:self.inherited_state_size], sax, c['caps'], c['visual'], c['gain'], c['theta'], drive, _m11.np.bool_(self.visual_output_connected), target)
        self._regional_current_kernel(((n * 32 + 255) // 256,), (256,), args)
    new = cp.empty(3 * k, dtype=cp.float64)
    args = (_m11.np.int32(k), r['kc_rows'], r['axon_ptr'], r['axon_pre'], r['axon_contacts'], c['caps'], state, x, b, sax, _m11.np.float64(p['K_ach_contact_hz']), _m11.np.float64(p['eta']), _m11.np.float64(p['x50']), _m11.np.float64(p['slope']), _m11.np.bool_(m['activity_dependent']), new)
    self._regional_axon_kernel(((k * 32 + 255) // 256,), (256,), args)
    return (frame.append('target', target, new), frame.append('rate', rate, r['rates']))

def _GpuProstheticOlfactoryBrain(self, state, drive, light, *, frame):
    import cupy as cp
    (size, o) = (self.parent_state_size, self._orn_pn_cuda)
    (target, rate) = _GpuPvlpAdaptationBrain(self, state[:size], drive, light, frame=frame)
    (af, ass, zf, zs) = state[size:].reshape(4, -1)
    q = o['held_rate'] / o['caps']
    rf = 1.0 / _m12.FAST.tau_A_s + _m12.FAST.r_per_spike * o['held_rate']
    rs = 1.0 / _m12.SLOW.tau_A_s + _m12.SLOW.r_per_spike * o['held_rate']
    if self.orn_synaptic_enabled:
        (c, k) = (self.cuda, len(self._pn_rows))
        args = (_m12.np.int32(k), o['pn_rows'], o['ptr'], o['slots'], c['indptr'], c['indices'], c['weights'], state[self.transmission_start:self.inherited_state_size], c['caps'], c['visual'], c['gain'], c['theta'], drive, _m12.np.bool_(self.visual_output_connected), zf, zs, _m12.np.float64(_m12.FAST_AREA), _m12.np.float64(_m12.SLOW_AREA), _m12.np.float64(_m12.NORMALIZATION), target)
        self._orn_pn_kernel(((k * 32 + 255) // 256,), (256,), args)
    return (frame.append('target', target, 1.0 / _m12.FAST.tau_A_s / rf, 1.0 / _m12.SLOW.tau_A_s / rs, q * af, q * ass), frame.append('rate', rate, rf, rs, cp.full(len(q), 1.0 / _m12.FAST.tau_g_s, dtype=cp.float64), cp.full(len(q), 1.0 / _m12.SLOW.tau_g_s, dtype=cp.float64)))

def _GpuPvlpAdaptationBrain(self, state, drive, light, *, frame):
    import cupy as cp
    (size, m) = (self.inherited_state_size, self.pvlp_adaptation_manifest)
    (target, rate) = _GpuCyborgRetinalPortBrain(self, state[:size], drive, light, frame=frame)
    rows = self._pvlp_cuda_rows
    if self.adaptation_active:
        (c, k) = (self.cuda, len(rows))
        args = (_m13.np.int32(k), rows, c['indptr'], c['indices'], c['weights'], state[self.transmission_start:size], c['caps'], c['visual'], c['gain'], c['theta'], drive, _m13.np.bool_(self.visual_output_connected), _m13.np.float64(m['beta']), state[size:], target)
        self._pvlp_kernel(((k * 32 + 255) // 256,), (256,), args)
    return (frame.append('target', target, state[rows]), frame.append('rate', rate, cp.full(len(rows), 1.0 / m['tau_s'], dtype=cp.float64)))

def _GpuCyborgRetinalPortBrain(self, state, drive, light, *, frame):
    (target, rate) = _GpuMeasuredT4VisualBrain(self, state, drive, light, frame=frame)
    if self.cyborg_retinal_port_manifest['enabled']:
        _m14._replace_coefficients(target, rate, self._retinal_port_cuda_rows, self._retinal_port_cuda_photo_positions, light)
    return (target, rate)

def _GpuMeasuredT4VisualBrain(self, state, drive, light, *, frame):
    (target, rate) = _GpuSynapticVisualBrain(self, state, drive, light, frame=frame)
    if self._mi9_alpha != 0.0:
        (c, selected) = (self.cuda, self._mi9_cuda)
        nrows = len(self._mi9_rows)
        args = (_m15.np.int32(nrows), selected['rows'], selected['indptr'], selected['positions'], selected['pres'], c['weights'], state[self.transmission_start:], _m15.np.float64(self.parameters['conductance_per_stored_weight']), _m15.np.float64(self._mi9_alpha), c['tau'], target, rate)
        self._mi9_kernel(((nrows * 32 + 255) // 256,), (256,), args)
    return (target, rate)

def _GpuSynapticVisualBrain(self, state, drive, light, *, frame):
    self.statistics['evaluations'] += 1
    (n, m) = (self.brain.n_neurons, len(self.pi))
    (p, c) = (self.parameters, self.cuda)
    released = state[:n].copy()
    released[c['vi']] = _m16.cp.clip((80.0 * released[c['vi']] - 15.0) / 40.0, 0.0, 1.0)
    (fast, adaptation) = (state[n:n + m], state[n + m:n + 2 * m])
    photo = _m16.cp.zeros(n, dtype=_m16.cp.float64)
    photo[c['pi']] = p['photoconductance_max'] * fast / (p['photo_half'] + p['adaptation_strength'] * adaptation + fast)
    (target, rate) = (frame.base('target', len(state)), frame.base('rate', len(state)))
    args = (_m16.np.int32(n), c['indptr'], c['indices'], c['weights'], state[self.transmission_start:], c['caps'], c['visual'], c['tau'], c['gain'], c['theta'], drive, photo, _m16.np.float64(p['conductance_per_stored_weight']), _m16.np.bool_(self.visual_output_connected), target, rate)
    self.kernel(((n * 32 + 255) // 256,), (256,), args)
    (target[n:n + m], rate[n:n + m]) = (light, 1.0 / p['photo_fast_tau_s'])
    (target[n + m:n + 2 * m], rate[n + m:n + 2 * m]) = (fast, 1.0 / p['photo_adaptation_tau_s'])
    target[self.transmission_start:] = released
    rate[self.transmission_start:] = 1.0 / p['synaptic_tau_s']
    return (target, rate)

assemble = _OrnPeripheralTerminalMixin
