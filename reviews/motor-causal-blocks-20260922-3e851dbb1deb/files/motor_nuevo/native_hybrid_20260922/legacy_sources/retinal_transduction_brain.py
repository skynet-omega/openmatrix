"""Local retinal transduction and restored chemical feedback in the same CNS.

Appends two states per mapped R1-R6 and supersedes its old voltage clamp. The
original voltage, all transmission states, regional Kenyon states, graph and
weights persist. Lai glutamate is interpreted as excitatory only at these
R1-R6 terminals (Hu2015 EKAR); other glutamate pathways retain their polarity.
"""
import copy

import numba
import numpy as np

from t4_gaba_brain import T4GabaBrain, GpuT4GabaBrain, GABA_KEYS
from kcgamma_regional_brain import _record_hash, _axonal_input
from synaptic_visual_brain import _hash_array
from retinal_phototransduction import PARAMETERS, STATE_ORDER, initial_state, coefficients, provenance

POLICY = 'recurrent_retina_microvillus_population_v1'
RETINAL_KEYS = GABA_KEYS | {'retinal_transduction_manifest'}


@numba.njit(cache=True, fastmath=False)
def retinal_voltage(rows, local_ptr, lai, ptr, idx, w, transmission,
                    bump, tau, scale, gmax, ephoto_norm, target, rate):
    for j in range(len(rows)):
        row = rows[j]
        exc, inh = 0., 0.
        for edge in range(ptr[row], ptr[row+1]):
            value = (w[edge]*scale)*transmission[idx[edge]]
            if lai[local_ptr[j]+edge-ptr[row]]:
                value = abs(value)
            if value >= 0.:
                exc += value
            else:
                inh -= value
        photo = gmax*bump[j]
        total = 1.+exc+inh+photo
        target[row] = (.25+exc+ephoto_norm*photo)/total
        rate[row] = total/tau[row]


CUDA_SOURCE = r'''
extern "C" __global__ void retinal_voltage(
 int k,const long long* rows,const long long* local_ptr,const bool* lai,
 const long long* ptr,const int* idx,const double* w,const double* s,
 const double* bump,const double* tau,double scale,double gmax,double ephoto,
 double* target,double* rate){
 int j=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(j>=k)return;
 long long row=rows[j];double exc=0.,inh=0.;
 for(long long e=ptr[row]+lane;e<ptr[row+1];e+=32){
  double v=(w[e]*scale)*s[idx[e]];
  if(lai[local_ptr[j]+e-ptr[row]])v=fabs(v);
  if(v>=0.)exc+=v;else inh-=v;
 }
 for(int d=16;d>0;d/=2){exc+=__shfl_down_sync(0xffffffff,exc,d);inh+=__shfl_down_sync(0xffffffff,inh,d);}
 if(lane==0){double photo=gmax*bump[j],total=1.+exc+inh+photo;
  target[row]=(.25+exc+ephoto*photo)/total;rate[row]=total/tau[row];}
}
'''


def feedback_selection(brain, rows, types):
    parts = [np.arange(brain.W.indptr[row], brain.W.indptr[row+1], dtype=np.int64) for row in rows]
    positions = np.concatenate(parts)
    ptr = np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int64)
    lai = types[brain.W.indices[positions]] == 'Lai'
    if not lai.any() or np.any(brain.W.data[positions[lai]] >= 0.):
        raise ValueError('Expected canonical negative glutamate Lai afferents')
    if not np.all(np.char.lower(brain.nt_labels[brain.W.indices[positions[lai]]]) == 'glutamate'):
        raise ValueError('Lai transmitter annotation disagrees with EKAR interpretation')
    return positions, ptr, lai


class RetinalTransductionBrain(T4GabaBrain):
    SCHEMA = 'matrix_retinal_transduction_brain_fp64_v1'
    RETINAL_PARENT_CLASS = T4GabaBrain

    @classmethod
    def adopt(cls, reference, *, enabled=True, initial_light=None):
        if type(enabled) is not bool or reference.SCHEMA != cls.RETINAL_PARENT_CLASS.SCHEMA:
            raise ValueError('Retinal integration requires the current complete T4 GABA family')
        old = reference.cyborg_retinal_port_manifest
        if not old['enabled']:
            raise ValueError('This version replaces the enabled historical retinal voltage clamp')
        rows, pp = old['target_rows'], old['photo_positions']
        if initial_light is None:
            raise ValueError('Declare the current pending optical input for the new-history assumption')
        light = np.asarray(initial_light, dtype=np.float64)
        if light.shape != reference.photo_ids.shape:
            raise ValueError('Initial optical vector differs from inherited photo identities')
        new = initial_state(light[pp])
        saved = reference.state_dict()
        prefix = saved['state']
        types = reference.measured_t4_manifest['canonical_node_types']
        positions, ptr, lai = feedback_selection(reference.brain, rows, types)
        m = dict(policy=POLICY, enabled=enabled, source_schema=reference.SCHEMA,
            adoption_time_ns=int(reference.time_ns), parent_state_size=len(prefix),
            target_rows=rows.copy(), target_ids=old['target_ids'].copy(), photo_positions=pp.copy(),
            state_order=STATE_ORDER.copy(), parameters=copy.deepcopy(PARAMETERS),
            parameter_status='Biologically constrained population approximation with uncalibrated optical/conductance gain and bump filter.',
            initial_light=light[pp].copy(), initial_parent_state_sha256=_hash_array(prefix),
            initial_new_state_sha256=_hash_array(new),
            initialization='New A,z at stationary values for current pending light. Earlier microvillus history unknown. Existing voltage/transmission/body are unchanged.',
            incoming_positions=positions, incoming_pairs=len(positions), lai_positions=positions[lai],
            incoming_weights_sha256=_hash_array(reference.weights64[positions]),
            node_ids_sha256=_hash_array(reference.brain.node_ids),
            canonical_types_sha256=_hash_array(types),
            anatomical_indptr_sha256=_hash_array(reference.brain.W.indptr),
            anatomical_indices_sha256=_hash_array(reference.brain.W.indices),
            inherited_retinal_manifest_sha256=_record_hash(old),
            provenance=provenance(),
            lai_receptor=dict(doi='10.1371/journal.pbio.1002115',
                url='https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002115',
                mechanism='Amacrine glutamate excites photoreceptor EKAR; transfer AC identity to canonical Lai.',
                crosswalk_doi='10.1371/journal.pone.0002110',
                scope='Only existing Lai->mapped R1-R6 terminals; use magnitude of inherited weight as excitatory conductance. No stored weight changes.',
                limitation='Polarity supported; generic0mV reversal, per-contact conductance and5ms chemical filter remain model assumptions. No Ih channel reconstruction.'),
            voltage_equation='target=(0.25+g_exc+1.25*g_photo)/(1+g_exc+g_inh+g_photo); rate=(1+g_exc+g_inh+g_photo)/inherited_tau. All local canonical chemical inputs included.',
            supersedes='Historical R1-R6 voltage clamp and its old fast/adaptation contribution on selected rows only; old states remain as inactive historical coordinates there.',
            bypass='Parent coefficients and solver error norm exactly restored; new states frozen.',
            parameter_fitting=False, biological_validation=False)
        m['record_sha256'] = _record_hash(m)
        saved.update(schema=cls.SCHEMA, state=np.r_[prefix, new], retinal_transduction_manifest=m)
        return cls.from_state(reference.brain, saved)

    @property
    def retinal_parent_state_size(self):
        return int(self.retinal_transduction_manifest['parent_state_size'])

    @property
    def retinal_transduction_state(self):
        return self.state[self.retinal_parent_state_size:].reshape(2, -1).copy()

    @property
    def axonal_state(self):
        return self.state[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3, -1).copy()

    def regional_ach_input(self, state=None):
        y = self.state if state is None else np.asarray(state)
        if y.shape != self.state.shape or y.dtype != np.float64:
            raise ValueError('Regional observation requires complete retinal CNS state')
        c = self._regional_cache
        release = (y[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3, -1)[2]
                   if self.regional_manifest['enabled'] else y[self.transmission_start+c['kc_rows']])
        return _axonal_input(c['kc_rows'], c['axon_ptr'], c['axon_pre'], c['axon_contacts'], self.caps, release)

    def _norm_size(self):
        if self.retinal_transduction_manifest['enabled']:
            return len(self.state)
        if self.regional_manifest['enabled']:
            return self.retinal_parent_state_size
        if self.orn_synaptic_enabled:
            return self.regional_parent_state_size
        return self.parent_state_size if self.adaptation_active else self.inherited_state_size

    def _build_retinal_cache(self):
        m = self.retinal_transduction_manifest
        positions, ptr, lai = feedback_selection(self.brain, m['target_rows'], self.measured_t4_manifest['canonical_node_types'])
        if (not np.array_equal(positions, m['incoming_positions'])
                or not np.array_equal(positions[lai], m['lai_positions'])
                or m['incoming_pairs'] != len(positions)
                or m['incoming_weights_sha256'] != _hash_array(self.weights64[positions])):
            raise ValueError('Retinal feedback anatomy or inherited conductances changed')
        self._retinal_cache = dict(rows=m['target_rows'], pp=m['photo_positions'], ptr=ptr, lai=lai)

    def _coefficients(self, state, drive, light):
        size = self.retinal_parent_state_size
        target, rate = T4GabaBrain._coefficients(self, state[:size], drive, light)
        if not self.retinal_transduction_manifest['enabled']:
            return np.r_[target, state[size:]], np.r_[rate, np.zeros(len(state)-size)]
        c, p = self._retinal_cache, PARAMETERS
        available, bump = state[size:].reshape(2, -1)
        at, ar, bt = coefficients(available, light[c['pp']])
        retinal_voltage(c['rows'], c['ptr'], c['lai'], self.brain.W.indptr, self.brain.W.indices,
            self.weights64, state[self.transmission_start:self.inherited_state_size], bump, self.tau,
            self.parameters['conductance_per_stored_weight'], p['maximum_relative_photoconductance'],
            (p['photo_reversal_mv']+80.)/80., target, rate)
        return np.r_[target, at, bt], np.r_[rate, ar, np.full(len(bt), 1./p['bump_filter_tau_s'])]

    def state_dict(self):
        saved = super().state_dict()
        saved['retinal_transduction_manifest'] = copy.deepcopy(self.retinal_transduction_manifest)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != RETINAL_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete retinal CNS state or wrong backend')
        m, state = saved['retinal_transduction_manifest'], saved['state']
        size = m['parent_state_size']
        old = saved['cyborg_retinal_port_manifest']
        if (m.get('policy') != POLICY or type(m.get('enabled')) is not bool
                or m.get('record_sha256') != _record_hash(m) or m.get('parameters') != PARAMETERS
                or m.get('state_order') != STATE_ORDER or m.get('source_schema') != cls.RETINAL_PARENT_CLASS.SCHEMA
                or m.get('biological_validation') is not False or not old['enabled']
                or type(size) is not int or size <= 0 or type(m.get('adoption_time_ns')) is not int
                or not 0 <= m['adoption_time_ns'] <= saved['time_ns']
                or not isinstance(state, np.ndarray) or state.dtype != np.float64
                or state.shape != (size+2*len(old['target_ids']),)
                or not np.isfinite(state).all() or np.any((state < 0.) | (state > 1.))):
            raise ValueError('Invalid retinal model, state layout or history')
        for key in ('target_rows', 'target_ids', 'photo_positions'):
            if not np.array_equal(m[key], old[key]):
                raise ValueError('Retinal mapping differs from inherited optical identities')
        parent = {k: v for k, v in saved.items() if k != 'retinal_transduction_manifest'}
        parent.update(schema=cls.RETINAL_PARENT_CLASS.SCHEMA, state=state[:size].copy())
        base = cls.RETINAL_PARENT_CLASS.from_state(brain, parent)
        for name, array in [('node_ids', brain.node_ids), ('canonical_types', base.measured_t4_manifest['canonical_node_types']),
                            ('anatomical_indptr', brain.W.indptr), ('anatomical_indices', brain.W.indices)]:
            if m[name+'_sha256'] != _hash_array(array):
                raise ValueError('Changed retinal canonical identity: '+name)
        if m['inherited_retinal_manifest_sha256'] != _record_hash(old):
            raise ValueError('Historical retinal device provenance changed')
        if saved['time_ns'] == m['adoption_time_ns']:
            new = initial_state(m['initial_light'])
            if (m['initial_parent_state_sha256'] != _hash_array(state[:size])
                    or m['initial_new_state_sha256'] != _hash_array(new)
                    or not np.array_equal(state[size:], new)):
                raise ValueError('Retinal adoption changed prior state or new-history assumption')
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.state = state.copy()
        obj.retinal_transduction_manifest = copy.deepcopy(m)
        obj._build_retinal_cache()
        return obj


class GpuRetinalTransductionBrain(RetinalTransductionBrain, GpuT4GabaBrain):
    SCHEMA = 'matrix_retinal_transduction_brain_fp64_cuda_v1'
    RETINAL_PARENT_CLASS = GpuT4GabaBrain

    def _build_retinal_cache(self):
        super()._build_retinal_cache()
        import cupy as cp
        self._retinal_cuda = {k: cp.asarray(v) for k, v in self._retinal_cache.items()}
        self._retinal_kernel = cp.RawKernel(CUDA_SOURCE, 'retinal_voltage',
            options=('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'))

    def coefficients_gpu(self, state, drive, light):
        import cupy as cp
        size = self.retinal_parent_state_size
        target, rate = GpuT4GabaBrain.coefficients_gpu(self, state[:size], drive, light)
        if not self.retinal_transduction_manifest['enabled']:
            return cp.concatenate((target, state[size:])), cp.concatenate((rate, cp.zeros(len(state)-size)))
        c, r, p = self.cuda, self._retinal_cuda, PARAMETERS
        available, bump = state[size:].reshape(2, -1)
        at, ar, bt = coefficients(available, light[r['pp']])
        k = len(available)
        self._retinal_kernel(((k*32+255)//256,), (256,),
            (np.int32(k), r['rows'], r['ptr'], r['lai'], c['indptr'], c['indices'], c['weights'],
             state[self.transmission_start:self.inherited_state_size], bump, c['tau'],
             np.float64(self.parameters['conductance_per_stored_weight']),
             np.float64(p['maximum_relative_photoconductance']), np.float64((p['photo_reversal_mv']+80.)/80.), target, rate))
        return cp.concatenate((target, at, bt)), cp.concatenate((rate, ar, cp.full(k, 1./p['bump_filter_tau_s'])))

    @staticmethod
    def backend_identity():
        info = GpuT4GabaBrain.backend_identity()
        info['retinal_transduction'] = 'Two stage-consistent FP64 microvillus/filter states per mapped R1-R6; canonical feedback with Lai EKAR polarity; optical gain remains prosthetic.'
        return info
