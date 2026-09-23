"""Conductance-to-rate prosthesis for the two canonical CvN7 cells.

The complete retinal CNS state is retained, including the two normalized
rate coordinates. This is a quasi-stationary LIF transfer and an inherited
rate filter, not membrane-voltage integration or generated spike events.
Rest/Rin are midpoints of reported CvN7 ranges. Missing electrical parameters
remain declared model priors; anatomical contacts are not measured nS.
"""
import copy
import math

import numba
import numpy as np

from retinal_transduction_brain import (
    RetinalTransductionBrain, GpuRetinalTransductionBrain, RETINAL_KEYS)
from kcgamma_regional_brain import _record_hash
from synaptic_visual_brain import _hash_array


POLICY = 'cvn7_quasistationary_conductance_lif_rate_prosthesis_v1'
CVN_KEYS = RETINAL_KEYS | {'cvn7_manifest'}
TARGET_IDS = np.array([10754, 556449], dtype=np.int64)
PARAMETERS = dict(rest_mV=-50., reset_mV=-50., input_resistance_MOhm=175.,
    membrane_tau_s=.020, threshold_mV=-45., refractory_s=.0022,
    excitatory_reversal_mV=0., inhibitory_reversal_mV=-68.)


def provenance():
    """Embedded receipts: restoring a checkpoint needs no external data file."""
    return dict(
        cvn7=dict(doi='10.1038/s41586-024-07222-5',
            article_url='https://www.nature.com/articles/s41586-024-07222-5',
            supplement_url='https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07222-5/MediaObjects/41586_2024_7222_MOESM1_ESM.pdf',
            supplement_sha256='0d79fcef0205b2eb27cd0f1e205862ecebea7610fb5bbb43fa6421c370bdc7d2',
            local_text='evidence/vision_empirical_20260908/cvn7/gorko_supplementary_information.txt',
            local_text_sha256='86d37d3c35ac63c88e625ec3883f76a643b4e2c91be55e228f0ebe170abdc46d',
            local_relevant_lines=[1063, 1089],
            local_physiology_receipt='work/cvn7_conductance_20260910/physiology.json',
            local_voltage_reference='evidence/circuit_integration_20260908/neck/cvn7_empirical_voltage_reference.npz',
            local_voltage_summary='evidence/circuit_integration_20260908/neck/cvn7_ephys_summary.json',
            reported_rest_mV=[-55., -45.], reported_Rin_MOhm=[150., 200.],
            scope='Somatic current-clamp in female CvN7 at22C, no holding current; current injection elicits spikes. Transfer to two MaleCNS identities uses range midpoints.',
            limitation='Rin includes morphology and active/resting synaptic contributions; its inverse is an effective input-conductance scale, not measured isolated leak.'),
        lif_priors=dict(doi='10.1038/s41586-024-07763-9',
            url='https://www.nature.com/articles/s41586-024-07763-9',
            code_url='https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py#L19-L28',
            code_sha256='fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e',
            local_code='data/model_assets_20260909/shiu_reference/source/model.py',
            scope='Shiu2024 model priors: membrane tau20ms, threshold-45mV, refractory2.2ms; not measurements of CvN7. Reset-50mV equals the selected resting-range midpoint.'),
        inhibitory_reversal=dict(doi='10.1038/s41586-022-04428-3',
            url='https://www.nature.com/articles/s41586-022-04428-3',
            scope='The T4/Groschner equivalent GABA reversal-68mV is transferred as an inhibitory prior to all negative CvN7 inputs, including glutamate. This does not identify their receptors or establish their true reversals.'),
        shared_hypotheses=[
            'Existing weight signs are retained; positive weights use0mV and negative weights use-68mV. No transmitter label overrides a stored sign.',
            'abs(weight)*inherited conductance_per_stored_weight*filtered normalized transmission is a conductance relative to effective resting input conductance; no presynaptic rate cap factor.',
            'All incoming chemical pairs remain; per-contact conductance and common transmission kinetics are unmeasured CvN7 hypotheses.',
            'Rin sets nS units and derived C_pF=tau_s*1e6/Rin_MOhm. Rin cancels from Vinf and firing-transfer predictions.',
            'No Vm or spike-event state is added. Stationary LIF frequency is divided by the inherited target cap and clipped to[0,1].',
            'The inherited target-neuron tau remains the output-rate low-pass time constant; it is distinct from the20ms membrane prior.',
            'Nonzero direct current has no supported conversion into this conductance model and must fail.',
            'No behavioral feedback, desired movement, parameter fitting or observed voltage trace is supplied to the neuron.'])


@numba.njit(cache=True, fastmath=False)
def cvn7_values(rows, ptr, idx, weights, transmission, caps, visual,
                scale, visual_connected, drive, rest, reset, threshold,
                membrane_tau, refractory, e_exc, e_inh):
    """CPU reference in CSR order; returns gE,gI,Vinf,tau_eff,f_inf,qtarget."""
    result = np.empty((len(rows), 6), dtype=np.float64)
    for j in range(len(rows)):
        row = rows[j]
        if not np.isfinite(drive[row]) or drive[row] != 0.:
            raise ValueError('CvN7 direct drive has no supported current-unit conversion')
        exc, inh = 0., 0.
        for edge in range(ptr[row], ptr[row+1]):
            pre = idx[edge]
            if not visual_connected and visual[pre]:
                continue
            value = (weights[edge]*scale)*transmission[pre]
            if value >= 0.:
                exc += value
            else:
                inh -= value
        total = 1.+exc+inh
        v_inf = (rest+exc*e_exc+inh*e_inh)/total
        tau_eff = membrane_tau/total
        f_inf = 0.
        if v_inf > threshold:
            f_inf = 1./(refractory+tau_eff*math.log((v_inf-reset)/(v_inf-threshold)))
        result[j, 0], result[j, 1], result[j, 2] = exc, inh, v_inf
        result[j, 3], result[j, 4] = tau_eff, f_inf
        result[j, 5] = min(f_inf/caps[row], 1.)
    return result


CUDA_SOURCE = r'''
extern "C" __global__ void cvn7_transfer(
 int k,const long long* rows,const long long* ptr,const int* idx,
 const double* w,const double* s,const double* caps,const bool* visual,
 const double* tau,double scale,bool connected,const double* drive,
 double rest,double reset,double threshold,double tau_m,double refractory,
 double e_exc,double e_inh,double* target,double* rate){
 int j=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
 if(j>=k)return;
 long long row=rows[j];double exc=0.,inh=0.;
 for(long long e=ptr[row]+lane;e<ptr[row+1];e+=32){
  int pre=idx[e];if(!connected&&visual[pre])continue;
  double v=(w[e]*scale)*s[pre];if(v>=0.)exc+=v;else inh-=v;
 }
 for(int d=16;d>0;d/=2){
  exc+=__shfl_down_sync(0xffffffff,exc,d);
  inh+=__shfl_down_sync(0xffffffff,inh,d);
 }
 if(lane==0){
  if(!isfinite(drive[row])||drive[row]!=0.){
   target[row]=nan("");rate[row]=nan("");return;
  }
  double total=1.+exc+inh,vinf=(rest+exc*e_exc+inh*e_inh)/total;
  double effective_tau=tau_m/total,f=0.;
  if(vinf>threshold)
   f=1./(refractory+effective_tau*log((vinf-reset)/(vinf-threshold)));
  target[row]=fmin(f/caps[row],1.);rate[row]=1./tau[row];
 }
}
'''


def _selection(reference):
    b = reference.brain
    rows = np.searchsorted(b.node_ids, TARGET_IDS)
    if (b.n_neurons != 166700 or len(reference.state) != 355470
            or np.any(rows >= b.n_neurons)
            or not np.array_equal(b.node_ids[rows], TARGET_IDS)
            or not np.all(reference.measured_t4_manifest['canonical_node_types'][rows] == 'CvN7')
            or np.any(reference.visual_mask[rows])):
        raise ValueError('Requires both canonical rate-mode CvN7 in the complete retinal CNS')
    for name, key in (('pvlp_adaptation_manifest', 'target_rows'),
                      ('orn_pn_synaptic_manifest', 'post_indices'),
                      ('cyborg_retinal_port_manifest', 'target_rows'),
                      ('retinal_transduction_manifest', 'target_rows'),
                      ('regional_manifest', 'target_rows')):
        manifest = getattr(reference, name, {})
        if np.intersect1d(rows, manifest.get(key, [])).size:
            raise ValueError('CvN7 overlaps an existing local target replacement: '+name)
    positions = np.concatenate([np.arange(b.W.indptr[r], b.W.indptr[r+1], dtype=np.int64)
                                for r in rows])
    if (not len(positions) or not np.isfinite(reference.weights64[positions]).all()
            or not np.isfinite(reference.caps[rows]).all() or np.any(reference.caps[rows] <= 0.)
            or not np.isfinite(reference.tau[rows]).all() or np.any(reference.tau[rows] <= 0.)):
        raise ValueError('Invalid CvN7 incoming weights, output caps or inherited rate filter')
    return rows.astype(np.int64), positions


def _values(hybrid, state, drive):
    p, b = PARAMETERS, hybrid.brain
    return cvn7_values(hybrid._cvn7_rows, b.W.indptr, b.W.indices,
        hybrid.weights64, state[hybrid.transmission_start:hybrid.inherited_state_size],
        hybrid.caps, hybrid.visual_mask,
        hybrid.parameters['conductance_per_stored_weight'], hybrid.visual_output_connected,
        drive, p['rest_mV'], p['reset_mV'], p['threshold_mV'], p['membrane_tau_s'],
        p['refractory_s'], p['excitatory_reversal_mV'], p['inhibitory_reversal_mV'])


class Cvn7ConductanceBrain(RetinalTransductionBrain):
    SCHEMA = 'matrix_cvn7_conductance_brain_fp64_v1'
    CVN_PARENT_CLASS = RetinalTransductionBrain

    @classmethod
    def adopt(cls, reference, *, enabled=True):
        if type(enabled) is not bool or reference.SCHEMA != cls.CVN_PARENT_CLASS.SCHEMA:
            raise ValueError('CvN7 adoption requires the exact retinal parent and boolean enabled')
        rows, positions = _selection(reference)
        saved = reference.state_dict()
        inherited = {key: value for key, value in saved.items() if key != 'schema'}
        m = dict(policy=POLICY, enabled=enabled, parameters=copy.deepcopy(PARAMETERS),
            parameter_status='Quasi-stationary LIF conductance-to-rate prosthesis. CvN7 rest/Rin range midpoints; other electrical quantities remain transferred or inherited model priors.',
            source_schema=reference.SCHEMA, adoption_time_ns=int(reference.time_ns),
            state_size=len(reference.state), new_states=0,
            initial_state_sha256=_hash_array(reference.state),
            initial_inherited_metadata_sha256=_record_hash(inherited),
            target_rows=rows, target_ids=TARGET_IDS.copy(),
            incoming_positions=positions, incoming_pairs=len(positions),
            selected_stored_weights_sha256=_hash_array(reference.brain.W.data[positions]),
            selected_effective_weights_sha256=_hash_array(reference.weights64[positions]),
            inherited_output_tau_s=reference.tau[rows].copy(),
            inherited_output_caps=reference.caps[rows].copy(),
            conductance_per_stored_weight=reference.parameters['conductance_per_stored_weight'],
            node_ids_sha256=_hash_array(reference.brain.node_ids),
            canonical_types_sha256=_hash_array(reference.measured_t4_manifest['canonical_node_types']),
            anatomical_indptr_sha256=_hash_array(reference.brain.W.indptr),
            anatomical_indices_sha256=_hash_array(reference.brain.W.indices),
            retinal_manifest_sha256=_record_hash(reference.retinal_transduction_manifest),
            provenance=provenance(),
            equation='gE/gI=sum(abs(weight)*scale*s_pre) by inherited sign; Vinf=(Erest+gE*Eexc+gI*Einh)/(1+gE+gI); tau_eff=20ms/(1+gE+gI); f_inf=stationary_LIF(Vinf,tau_eff); qtarget=min(f_inf/cap_target,1); dq/dt=(qtarget-q)/inherited_tau_target.',
            state_contract='Unchanged normalized rate coordinates and transmission states; no Vm state or spike events. All inherited CNS state and anatomy persist.',
            bypass='When disabled return parent coefficients exactly; unchanged state, solver norm, visual gate and prior parameters.',
            direct_drive_policy='Nonzero or nonfinite CvN7 direct drive is unsupported and rejected; no invented model-current to pA conversion.',
            parameter_fitting=False, biological_validation=False)
        m['record_sha256'] = _record_hash(m)
        saved.update(schema=cls.SCHEMA, cvn7_manifest=m)
        return cls.from_state(reference.brain, saved)

    def _build_cvn7_cache(self):
        m, b = self.cvn7_manifest, self.brain
        rows, positions = _selection(self)
        if (m.get('policy') != POLICY or type(m.get('enabled')) is not bool
                or m.get('record_sha256') != _record_hash(m)
                or m.get('parameters') != PARAMETERS or m.get('provenance') != provenance()
                or m.get('source_schema') != self.CVN_PARENT_CLASS.SCHEMA
                or m.get('new_states') != 0 or m.get('state_size') != len(self.state)
                or m.get('biological_validation') is not False or m.get('parameter_fitting') is not False
                or type(m.get('adoption_time_ns')) is not int
                or not 0 <= m['adoption_time_ns'] <= self.time_ns
                or m.get('conductance_per_stored_weight') != self.parameters['conductance_per_stored_weight']
                or not np.isfinite(m['conductance_per_stored_weight'])
                or m['conductance_per_stored_weight'] <= 0.):
            raise ValueError('Changed CvN7 prosthesis, provenance, fixed parameters or adoption clock')
        for key, value in [('target_rows', rows), ('target_ids', TARGET_IDS),
                           ('incoming_positions', positions),
                           ('inherited_output_tau_s', self.tau[rows]),
                           ('inherited_output_caps', self.caps[rows])]:
            if not np.array_equal(m.get(key), value):
                raise ValueError('Changed CvN7 mapping or inherited rate contract: '+key)
        for key, value in [('node_ids', b.node_ids),
                           ('canonical_types', self.measured_t4_manifest['canonical_node_types']),
                           ('anatomical_indptr', b.W.indptr), ('anatomical_indices', b.W.indices),
                           ('selected_stored_weights', b.W.data[positions]),
                           ('selected_effective_weights', self.weights64[positions])]:
            if m.get(key+'_sha256') != _hash_array(value):
                raise ValueError('Changed CvN7 canonical data: '+key)
        if (m['incoming_pairs'] != len(positions)
                or m['retinal_manifest_sha256'] != _record_hash(self.retinal_transduction_manifest)):
            raise ValueError('CvN7 adoption differs from the inherited retinal component')
        if self.time_ns == m['adoption_time_ns']:
            inherited = {k: v for k, v in self.state_dict().items()
                         if k not in ('schema', 'cvn7_manifest')}
            if (m['initial_state_sha256'] != _hash_array(self.state)
                    or m['initial_inherited_metadata_sha256'] != _record_hash(inherited)):
                raise ValueError('CvN7 adoption must preserve all inherited state and metadata')
        self._cvn7_rows, self._cvn7_positions = rows, positions

    def _coefficients(self, state, drive, light):
        if self.cvn7_manifest['enabled'] and np.any(drive[self._cvn7_rows] != 0.):
            raise ValueError('CvN7 direct drive has no supported current-unit conversion')
        target, rate = RetinalTransductionBrain._coefficients(self, state, drive, light)
        if self.cvn7_manifest['enabled']:
            values = _values(self, state, drive)
            target[self._cvn7_rows] = values[:, 5]
            rate[self._cvn7_rows] = 1./self.tau[self._cvn7_rows]
        return target, rate

    def state_dict(self):
        saved = super().state_dict()
        saved['cvn7_manifest'] = copy.deepcopy(self.cvn7_manifest)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != CVN_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete CvN7 retinal CNS state or wrong backend')
        parent = {k: v for k, v in saved.items() if k != 'cvn7_manifest'}
        parent['schema'] = cls.CVN_PARENT_CLASS.SCHEMA
        base = cls.CVN_PARENT_CLASS.from_state(brain, parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.cvn7_manifest = copy.deepcopy(saved['cvn7_manifest'])
        obj._build_cvn7_cache()
        return obj


class GpuCvn7ConductanceBrain(Cvn7ConductanceBrain, GpuRetinalTransductionBrain):
    SCHEMA = 'matrix_cvn7_conductance_brain_fp64_cuda_v1'
    CVN_PARENT_CLASS = GpuRetinalTransductionBrain

    def _build_cvn7_cache(self):
        super()._build_cvn7_cache()
        import cupy as cp
        self._cvn7_cuda_rows = cp.asarray(self._cvn7_rows)
        self._cvn7_kernel = cp.RawKernel(CUDA_SOURCE, 'cvn7_transfer',
            options=('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'))

    def coefficients_gpu(self, state, drive, light):
        target, rate = GpuRetinalTransductionBrain.coefficients_gpu(self, state, drive, light)
        if self.cvn7_manifest['enabled']:
            c, p = self.cuda, PARAMETERS
            self._cvn7_kernel((1,), (256,), (np.int32(len(self._cvn7_rows)),
                self._cvn7_cuda_rows, c['indptr'], c['indices'], c['weights'],
                state[self.transmission_start:self.inherited_state_size], c['caps'],
                c['visual'], c['tau'], np.float64(self.parameters['conductance_per_stored_weight']),
                np.bool_(self.visual_output_connected), drive,
                *[np.float64(p[key]) for key in ('rest_mV', 'reset_mV', 'threshold_mV',
                    'membrane_tau_s', 'refractory_s', 'excitatory_reversal_mV',
                    'inhibitory_reversal_mV')], target, rate))
        return target, rate

    @staticmethod
    def backend_identity():
        result = GpuRetinalTransductionBrain.backend_identity()
        result['cvn7_conductance'] = 'Two fixed CvN7 rows: quasi-stationary LIF conductance-to-rate prosthesis; existing rate filters/state retained; fixed FP64 warp reduction.'
        return result


def observe_cvn7(hybrid, drive=None):
    """Read the new transfer; None explicitly assumes zero external drive.

    No advance or GPU evaluation occurs. CPU CSR sums may differ from CUDA
    warp sums by roundoff. Disabled observations expose the bypassed candidate
    values, never label them as the coefficients applied by the parent.
    """
    if not isinstance(hybrid, Cvn7ConductanceBrain):
        raise TypeError('CvN7 observation requires the new conductance component')
    b, p = hybrid.brain, PARAMETERS
    assumed_zero = drive is None
    drive = np.zeros(b.n_neurons, dtype=np.float64) if assumed_zero else np.asarray(drive, dtype=np.float64)
    if drive.shape != (b.n_neurons,) or not np.isfinite(drive).all():
        raise ValueError('CvN7 observer requires a finite canonical direct-drive vector')
    values = _values(hybrid, hybrid.state, drive)
    transmission = hybrid.transmission_release()
    q = hybrid.release()
    g_scale_nS = 1000./p['input_resistance_MOhm']
    motors = []
    for row, data in zip(hybrid._cvn7_rows, values):
        pos = np.arange(b.W.indptr[row], b.W.indptr[row+1])
        pres = b.W.indices[pos]
        included = np.ones(len(pos), dtype=bool) if hybrid.visual_output_connected else ~hybrid.visual_mask[pres]
        exc, inh, v_inf, tau_eff, f_inf, q_target = data
        motors.append(dict(body_id=int(b.node_ids[row]), canonical_row=int(row), type='CvN7',
            g_exc_relative=float(exc), g_inh_relative=float(inh),
            g_exc_nS=float(exc*g_scale_nS), g_inh_nS=float(inh*g_scale_nS),
            V_inf_mV=float(v_inf), tau_eff_s=float(tau_eff), f_inf_Hz=float(f_inf),
            q_target=float(q_target), q=float(q[row]), transmission_s=float(transmission[row]),
            output_rate_model_Hz=float(q[row]*hybrid.caps[row]),
            rate_filter_tau_s=float(hybrid.tau[row]), output_cap_Hz=float(hybrid.caps[row]),
            incoming_pairs=len(pos), included_pairs=int(included.sum()),
            blocked_visual_pairs=int((~included).sum()),
            stored_positive_pairs=int(np.count_nonzero(b.W.data[pos] > 0.)),
            stored_negative_pairs=int(np.count_nonzero(b.W.data[pos] < 0.)),
            direct_drive=float(drive[row])))
    return dict(schema='matrix_cvn7_conductance_observation_v1', time_ns=int(hybrid.time_ns),
        applied=hybrid.cvn7_manifest['enabled'],
        scope='Current quasi-stationary transfer when enabled; counterfactual candidate when bypassed. No simulated membrane or observed spike events.',
        direct_drive_assumption='zero explicitly assumed by caller omission' if assumed_zero else 'caller supplied full held vector',
        summation='FP64 CPU CSR order; CUDA warp order may differ by roundoff',
        visual_output_connected=hybrid.visual_output_connected,
        effective_resting_conductance_scale_nS=g_scale_nS,
        derived_capacitance_pF=p['membrane_tau_s']*1e6/p['input_resistance_MOhm'],
        capacitance_status='Derived from the20ms prior and effective Rin; not measured CvN7 capacitance. Rin cancels from transfer predictions.',
        parameters=copy.deepcopy(p), biological_validation=False, motors=motors)
