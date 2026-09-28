"""Published GABA reversal on existing Mi4/C3 -> T4 pairs in the live CNS.

The -68 mV equivalent reversal is transferred from Groschner2022's model,
supported by somatic voltage-clamp experiments. It is not a receptor subtype
assignment or a measurement of this male connectome. All inherited states,
weights, total conductances, kinetics and regional KC dynamics remain intact.
"""
from pathlib import Path
import copy
import csv
import hashlib
import json

import numpy as np

from kcgamma_regional_brain import (
    KcGammaRegionalBrain, GpuKcGammaRegionalBrain, REGIONAL_KEYS, _record_hash)
from measured_t4_visual_brain import _correct_targets
from synaptic_visual_brain import _hash_array

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'published_mi4_c3_t4_gaba_reversal_v1'
E_GABA_MV = -68.0
POST_TYPES = ('T4a', 'T4b', 'T4c', 'T4d')
PRE_TYPES = ('Mi4', 'C3')
GABA_KEYS = REGIONAL_KEYS | {'t4_gaba_manifest'}


def constraints():
    folder = ROOT/'data/visual_physiology_targets/v0.1'
    source = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    path = folder/'fig2_iv_cells.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if (digest != source['files'][path.name]
            or source['source_article_doi'] != '10.1038/s41586-022-04428-3'):
        raise ValueError('Changed source of the electrical constraints')
    with path.open(newline='', encoding='utf-8') as stream:
        rows = [dict(cell_id=r['cell_id'], historical_role=r['role'],
                     slope_nS=float(r['slope_nS']), intercept_pA=float(r['intercept_pA']),
                     somatic_reversal_mv=float(r['reversal_mv']))
                for r in csv.DictReader(stream) if r['transmitter'] == 'GABA']
    if len(rows) != 5 or not all(np.isfinite(r['somatic_reversal_mv']) for r in rows):
        raise ValueError('The five observed GABA cells must remain visible')
    return dict(article_doi=source['source_article_doi'], dataset_doi=source['source_dataset_doi'],
        article_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC8891015/',
        origin='Published conductance-model E_GABA=-68mV, measured/modelled in voltage-clamp experiments; no fit to MATRIX activity or behavior.',
        equation_scope='Published numerator includes E_GABA*(g_Mi4+g_C3); denominator includes both conductances.',
        local_measurement_path=str(path), local_measurement_sha256=digest, observed_cells=rows,
        limits=['Somatic currents after1mM GABA puff for100ms onto M10 dendrites; not an endogenous single-synapse conductance.',
                'Young adult females, T4c/d-biased label, transferred to male T4a/b/c/d by type.',
                'Cell estimates span about-49 to-87mV; the historical subsets disagree. No heldout validation is claimed.',
                'Possible GABA_B/GIRK contribution; this is not an exclusive Rdl/chloride assignment.',
                'Per-edge conductance and synaptic time constant remain inherited hypotheses.'])


def selection(brain, types):
    t4 = np.flatnonzero(np.isin(types, POST_TYPES))
    sources = np.isin(types, PRE_TYPES)
    parts = []
    for row in t4:
        lo, hi = brain.W.indptr[row:row+2]
        parts.append(lo + np.flatnonzero(sources[brain.W.indices[lo:hi]]))
    positions = np.concatenate(parts).astype(np.int64)
    if not len(positions):
        raise ValueError('No canonical Mi4/C3 -> T4 pairs')
    pres = brain.W.indices[positions]
    posts = np.searchsorted(brain.W.indptr, positions, side='right')-1
    if (not np.all(np.char.lower(np.asarray(brain.nt_labels[pres], dtype='U')) == 'gaba')
            or np.any(brain.W.data[positions] >= 0)):
        raise ValueError('Requires recorded GABA sources and existing negative weights')
    return positions, pres.astype(np.int64), posts.astype(np.int64)


class T4GabaBrain(KcGammaRegionalBrain):
    SCHEMA = 'matrix_t4_gaba_brain_fp64_v1'
    GABA_PARENT_CLASS = KcGammaRegionalBrain

    @classmethod
    def adopt(cls, reference, *, enabled=True):
        if type(enabled) is not bool or reference.SCHEMA != cls.GABA_PARENT_CLASS.SCHEMA:
            raise ValueError('Adopt the exact current regional brain with an explicit connection')
        b = reference.brain
        types = reference.measured_t4_manifest['canonical_node_types']
        positions, pres, posts = selection(b, types)
        if not np.all(reference.visual_mask[pres] & reference.visual_mask[posts]):
            raise ValueError('Equivalent reversal requires graded visual source/target cells')
        if not np.array_equal(reference.weights64[positions], b.W.data[positions].astype(float)):
            raise ValueError('Unarchived effective override on selected pairs')
        saved = reference.state_dict()
        m = dict(policy=POLICY, enabled=enabled, E_GABA_mV=E_GABA_MV,
            source_schema=reference.SCHEMA, adoption_time_ns=reference.time_ns,
            source_state_sha256=_hash_array(reference.state), constraints=constraints(),
            positions=positions, pre_ids=b.node_ids[pres].copy(), post_ids=b.node_ids[posts].copy(),
            target_rows=np.unique(posts), target_ids=b.node_ids[np.unique(posts)].copy(),
            node_ids_sha256=_hash_array(b.node_ids), canonical_types_sha256=_hash_array(types),
            anatomical_indptr_sha256=_hash_array(b.W.indptr), anatomical_indices_sha256=_hash_array(b.W.indices),
            selected_weights_sha256=_hash_array(b.W.data[positions]),
            correction='Add ((E_GABA+80)/80)*g_Mi4_C3/(rate*tau) to T4 normalized voltage target; unchanged rate and conductance.',
            other_inputs='All unselected GABA and other afferents keep their inherited reversal and contribution.',
            biological_validation=False, parameter_fitting=False)
        m['record_sha256'] = _record_hash(m)
        saved.update(schema=cls.SCHEMA, t4_gaba_manifest=m)
        return cls.from_state(b, saved)

    def _build_gaba_cache(self):
        m, b = self.t4_gaba_manifest, self.brain
        types = self.measured_t4_manifest['canonical_node_types']
        if (m.get('policy') != POLICY or type(m.get('enabled')) is not bool
                or m.get('E_GABA_mV') != E_GABA_MV or m.get('source_schema') != self.GABA_PARENT_CLASS.SCHEMA
                or m.get('record_sha256') != _record_hash(m) or m.get('biological_validation') is not False):
            raise ValueError('Changed GABA mechanism, fixed reversal or provenance')
        for name, array in [('node_ids', b.node_ids), ('canonical_types', types),
                            ('anatomical_indptr', b.W.indptr), ('anatomical_indices', b.W.indices)]:
            if m[name+'_sha256'] != _hash_array(array):
                raise ValueError('GABA constraint differs from canonical identity: '+name)
        positions, pres, posts = selection(b, types)
        rows, counts = np.unique(posts, return_counts=True)
        for name, array in [('positions', positions), ('pre_ids', b.node_ids[pres]),
                            ('post_ids', b.node_ids[posts]), ('target_rows', rows), ('target_ids', b.node_ids[rows])]:
            if not np.array_equal(m[name], array):
                raise ValueError('Changed GABA canonical mapping: '+name)
        if (m['selected_weights_sha256'] != _hash_array(b.W.data[positions])
                or type(m['adoption_time_ns']) is not int
                or not 0 <= m['adoption_time_ns'] <= self.time_ns):
            raise ValueError('Changed selected weights or GABA adoption clock')
        if self.time_ns == m['adoption_time_ns'] and m['source_state_sha256'] != _hash_array(self.state):
            raise ValueError('GABA adoption must not reset a neural or retinal state')
        self._gaba_rows = rows.astype(np.int64)
        self._gaba_ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self._gaba_positions, self._gaba_pres = positions, pres

    def _coefficients(self, state, drive, light):
        target, rate = super()._coefficients(state, drive, light)
        if self.t4_gaba_manifest['enabled']:
            _correct_targets(self._gaba_rows, self._gaba_ptr, self._gaba_positions,
                self._gaba_pres, self.weights64, state[self.transmission_start:self.inherited_state_size],
                self.parameters['conductance_per_stored_weight'], (E_GABA_MV+80.)/80.,
                self.tau, target, rate)
        return target, rate

    def state_dict(self):
        saved = super().state_dict()
        saved['t4_gaba_manifest'] = copy.deepcopy(self.t4_gaba_manifest)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != GABA_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete T4 GABA state or wrong backend')
        parent = {k: v for k, v in saved.items() if k != 't4_gaba_manifest'}
        parent['schema'] = cls.GABA_PARENT_CLASS.SCHEMA
        base = cls.GABA_PARENT_CLASS.from_state(brain, parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.t4_gaba_manifest = copy.deepcopy(saved['t4_gaba_manifest'])
        obj._build_gaba_cache()
        return obj


class GpuT4GabaBrain(T4GabaBrain, GpuKcGammaRegionalBrain):
    SCHEMA = 'matrix_t4_gaba_brain_fp64_cuda_v1'
    GABA_PARENT_CLASS = GpuKcGammaRegionalBrain

    def _build_gaba_cache(self):
        super()._build_gaba_cache()
        import cupy as cp
        from gpu_measured_t4_visual_brain import CORRECTION_CUDA_SOURCE
        self._gaba_cuda = {k: cp.asarray(v) for k, v in dict(rows=self._gaba_rows,
            ptr=self._gaba_ptr, positions=self._gaba_positions, pres=self._gaba_pres).items()}
        self._gaba_kernel = cp.RawKernel(CORRECTION_CUDA_SOURCE, 'mi9_reversal',
            options=('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'))

    def coefficients_gpu(self, state, drive, light):
        target, rate = GpuKcGammaRegionalBrain.coefficients_gpu(self, state, drive, light)
        if self.t4_gaba_manifest['enabled']:
            c, g = self.cuda, self._gaba_cuda
            n = len(self._gaba_rows)
            self._gaba_kernel(((n*32+255)//256,), (256,), (np.int32(n), g['rows'], g['ptr'],
                g['positions'], g['pres'], c['weights'], state[self.transmission_start:self.inherited_state_size],
                np.float64(self.parameters['conductance_per_stored_weight']), np.float64((E_GABA_MV+80.)/80.),
                c['tau'], target, rate))
        return target, rate

    @staticmethod
    def backend_identity():
        info = GpuKcGammaRegionalBrain.backend_identity()
        info['t4_gaba_reversal'] = 'Published -68mV on existing Mi4/C3->T4 pairs; unchanged conductance; fixed FP64 warp reduction.'
        return info
