"""Dimensionless local-release port for inherited general rate consumers.

Kernel-area normalization transfers one expected local site event to one
inherited rate-proxy event. This declared hypothesis is not a Ca-to-Hz
measurement. Anatomical weights and receptor efficacies are not fitted here.
"""
import hashlib
import numpy as np
from pn_prepared_kc_boundary import zero_release_tail

SCHEMA = 'PN_general_output_port_v1'


class GeneralOutputPort:
    def __init__(self, *, site_ids, relation_sites, target_ids, fractions,
                 site_fast, site_slow, rise_s, decay_s, source_cap_Hz,
                 legacy_transmission, legacy_tau_s, time_ns, provenance):
        identifiers = [np.asarray(x) for x in (site_ids, relation_sites, target_ids)]
        if any(x.ndim != 1 or x.dtype.kind not in 'iu' for x in identifiers):
            raise ValueError('Anatomical identifiers must be integer vectors')
        self.site_ids, relations, targets = [x.astype(np.int64, copy=True) for x in identifiers]
        fractions = np.asarray(fractions, dtype=float)
        if (self.site_ids.ndim != 1 or not len(self.site_ids)
                or np.any(np.diff(self.site_ids) <= 0)
                or relations.ndim != 1 or not len(relations)
                or targets.shape != relations.shape or fractions.shape != relations.shape
                or not np.isfinite(fractions).all() or np.any(fractions <= 0)
                or not np.isin(relations, self.site_ids).all()):
            raise ValueError('Complete anatomical site-to-target relations required')
        self.targets, self.target_slot = np.unique(targets, return_inverse=True)
        self.site_slot = np.searchsorted(self.site_ids, relations)
        np.testing.assert_allclose(np.bincount(self.target_slot, weights=fractions),
                                   1., rtol=0, atol=1e-12)
        a = [np.asarray(x, dtype=float) for x in (site_fast, site_slow, rise_s, decay_s)]
        if (any(x.shape != self.site_ids.shape or not np.isfinite(x).all() for x in a)
                or np.any(a[0] < 0) or np.any(a[1] < a[0])
                or np.any(a[2] <= 0) or np.any(a[3] <= a[2])
                or not np.isfinite(source_cap_Hz) or source_cap_Hz <= 0
                or not np.isfinite(legacy_tau_s) or legacy_tau_s <= 0
                or not np.isscalar(legacy_transmission) or not 0 <= legacy_transmission <= 1
                or type(time_ns) is not int or time_ns < 0
                or not isinstance(provenance, str) or not provenance.strip()):
            raise ValueError('Explicit pulse, cap, legacy history and provenance required')
        self.fast, self.slow, self.rise, self.decay = [x.copy() for x in a]
        peak = np.log(self.decay / self.rise) / (1 / self.rise - 1 / self.decay)
        self.norm = np.exp(-peak / self.decay) - np.exp(-peak / self.rise)
        self.area_s = (self.decay - self.rise) / self.norm
        self.gain = fractions / (source_cap_Hz * self.area_s[self.site_slot])
        self.legacy = float(legacy_transmission)
        self.tau = float(legacy_tau_s)
        self.time_ns = self.origin_ns = time_ns
        self.provenance = provenance
        h = hashlib.sha256((SCHEMA + provenance + repr((time_ns, source_cap_Hz,
                            legacy_transmission, legacy_tau_s))).encode())
        for x in (self.site_ids, relations, targets, fractions, *a):
            h.update(x.tobytes())
        self.identity = h.hexdigest()
        for x in (self.site_ids, self.targets, self.site_slot, self.target_slot,
                  self.rise, self.decay, self.norm, self.area_s, self.gain):
            x.flags.writeable = False

    def output(self, activation):
        activation = np.asarray(activation, dtype=float)
        if activation.shape != self.fast.shape or not np.isfinite(activation).all():
            raise ValueError('Wrong local release field')
        new = activation - (self.slow - self.fast) / self.norm
        if np.any(new < 0):
            raise ValueError('Release below its preserved prehistory')
        result = self.legacy + np.bincount(self.target_slot,
                    weights=new[self.site_slot] * self.gain, minlength=len(self.targets))
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite general output')
        return result

    def advance(self, dt_ns):
        if type(dt_ns) is not int or not 0 < dt_ns <= 125000:
            raise ValueError('Positive bounded electrical coupling required')
        fast, slow = zero_release_tail(self.fast, self.slow, self.rise, self.decay, dt_ns)
        self.fast, self.slow = fast, slow
        self.legacy *= np.exp(-dt_ns * 1e-9 / self.tau)
        self.time_ns += dt_ns

    def state_dict(self):
        return dict(schema=SCHEMA, identity=self.identity, time_ns=self.time_ns,
                    fast=self.fast.copy(), slow=self.slow.copy(), legacy=self.legacy)

    def validate_state(self, state):
        if (set(state) != {'schema', 'identity', 'time_ns', 'fast', 'slow', 'legacy'}
                or state['schema'] != SCHEMA or state['identity'] != self.identity
                or type(state['time_ns']) is not int or state['time_ns'] < self.origin_ns
                or not np.isscalar(state['legacy']) or not 0 <= state['legacy'] <= 1):
            raise ValueError('Wrong general output identity or clock')
        a, b = np.asarray(state['fast']), np.asarray(state['slow'])
        if (a.shape != self.fast.shape or b.shape != a.shape
                or a.dtype.kind not in 'fiu' or b.dtype.kind not in 'fiu'
                or not np.isfinite([a, b]).all() or np.any(a < 0) or np.any(b < a)):
            raise ValueError('Invalid general output prehistory')
        return a.astype(float, copy=True), b.astype(float, copy=True)

    def load_state_dict(self, state):
        fast, slow = self.validate_state(state)
        self.fast, self.slow = fast, slow
        self.legacy, self.time_ns = float(state['legacy']), state['time_ns']
