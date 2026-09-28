"""Finite chemical transmission on the existing CNS, without changing weights.

tau_syn * ds/dt = release - s. Each canonical source has one normalized s,
shared by its chemical outputs. For a common linear time constant this is
equivalent to one filter per connection with identical initialization. It is
a lumped kinetic hypothesis, not receptor-specific physiology or an axonal
delay. A 5 ms prior is borrowed from Shiu 2024's model; the transfer to these
graded visual cells and this male CNS has not been calibrated.

The old neural/retinal state is an unchanged prefix. Migration initializes
s=release at the adoption instant: no instantaneous output/input jump, but
unknown pre-adoption synaptic history is NOT reconstructed. Direct external
sensory drives and the separate neuromuscular model retain their old meaning.
"""
import copy
import hashlib
import numpy as np

from hybrid_visual_brain import HybridVisualBrain
from visual_sparse_kernel import coefficients as sparse_coefficients


DEFAULT_SYNAPTIC_TAU_S = .005
SOURCE_URL = 'https://doi.org/10.1038/s41586-024-07763-9'
STATE_KEYS = {'schema', 'visual_ids', 'photo_ids', 'parameters', 'state',
              'next_step_ns', 'statistics', 'time_ns', 'visual_output_connected',
              'synaptic_migration'}


def _hash_array(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode())
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


class SynapticVisualBrain(HybridVisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_synaptic_fp64_v1'

    def __init__(self, *args, **kwargs):
        raise TypeError('Adopt an existing HybridVisualBrain explicitly with .adopt().')

    @classmethod
    def adopt(cls, reference, synaptic_tau_s=DEFAULT_SYNAPTIC_TAU_S):
        if isinstance(reference, SynapticVisualBrain):
            raise ValueError('Synaptic kinetics already present; restore the saved state instead.')
        if not isinstance(reference, HybridVisualBrain):
            raise TypeError('A complete hybrid neural state is required.')
        tau = float(synaptic_tau_s)
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError('Synaptic time constant must be finite and positive.')
        saved = reference.state_dict()
        if saved['schema'] not in {'matrix_hybrid_visual_brain_v1',
                                   'matrix_hybrid_visual_brain_fp64_cuda_v1'}:
            raise ValueError('Unsupported source neural family.')
        prefix = saved['state']
        transmission = reference.release()
        saved['synaptic_migration'] = dict(
            source_schema=saved['schema'], time_ns=int(reference.time_ns),
            old_state_sha256=_hash_array(prefix), node_ids_sha256=_hash_array(reference.brain.node_ids),
            initial_transmission_sha256=_hash_array(transmission),
            initialization='s equals instantaneous normalized release; older synaptic history unavailable',
            parameter_status='candidate common time constant; not measured per cell/receptor',
            parameter_source=SOURCE_URL, adoption_tau_s=tau,
            preserved='old state prefix, clock, IDs, topology, weights, gains, thresholds and RNG')
        saved['parameters']['synaptic_tau_s'] = tau
        saved['state'] = np.concatenate((prefix, transmission))
        saved['schema'] = cls.SCHEMA
        return cls.from_state(reference.brain, saved)

    @property
    def transmission_start(self):
        return self.brain.n_neurons + 2*len(self.pi)

    def transmission_release(self, state=None):
        state = self.state if state is None else state
        return state[self.transmission_start:].copy()

    def _coefficients(self, state, drive, light):
        self.statistics['evaluations'] += 1
        n, m = self.brain.n_neurons, len(self.pi)
        p = self.parameters
        fast, adaptation = state[n:n+m], state[n+m:n+2*m]
        photo = np.zeros(n, dtype=np.float64)
        photo[self.pi] = p['photoconductance_max']*fast/(
            p['photo_half']+p['adaptation_strength']*adaptation+fast)
        target, rate = np.empty_like(state), np.empty_like(state)
        target[:n], rate[:n] = sparse_coefficients(
            self.brain.W.indptr, self.brain.W.indices, self.weights64,
            state[self.transmission_start:], self.caps, self.visual_mask,
            self.tau, self.rate_gain, self.rate_theta, drive, photo,
            p['conductance_per_stored_weight'], self.visual_output_connected)
        target[n:n+m], rate[n:n+m] = light, 1./p['photo_fast_tau_s']
        target[n+m:n+2*m], rate[n+m:n+2*m] = fast, 1./p['photo_adaptation_tau_s']
        target[self.transmission_start:] = self.release(state)
        rate[self.transmission_start:] = 1./p['synaptic_tau_s']
        return target, rate

    def state_dict(self):
        saved = super().state_dict()
        saved['synaptic_migration'] = copy.deepcopy(self.synaptic_migration)
        return saved

    @classmethod
    def from_state(cls, brain, saved):
        if set(saved) != STATE_KEYS or saved.get('schema') != cls.SCHEMA:
            raise ValueError('Incomplete synaptic hybrid state or wrong neural family.')
        tau = saved['parameters'].get('synaptic_tau_s')
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not np.isfinite(tau) or tau <= 0:
            raise ValueError('Invalid saved synaptic time constant.')
        migration = saved['synaptic_migration']
        if not isinstance(migration, dict) or migration.get('node_ids_sha256') != _hash_array(brain.node_ids):
            raise ValueError('Synaptic migration does not match canonical neuron identities.')
        if migration.get('adoption_tau_s') != tau:
            raise ValueError('Kinetics parameter differs from its explicit adoption.')
        obj = cls.__new__(cls)
        obj.brain = brain
        for name in STATE_KEYS-{'schema'}:
            setattr(obj, name, copy.deepcopy(saved[name]))
        obj._build()
        if (obj.state.dtype != np.float64 or obj.state.shape != (2*brain.n_neurons+2*len(obj.pi),)
                or not np.isfinite(obj.state).all() or np.any((obj.state < 0) | (obj.state > 1))):
            raise ValueError('Invalid authoritative neural/transmission state.')
        if (obj.time_ns != brain.time_ns
                or not np.array_equal((obj.release()*obj.caps).astype(np.float32), brain.rates)):
            raise ValueError('Published view or neural clock disagrees with saved state.')
        if type(migration.get('time_ns')) is not int or not 0 <= migration['time_ns'] <= obj.time_ns:
            raise ValueError('Invalid synaptic adoption time.')
        return obj

    def observe_signals(self, node_ids):
        """Read stage signals without advancing; preserves individual cells.

        Coordinates mean graded normalized voltage for visual cells and
        normalized rate elsewhere. Only the visual subset has a voltage.
        The published rate-equivalent view of graded cells is not firing Hz.
        """
        ids = np.asarray(node_ids)
        if ids.dtype.kind not in 'iu' or ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError('Expected unique integer canonical IDs.')
        rows = np.searchsorted(self.brain.node_ids, ids)
        if np.any(rows >= self.brain.n_neurons) or not np.array_equal(self.brain.node_ids[rows], ids):
            raise ValueError('Unknown neural stage ID.')
        visual = self.visual_mask[rows]
        return dict(node_ids=ids.copy(), time_ns=int(self.time_ns),
                    normalized_state=self.state[rows].copy(), visual_mask=visual.copy(),
                    normalized_release=self.release()[rows],
                    normalized_transmission=self.transmission_release()[rows],
                    visual_voltage_ids=ids[visual].copy(),
                    visual_voltage_mv=-80.+80.*self.state[rows[visual]])
