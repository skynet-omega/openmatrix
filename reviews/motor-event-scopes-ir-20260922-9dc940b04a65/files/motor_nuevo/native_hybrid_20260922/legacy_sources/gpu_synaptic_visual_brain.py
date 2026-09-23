"""FP64 CUDA implementation of the same finite chemical transmission model."""
import cupy as cp
import numpy as np
from synaptic_visual_brain import SynapticVisualBrain
from gpu_visual_brain import GpuVisualBrain


class GpuSynapticVisualBrain(SynapticVisualBrain, GpuVisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_synaptic_fp64_cuda_v1'

    def coefficients_gpu(self, state, drive, light):
        self.statistics['evaluations'] += 1
        n, m = self.brain.n_neurons, len(self.pi)
        p, c = self.parameters, self.cuda
        released = state[:n].copy()
        released[c['vi']] = cp.clip((80.*released[c['vi']]-15.)/40., 0., 1.)
        fast, adaptation = state[n:n+m], state[n+m:n+2*m]
        photo = cp.zeros(n, dtype=cp.float64)
        photo[c['pi']] = p['photoconductance_max']*fast/(
            p['photo_half']+p['adaptation_strength']*adaptation+fast)
        target, rate = cp.empty_like(state), cp.empty_like(state)
        args = (np.int32(n), c['indptr'], c['indices'], c['weights'],
                state[self.transmission_start:], c['caps'], c['visual'], c['tau'],
                c['gain'], c['theta'], drive, photo,
                np.float64(p['conductance_per_stored_weight']),
                np.bool_(self.visual_output_connected), target, rate)
        self.kernel(((n*32+255)//256,), (256,), args)
        target[n:n+m], rate[n:n+m] = light, 1./p['photo_fast_tau_s']
        target[n+m:n+2*m], rate[n+m:n+2*m] = fast, 1./p['photo_adaptation_tau_s']
        target[self.transmission_start:] = released
        rate[self.transmission_start:] = 1./p['synaptic_tau_s']
        return target, rate
