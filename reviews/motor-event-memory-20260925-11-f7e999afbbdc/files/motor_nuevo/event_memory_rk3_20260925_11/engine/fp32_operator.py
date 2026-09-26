"""FP32 backend for the dominant CSR coefficient of the real CNS model.

Mutable FP64 weights are converted for each evaluation because PN/APL and
other owners can change the effective operator between evaluations. The
resident time integrator and other model-specific kernels remain FP64.
"""
from pathlib import Path

import cupy as cp
import numpy as np


class FastCSR:
    def __init__(self, brain, *, persistent=False, audit=False):
        c = brain.cuda
        self.original = brain.kernel
        self.kernel = cp.RawKernel(
            Path(__file__).with_name('fast_fp32_coefficient.cu').read_text(),
            'coefficient_fast',
            options=('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'),
        )
        self.mirror = None
        self.writer_groups = {}
        if persistent:
            from persistent_weights import PersistentFP32
            from model_weight_writers import declared_positions
            positions, self.writer_groups = declared_positions(brain)
            self.mirror = PersistentFP32(c['weights'], positions, audit=audit)
            self.weights = self.mirror.values
        else:
            self.weights = cp.empty(c['weights'].shape, dtype=cp.float32)
        n = brain.brain.n_neurons
        self.release = cp.empty(n, dtype=cp.float32)
        self.drive = cp.empty(n, dtype=cp.float32)
        self.photo = cp.empty(n, dtype=cp.float32)
        self.caps = c['caps'].astype(cp.float32)
        self.tau = c['tau'].astype(cp.float32)
        self.gain = c['gain'].astype(cp.float32)
        self.theta = c['theta'].astype(cp.float32)
        self.calls = 0

    def __call__(self, grid, block, args):
        (n, ptr, idx, weights, release, caps, visual, tau, gain, theta,
         drive, photo, scale, connected, target, rate) = args
        if (weights.size != self.weights.size or release.size != self.release.size
                or drive.size != self.drive.size or photo.size != self.photo.size):
            raise ValueError('FP32 CSR layout changed')
        # Device-side copies become CUDA graph nodes. Refreshing weights avoids
        # stale PN/APL edges without changing those model owners.
        if self.mirror is None:
            cp.copyto(self.weights, weights, casting='unsafe')
        else:
            self.mirror.refresh(weights)
        cp.copyto(self.release, release, casting='unsafe')
        cp.copyto(self.drive, drive, casting='unsafe')
        cp.copyto(self.photo, photo, casting='unsafe')
        self.kernel(grid, block, (
            n, ptr, idx, self.weights, self.release, self.caps, visual,
            self.tau, self.gain, self.theta, self.drive, self.photo,
            np.float32(scale), connected, target, rate,
        ))
        self.calls += 1
