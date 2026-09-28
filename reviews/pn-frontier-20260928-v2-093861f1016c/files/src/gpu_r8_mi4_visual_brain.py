"""CUDA backend of the same explicit R8p/R8y -> Mi4 receptor hypothesis.

Only the selected effective weight cache changes; the inherited deterministic
FP64 reductions, kinetic state and measured Mi9-to-T4 reversal remain intact.
"""
from r8_mi4_visual_brain import R8Mi4VisualBrain
from gpu_measured_t4_visual_brain import GpuMeasuredT4VisualBrain


class GpuR8Mi4VisualBrain(R8Mi4VisualBrain, GpuMeasuredT4VisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_r8_mi4_fp64_cuda_v1'

    @staticmethod
    def backend_identity():
        info = GpuMeasuredT4VisualBrain.backend_identity()
        info['r8_mi4_interpretation'] = 'selected effective sign cache only; inherited deterministic FP64 reductions'
        return info
