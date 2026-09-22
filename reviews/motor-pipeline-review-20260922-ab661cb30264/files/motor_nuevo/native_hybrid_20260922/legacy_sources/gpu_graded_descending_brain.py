"""Same graded DNp20 candidate using the inherited deterministic FP64 CUDA kernel."""
from graded_descending_brain import GradedDescendingBrain
from gpu_r8_mi4_visual_brain import GpuR8Mi4VisualBrain


class GpuGradedDescendingBrain(GradedDescendingBrain, GpuR8Mi4VisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_graded_descending_fp64_cuda_v1'

    @staticmethod
    def backend_identity():
        info = GpuR8Mi4VisualBrain.backend_identity()
        info['graded_descending'] = 'two canonical DNp20 rows use existing graded conductance/release; optional VS chemical cache cut; no new kernel'
        return info
