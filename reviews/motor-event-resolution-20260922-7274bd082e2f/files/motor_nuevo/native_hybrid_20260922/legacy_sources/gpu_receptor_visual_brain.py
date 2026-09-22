"""CUDA execution of the same candidate R8p/R8y -> Mi1 receptor policy."""
from receptor_visual_brain import ReceptorVisualBrain
from gpu_synaptic_visual_brain import GpuSynapticVisualBrain


class GpuReceptorVisualBrain(ReceptorVisualBrain, GpuSynapticVisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_receptor_fp64_cuda_v1'
