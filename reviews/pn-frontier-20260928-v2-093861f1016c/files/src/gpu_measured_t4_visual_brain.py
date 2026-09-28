"""CUDA backend of the same experimental Mi9 -> T4 reversal hypothesis.

One fixed warp reduction per selected T4 row; no atomics. FP64 throughout.
The authoritative state and calibration record remain portable CPU arrays.
"""
import cupy as cp
import numpy as np

from measured_t4_visual_brain import MeasuredT4VisualBrain
from gpu_receptor_visual_brain import GpuReceptorVisualBrain


CORRECTION_CUDA_SOURCE = r'''
extern "C" __global__ void mi9_reversal(
 int nrows, const long long* rows, const long long* ptr,
 const long long* positions, const long long* pres,
 const double* weights, const double* transmission, double scale,
 double alpha, const double* tau, double* target, const double* rate) {
 int i=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 if(i>=nrows) return;
 double g=0.;
 for(long long e=ptr[i]+lane; e<ptr[i+1]; e+=32)
   g-=(weights[positions[e]]*scale)*transmission[pres[e]];
 for(int d=16;d>0;d/=2) g+=__shfl_down_sync(0xffffffff,g,d);
 if(lane==0) {
   long long row=rows[i];
   target[row]+=alpha*g/(rate[row]*tau[row]);
 }
}
'''


class GpuMeasuredT4VisualBrain(MeasuredT4VisualBrain, GpuReceptorVisualBrain):
    SCHEMA = 'matrix_hybrid_visual_brain_measured_t4_fp64_cuda_v1'

    def _build(self):
        super()._build()
        self._mi9_cuda = {key: cp.asarray(value, dtype=cp.int64) for key, value in (
            ('rows', self._mi9_rows), ('indptr', self._mi9_indptr),
            ('positions', self._mi9_positions), ('pres', self._mi9_pres))}
        self._mi9_kernel = cp.RawKernel(CORRECTION_CUDA_SOURCE, 'mi9_reversal',
            options=('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'))

    def coefficients_gpu(self, state, drive, light):
        target, rate = super().coefficients_gpu(state, drive, light)
        if self._mi9_alpha != 0.:
            c, selected = self.cuda, self._mi9_cuda
            nrows = len(self._mi9_rows)
            args = (np.int32(nrows), selected['rows'], selected['indptr'],
                selected['positions'], selected['pres'], c['weights'],
                state[self.transmission_start:], np.float64(self.parameters['conductance_per_stored_weight']),
                np.float64(self._mi9_alpha), c['tau'], target, rate)
            self._mi9_kernel(((nrows*32+255)//256,), (256,), args)
        return target, rate

    @staticmethod
    def backend_identity():
        info = GpuReceptorVisualBrain.backend_identity()
        info['mi9_reversal_reduction'] = 'one warp per selected T4 row; fixed binary tree; no atomics; FP64; no FMA or fast math'
        return info
