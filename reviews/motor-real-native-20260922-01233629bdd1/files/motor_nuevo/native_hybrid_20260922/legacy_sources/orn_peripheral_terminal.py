"""Explicit bounded engineering terminal port, not identified Or42b biophysics."""
import numpy as np

POLICY = 'Or42b_mean_input_bounded_terminal_modulation_v1'
PARAMETERS = dict(basal_hz=9., increment_hz=83.667, world_drive_anchor=80.,
                  modulation_bound=.3, recurrent_threshold_hz=40.)


def peripheral_hz(drive):
    p = PARAMETERS
    return p['basal_hz'] + p['increment_hz'] * np.clip(
        np.asarray(drive, dtype=float) / p['world_drive_anchor'], 0., 1.)


def terminal_target(drive, caps, gain_over_cap, other_input, recurrent_input,
                    *, recurrent_connected):
    """Independent CPU oracle; input sums retain inherited nonphysical W units."""
    if type(recurrent_connected) is not bool:
        raise ValueError('Explicit recurrent connection policy required')
    r = peripheral_hz(drive)
    active = recurrent_connected & (r > PARAMETERS['recurrent_threshold_hz'])
    x = np.asarray(gain_over_cap) * (np.asarray(other_input) + active * np.asarray(recurrent_input))
    return np.clip(r / np.asarray(caps) * (1. + PARAMETERS['modulation_bound'] * np.tanh(x)), 0., 1.)


CUDA_CODE = r'''
extern "C" __global__ void orn_terminal(
    const long long* rows, const long long* indptr, const long long* pres,
    const long long* positions, const long long* pn_slot, const bool* recurrent,
    const double* weights, const double* caps, const double* gain,
    const double* transmission, const double* pn_release, const double* drive,
    const bool connected, double* target) {
    int j=blockDim.x*blockIdx.x+threadIdx.x; if(j>=74)return;
    long long row=rows[j];
    double u=fmin(1.,fmax(0.,drive[row]/80.));
    double r=9.+83.667*u, input=0.;
    bool active=connected && r>40.;
    for(long long e=indptr[j];e<indptr[j+1];++e) {
        if(recurrent[e] && !active)continue;
        double release=pn_slot[e]>=0 ? pn_release[pn_slot[e]] : transmission[pres[e]];
        input+=weights[positions[e]]*caps[pres[e]]*release;
    }
    target[row]=fmin(1.,fmax(0.,r/caps[row]*(1.+.3*tanh(gain[row]*input))));
}
'''
