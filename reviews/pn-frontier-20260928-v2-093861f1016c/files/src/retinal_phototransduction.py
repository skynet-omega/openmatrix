"""Reduced microvillus availability, with explicitly prosthetic optical gain.

This is a deterministic population approximation, not the Juusola stochastic
phototransduction model. N and the recovery order of magnitude have biological
support; a single exponential recovery is our reduction. The old 5 ms filter
is retained only as an uncalibrated bump filter, never as a voltage clamp.
"""
import numpy as np

PARAMETERS = dict(microvilli_per_cell=30000, recovery_tau_s=.100,
    absorbed_photons_per_s_at_unit_light=1_000_000., bump_filter_tau_s=.005,
    maximum_relative_photoconductance=1., photo_reversal_mv=20.)
STATE_ORDER = ['microvilli_available_fraction',
               'filtered_successful_absorption_fraction']


def initial_state(light):
    """New-history assumption: stationary availability at the current image."""
    light = np.asarray(light, dtype=np.float64)
    if light.ndim != 1 or not np.isfinite(light).all() or np.any((light < 0.) | (light > 1.)):
        raise ValueError('Initialization requires local normalized optical input')
    p = PARAMETERS
    rate = p['absorbed_photons_per_s_at_unit_light']*light/p['microvilli_per_cell']
    available = 1./(1.+p['recovery_tau_s']*rate)
    return np.r_[available, light*available]


def coefficients(available, light):
    """Return A target/rate and bump target; NumPy and CuPy compatible."""
    p = PARAMETERS
    recovery = 1./p['recovery_tau_s']
    rate = recovery+p['absorbed_photons_per_s_at_unit_light']*light/p['microvilli_per_cell']
    return recovery/rate, rate, light*available


def provenance():
    return dict(
        structural_source='Juusola2017 doi:10.7554/eLife.26117; local Gillespie_sim_b2p3_Skew_speed205.m: 30000 microvilli.',
        recovery_source='Song2012 doi:10.1016/j.cub.2012.05.047: recovery order 100-200ms; Juusola2017 gives variable 50-300ms. A single exponential100ms is a reduction, not an individual measurement.',
        reversal_source='Juusola2017 local Vol_FeedbackCluster.m uses TRPrev=20mV for voltage-dependent driving force.',
        optical_gain='Unit image luminance is assigned1e6 absorbed photons/s per receptor by the virtual apparatus; no measured radiometry or spectrum.',
        conductance_gain='g_photo/g_leak = inherited maximum1 * filtered(L*A). Absolute channel count/current and C/g_leak are uncalibrated.',
        kinetics='The inherited5ms fast filter remains a device approximation; no measured bump waveform or latency is claimed.',
        equation='dA/dt=(1-A)/tau_rec-(phi/N)*A; dz/dt=(L*A-z)/tau_filter; phi=phi_max*L; g_photo=z*g_max.',
        omissions=['Stochastic quantum bumps and refractoriness distribution',
                   'Rhodopsin/G-protein/PLC cascade and calcium-dependent bump adaptation',
                   'Photomechanical movements, channel pumps and measured intrinsic membrane conductances'],
        claim='Partial biological constraint within a functional retinal prosthesis; not reconstructed biological retina.')
