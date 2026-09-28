import numpy as np
import pytest
from pn_frontier_trial import bounded_total_control, baseline_comparison


def test_dose_control_handles_saturation_without_clipping():
    m=np.array([[1., .1, 0], [1., .9, 0]])
    t=np.array([[1., .3, 0], [1., .7, 0]])
    caps=np.array([2., 3., 4.])
    x=bounded_total_control(m,t,caps)
    assert np.all((x>=0)&(x<=1))
    np.testing.assert_allclose(x@caps,t@caps,rtol=0,atol=2e-7)
    assert x[0,0]==1
    assert x[0,2]>0  # Explicit redistribution; not falsely direction-pure.


def test_dose_control_identical_input_is_exact_fp32():
    m=np.array([[1.,.125,0]],dtype=np.float32)
    assert np.array_equal(bounded_total_control(m,m,[1,1,1]),m)


def test_dose_control_rejects_invalid_signal_without_clipping():
    with pytest.raises(ValueError,match='domain'):
        bounded_total_control([[1.1]],[[.5]],[1.])


def test_baseline_fidelity_cannot_hide_oscillating_errors_in_a_mean():
    ref={'DN_q_usada':np.zeros((2,4)), 'DN_baseline':np.zeros((2,4)),
         'command_yaw_rate_rad_s':np.zeros(2),'command_forward_mm_s':np.zeros(2)}
    trial={k:v.copy() for k,v in ref.items()}
    trial['command_yaw_rate_rad_s']=np.deg2rad([.01,-.01])
    c={'window_ms':[1,2], 'baseline_max_DNb_error_q':1.6e-6, 'baseline_max_yaw_error_deg_s':.002}
    assert not baseline_comparison(trial,ref,c)['passed']
