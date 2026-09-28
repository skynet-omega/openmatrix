import numpy as np
import pytest
from signal_contrast_preflight import decompose, domain_summary, validate_summary


def test_subtraction_can_leave_nonnegative_domain():
    parts = decompose([[0, .25]], [[.4, .8]], [[.8, .4]])
    assert parts['null_common_L'][0, 0] == -.2
    assert not domain_summary(parts['null_common_L'], 0, 1)['domain_valid']
    assert domain_summary(parts['null_lateral'], 0, 1)['domain_valid']


def test_condition_decomposition_is_reconstructible_but_not_dose_matched():
    n = np.array([[.1, .2], [.2, .3]])
    l = np.array([[.3, .6], [.4, .8]])
    r = np.array([[.2, .5], [.3, .7]])
    p = decompose(n, l, r)
    np.testing.assert_allclose(n+p['common']+p['lateral'], l, rtol=0, atol=2e-16)
    np.testing.assert_allclose(n+p['common']-p['lateral'], r, rtol=0, atol=2e-16)
    assert np.any(p['lateral'].sum(1) != 0)


def test_same_summary_does_not_identify_history_or_filtered_effect():
    a = np.array([.2, .1, .7, .3, .4])
    b = np.array([.2, .7, .1, .3, .4])
    summary = lambda x: (x[0], x[-1], x.min(), x.max(), len(x))
    assert summary(a) == summary(b)
    # A fixed causal filter distinguishes histories that the summary cannot.
    assert np.dot(a, .5 ** np.arange(4, -1, -1)) != np.dot(b, .5 ** np.arange(4, -1, -1))


def test_summary_rejects_corrupted_extrema_and_counts():
    s = {k: np.array([[.2, .3]]) for k in ('first', 'last', 'lo', 'hi')}
    s['counts'] = np.array([[5, 5]], dtype=np.uint64)
    validate_summary(s, (1, 2), 0, 1)
    s['first'][0, 0] = .1
    with pytest.raises(ValueError, match='extrema'):
        validate_summary(s, (1, 2), 0, 1)
    s['first'][0, 0] = .2
    s['counts'][0, 1] = 4
    with pytest.raises(ValueError, match='coverage'):
        validate_summary(s, (1, 2), 0, 1)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_is_rejected(bad):
    with pytest.raises(ValueError, match='finite'):
        decompose([[0]], [[bad]], [[0]])
