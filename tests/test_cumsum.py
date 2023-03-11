import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats as st

from aeppl.joint_logprob import DensityNotFound, joint_logprob
from tests.utils import assert_no_rvs


@pytest.mark.parametrize(
    "size, axis",
    [
        (10, None),
        (10, 0),
        ((2, 10), 0),
        ((2, 10), 1),
        ((3, 2, 10), 0),
        ((3, 2, 10), 1),
        ((3, 2, 10), 2),
    ],
)
def test_normal_cumsum(size, axis):
    srng = at.random.RandomStream(2023532)

    rv = srng.normal(0, 1, size=size).cumsum(axis)
    logp, (vv,) = joint_logprob(rv)
    assert_no_rvs(logp)

    assert np.isclose(
        st.norm(0, 1).logpdf(np.ones(size)).sum(),
        logp.eval({vv: np.ones(size).cumsum(axis)}),
    )


@pytest.mark.parametrize(
    "size, axis",
    [
        (10, None),
        (10, 0),
        ((2, 10), 0),
        ((2, 10), 1),
        ((3, 2, 10), 0),
        ((3, 2, 10), 1),
        ((3, 2, 10), 2),
    ],
)
def test_bernoulli_cumsum(size, axis):
    srng = at.random.RandomStream(2023532)

    rv = srng.bernoulli(0.9, size=size).cumsum(axis)
    logp, (vv,) = joint_logprob(rv)
    assert_no_rvs(logp)

    assert np.isclose(
        st.bernoulli(0.9).logpmf(np.ones(size)).sum(),
        logp.eval({vv: np.ones(size, int).cumsum(axis)}),
    )


def test_destructive_cumsum_fails():
    """Test that a cumsum that mixes dimensions fails"""
    srng = at.random.RandomStream(2023532)

    x_rv = srng.normal(size=(2, 2, 2)).cumsum()
    with pytest.raises(DensityNotFound):
        joint_logprob(x_rv)


def test_deterministic_cumsum():
    """Test that deterministic cumsum is not affected"""
    srng = at.random.RandomStream(2023532)

    x_rv = srng.normal(1, 1, size=5)
    cumsum_x_rv = at.cumsum(x_rv)
    y_rv = srng.normal(cumsum_x_rv, 1)

    logp, (x_vv, y_vv) = joint_logprob(x_rv, y_rv)
    assert_no_rvs(logp)

    logp_fn = aesara.function([x_vv, y_vv], logp)
    assert np.isclose(
        logp_fn(np.ones(5), np.arange(5) + 1),
        st.norm(1, 1).logpdf(1) * 10,
    )
