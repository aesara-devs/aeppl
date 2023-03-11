import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy as sp
import scipy.stats as st

from aeppl.joint_logprob import conditional_logprob, joint_logprob
from aeppl.transforms import LogTransform, TransformValuesRewrite
from tests.utils import assert_no_rvs


@aesara.config.change_flags(compute_test_value="raise")
def test_continuous_rv_clip():
    srng = at.random.RandomStream(0)

    x_rv = srng.normal(0.5, 1)
    cens_x_rv = at.clip(x_rv, -2, 2)

    logp, vv = joint_logprob(cens_x_rv)
    assert_no_rvs(logp)

    logp_fn = aesara.function(vv, logp)
    ref_scipy = st.norm(0.5, 1)

    assert logp_fn(-3) == -np.inf
    assert logp_fn(3) == -np.inf

    assert np.isclose(logp_fn(-2), ref_scipy.logcdf(-2))
    assert np.isclose(logp_fn(2), ref_scipy.logsf(2))
    assert np.isclose(logp_fn(0), ref_scipy.logpdf(0))


def test_discrete_rv_clip():
    srng = at.random.RandomStream(0)

    x_rv = srng.poisson(2)
    cens_x_rv = at.clip(x_rv, 1, 4)

    logp, vv = joint_logprob(cens_x_rv)
    assert_no_rvs(logp)

    logp_fn = aesara.function(vv, logp)
    ref_scipy = st.poisson(2)

    assert logp_fn(0) == -np.inf
    assert logp_fn(5) == -np.inf

    assert np.isclose(logp_fn(1), ref_scipy.logcdf(1))
    assert np.isclose(logp_fn(4), np.logaddexp(ref_scipy.logsf(4), ref_scipy.logpmf(4)))
    assert np.isclose(logp_fn(2), ref_scipy.logpmf(2))


def test_one_sided_clip():
    srng = at.random.RandomStream(0)

    x_rv = srng.normal(0, 1)
    lb_cens_x_rv = at.clip(x_rv, -1, x_rv)
    ub_cens_x_rv = at.clip(x_rv, x_rv, 1)

    lb_logp, (lb_cens_x_vv,) = joint_logprob(lb_cens_x_rv)
    ub_logp, (ub_cens_x_vv,) = joint_logprob(ub_cens_x_rv)
    assert_no_rvs(lb_logp)
    assert_no_rvs(ub_logp)

    logp_fn = aesara.function([lb_cens_x_vv, ub_cens_x_vv], [lb_logp, ub_logp])
    ref_scipy = st.norm(0, 1)

    assert np.all(np.array(logp_fn(-2, 2)) == -np.inf)
    assert np.all(np.array(logp_fn(2, -2)) != -np.inf)
    np.testing.assert_almost_equal(logp_fn(-1, 1), ref_scipy.logcdf(-1))
    np.testing.assert_almost_equal(logp_fn(1, -1), ref_scipy.logpdf(-1))


def test_useless_clip():
    srng = at.random.RandomStream(0)

    x_rv = srng.normal(0.5, 1, size=3)
    cens_x_rv = at.clip(x_rv, x_rv, x_rv)

    logps, (cens_x_vv,) = conditional_logprob(cens_x_rv)
    logp = logps[cens_x_rv]
    assert_no_rvs(logp)

    logp_fn = aesara.function([cens_x_vv], logp)
    ref_scipy = st.norm(0.5, 1)

    np.testing.assert_allclose(logp_fn([-2, 0, 2]), ref_scipy.logpdf([-2, 0, 2]))


def test_random_clip():
    srng = at.random.RandomStream(0)

    lb_rv = srng.normal(0, 1, size=2)
    x_rv = srng.normal(0, 2)
    cens_x_rv = at.clip(x_rv, lb_rv, [1, 1])

    logps, (cens_x_vv, lb_vv) = conditional_logprob(cens_x_rv, lb_rv)
    logp = at.add(*logps.values())
    assert_no_rvs(logp)

    logp_fn = aesara.function([lb_vv, cens_x_vv], logp)
    res = logp_fn([0, -1], [-1, -1])
    assert res[0] == -np.inf
    assert res[1] != -np.inf


def test_broadcasted_clip_constant():
    srng = at.random.RandomStream(0)

    lb_rv = srng.uniform(0, 1)
    x_rv = srng.normal(0, 2)
    cens_x_rv = at.clip(x_rv, lb_rv, [1, 1])

    logp, _ = joint_logprob(cens_x_rv, lb_rv)
    assert_no_rvs(logp)


def test_broadcasted_clip_random():
    srng = at.random.RandomStream(0)

    lb_rv = srng.normal(0, 1)
    x_rv = srng.normal(0, 2, size=2)
    cens_x_rv = at.clip(x_rv, lb_rv, 1)

    logp, _ = joint_logprob(cens_x_rv, lb_rv)
    assert_no_rvs(logp)


def test_fail_multiple_clip_single_base():
    """Test failure when multiple values are assigned to the same clipped term."""
    srng = at.random.RandomStream(0)

    base_rv = srng.normal(0, 1)
    cens_rv1 = at.clip(base_rv, -1, 1)
    cens_rv1.name = "cens1"
    cens_rv2 = at.clip(base_rv, -1, 1)
    cens_rv2.name = "cens2"

    with pytest.raises(ValueError):
        conditional_logprob(cens_rv1, cens_rv2)


def test_deterministic_clipping():
    srng = at.random.RandomStream(0)

    x_rv = srng.normal(0, 1)
    clip = at.clip(x_rv, 0, 0)
    y_rv = srng.normal(clip, 1)

    logp, (x_vv, y_vv) = joint_logprob(x_rv, y_rv)
    assert_no_rvs(logp)

    logp_fn = aesara.function([x_vv, y_vv], logp)
    assert np.isclose(
        logp_fn(-1, 1),
        st.norm(0, 1).logpdf(-1) + st.norm(0, 1).logpdf(1),
    )


def test_clip_transform():
    srng = at.random.RandomStream(0)

    x_rv = srng.normal(0.5, 1)
    cens_x_rv = at.clip(x_rv, 0, x_rv)

    transform = TransformValuesRewrite({cens_x_rv: LogTransform()})
    logp, (cens_x_vv,) = joint_logprob(cens_x_rv, extra_rewrites=transform)

    cens_x_vv_testval = -1
    obs_logp = logp.eval({cens_x_vv: cens_x_vv_testval})
    exp_logp = (
        sp.stats.norm(0.5, 1).logpdf(np.exp(cens_x_vv_testval)) + cens_x_vv_testval
    )

    assert np.isclose(obs_logp, exp_logp)


@pytest.mark.parametrize(
    "rounding_op, expected_logp_fn",
    (
        (
            at.round,
            lambda x_sp, test_value: np.log(
                x_sp.cdf(test_value + 0.5) - x_sp.cdf(test_value - 0.5)
            ),
        ),
        (
            at.floor,
            lambda x_sp, test_value: np.log(
                x_sp.cdf(test_value + 1.0) - x_sp.cdf(test_value)
            ),
        ),
        (
            at.ceil,
            lambda x_sp, test_value: np.log(
                x_sp.cdf(test_value) - x_sp.cdf(test_value - 1.0)
            ),
        ),
    ),
)
def test_rounding(rounding_op, expected_logp_fn):
    srng = at.random.RandomStream(0)

    loc = 1
    scale = 2
    test_value = np.arange(-3, 4)

    x = srng.normal(loc, scale, size=test_value.shape, name="x")
    xr = rounding_op(x)
    xr.name = "xr"

    logp, (xr_vv,) = conditional_logprob(xr)
    logp = logp[xr]
    assert logp is not None

    x_sp = st.norm(loc, scale)
    expected_logp = expected_logp_fn(x_sp, test_value)

    assert np.allclose(
        logp.eval({xr_vv: test_value}),
        expected_logp,
    )
