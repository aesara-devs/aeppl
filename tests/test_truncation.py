import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy as sp
import scipy.stats
import scipy.stats as st
from aesara.tensor.random.basic import GeometricRV, NormalRV, UniformRV

from aeppl import factorized_joint_logprob, joint_logprob, logprob
from aeppl.logprob import ParameterValueError, _icdf
from aeppl.transforms import LogTransform, TransformValuesOpt
from aeppl.truncation import TruncatedRV, TruncationError, _truncated, truncate
from tests.utils import assert_no_rvs


@aesara.config.change_flags(compute_test_value="raise")
def test_continuous_rv_censoring():
    x_rv = at.random.normal(0.5, 1)
    cens_x_rv = at.clip(x_rv, -2, 2)

    cens_x_vv = cens_x_rv.clone()
    cens_x_vv.tag.test_value = 0

    logp = joint_logprob({cens_x_rv: cens_x_vv})
    assert_no_rvs(logp)

    logp_fn = aesara.function([cens_x_vv], logp)
    ref_scipy = st.norm(0.5, 1)

    assert logp_fn(-3) == -np.inf
    assert logp_fn(3) == -np.inf

    assert np.isclose(logp_fn(-2), ref_scipy.logcdf(-2))
    assert np.isclose(logp_fn(2), ref_scipy.logsf(2))
    assert np.isclose(logp_fn(0), ref_scipy.logpdf(0))


def test_discrete_rv_censoring():
    x_rv = at.random.poisson(2)
    cens_x_rv = at.clip(x_rv, 1, 4)

    cens_x_vv = cens_x_rv.clone()

    logp = joint_logprob({cens_x_rv: cens_x_vv})
    assert_no_rvs(logp)

    logp_fn = aesara.function([cens_x_vv], logp)
    ref_scipy = st.poisson(2)

    assert logp_fn(0) == -np.inf
    assert logp_fn(5) == -np.inf

    assert np.isclose(logp_fn(1), ref_scipy.logcdf(1))
    assert np.isclose(logp_fn(4), np.logaddexp(ref_scipy.logsf(4), ref_scipy.logpmf(4)))
    assert np.isclose(logp_fn(2), ref_scipy.logpmf(2))


def test_one_sided_censoring():
    x_rv = at.random.normal(0, 1)
    lb_cens_x_rv = at.clip(x_rv, -1, x_rv)
    ub_cens_x_rv = at.clip(x_rv, x_rv, 1)

    lb_cens_x_vv = lb_cens_x_rv.clone()
    ub_cens_x_vv = ub_cens_x_rv.clone()

    lb_logp = joint_logprob({lb_cens_x_rv: lb_cens_x_vv})
    ub_logp = joint_logprob({ub_cens_x_rv: ub_cens_x_vv})
    assert_no_rvs(lb_logp)
    assert_no_rvs(ub_logp)

    logp_fn = aesara.function([lb_cens_x_vv, ub_cens_x_vv], [lb_logp, ub_logp])
    ref_scipy = st.norm(0, 1)

    assert np.all(np.array(logp_fn(-2, 2)) == -np.inf)
    assert np.all(np.array(logp_fn(2, -2)) != -np.inf)
    np.testing.assert_almost_equal(logp_fn(-1, 1), ref_scipy.logcdf(-1))
    np.testing.assert_almost_equal(logp_fn(1, -1), ref_scipy.logpdf(-1))


def test_useless_censoring():
    x_rv = at.random.normal(0.5, 1, size=3)
    cens_x_rv = at.clip(x_rv, x_rv, x_rv)

    cens_x_vv = cens_x_rv.clone()

    logp = joint_logprob({cens_x_rv: cens_x_vv}, sum=False)
    assert_no_rvs(logp)

    logp_fn = aesara.function([cens_x_vv], logp)
    ref_scipy = st.norm(0.5, 1)

    np.testing.assert_allclose(logp_fn([-2, 0, 2]), ref_scipy.logpdf([-2, 0, 2]))


def test_random_censoring():
    lb_rv = at.random.normal(0, 1, size=2)
    x_rv = at.random.normal(0, 2)
    cens_x_rv = at.clip(x_rv, lb_rv, [1, 1])

    lb_vv = lb_rv.clone()
    cens_x_vv = cens_x_rv.clone()
    logp = joint_logprob({cens_x_rv: cens_x_vv, lb_rv: lb_vv}, sum=False)
    assert_no_rvs(logp)

    logp_fn = aesara.function([lb_vv, cens_x_vv], logp)
    res = logp_fn([0, -1], [-1, -1])
    assert res[0] == -np.inf
    assert res[1] != -np.inf


def test_broadcasted_censoring_constant():
    lb_rv = at.random.uniform(0, 1)
    x_rv = at.random.normal(0, 2)
    cens_x_rv = at.clip(x_rv, lb_rv, [1, 1])

    lb_vv = lb_rv.clone()
    cens_x_vv = cens_x_rv.clone()

    logp = joint_logprob({cens_x_rv: cens_x_vv, lb_rv: lb_vv})
    assert_no_rvs(logp)


def test_broadcasted_censoring_random():
    lb_rv = at.random.normal(0, 1)
    x_rv = at.random.normal(0, 2, size=2)
    cens_x_rv = at.clip(x_rv, lb_rv, 1)

    lb_vv = lb_rv.clone()
    cens_x_vv = cens_x_rv.clone()

    logp = joint_logprob({cens_x_rv: cens_x_vv, lb_rv: lb_vv})
    assert_no_rvs(logp)


def test_fail_base_and_censored_have_values():
    """Test failure when both base_rv and clipped_rv are given value vars"""
    x_rv = at.random.normal(0, 1)
    cens_x_rv = at.clip(x_rv, x_rv, 1)
    cens_x_rv.name = "cens_x"

    x_vv = x_rv.clone()
    cens_x_vv = cens_x_rv.clone()
    logp_terms = factorized_joint_logprob({cens_x_rv: cens_x_vv, x_rv: x_vv})
    assert cens_x_vv not in logp_terms


def test_fail_multiple_censored_single_base():
    """Test failure when multiple clipped_rvs share a single base_rv"""
    base_rv = at.random.normal(0, 1)
    cens_rv1 = at.clip(base_rv, -1, 1)
    cens_rv1.name = "cens1"
    cens_rv2 = at.clip(base_rv, -1, 1)
    cens_rv2.name = "cens2"

    cens_vv1 = cens_rv1.clone()
    cens_vv2 = cens_rv2.clone()
    logp_terms = factorized_joint_logprob({cens_rv1: cens_vv1, cens_rv2: cens_vv2})
    assert cens_rv2 not in logp_terms


def test_deterministic_clipping():
    x_rv = at.random.normal(0, 1)
    clip = at.clip(x_rv, 0, 0)
    y_rv = at.random.normal(clip, 1)

    x_vv = x_rv.clone()
    y_vv = y_rv.clone()
    logp = joint_logprob({x_rv: x_vv, y_rv: y_vv})
    assert_no_rvs(logp)

    logp_fn = aesara.function([x_vv, y_vv], logp)
    assert np.isclose(
        logp_fn(-1, 1),
        st.norm(0, 1).logpdf(-1) + st.norm(0, 1).logpdf(1),
    )


def test_censored_transform():
    x_rv = at.random.normal(0.5, 1)
    cens_x_rv = at.clip(x_rv, 0, x_rv)

    cens_x_vv = cens_x_rv.clone()

    transform = TransformValuesOpt({cens_x_vv: LogTransform()})
    logp = joint_logprob({cens_x_rv: cens_x_vv}, extra_rewrites=transform)

    cens_x_vv_testval = -1
    obs_logp = logp.eval({cens_x_vv: cens_x_vv_testval})
    exp_logp = (
        sp.stats.norm(0.5, 1).logpdf(np.exp(cens_x_vv_testval)) + cens_x_vv_testval
    )

    assert np.isclose(obs_logp, exp_logp)


class IcdfNormalRV(NormalRV):
    """Normal RV that has icdf but not truncated dispatching"""


class RejectionNormalRV(NormalRV):
    """Normal RV that has neither icdf nor truncated dispatching."""


class IcdfGeometricRV(GeometricRV):
    """Geometric RV that has neither icdf nor truncated dispatching."""


class RejectionGeometricRV(GeometricRV):
    """Geometric RV that has neither icdf nor truncated dispatching."""


icdf_normal = IcdfNormalRV()
rejection_normal = RejectionNormalRV()
icdf_geometric = IcdfGeometricRV()
rejection_geometric = RejectionGeometricRV()


@_truncated.register(IcdfNormalRV)
@_truncated.register(RejectionNormalRV)
@_truncated.register(IcdfGeometricRV)
@_truncated.register(RejectionGeometricRV)
def _truncated_not_implemented(*args, **kwargs):
    raise NotImplementedError()


@_icdf.register(RejectionNormalRV)
@_icdf.register(RejectionGeometricRV)
def _icdf_not_implemented(*args, **kwargs):
    raise NotImplementedError()


def test_truncation_specialized_op():
    x = at.random.uniform(0, 10, name="x", size=100)

    rng = aesara.shared(np.random.RandomState())
    xt, _ = truncate(x, lower=5, upper=15, rng=rng)
    assert isinstance(xt.owner.op, UniformRV)
    assert xt.owner.inputs[0] is rng

    lower_upper = at.stack(xt.owner.inputs[3:])
    assert np.all(lower_upper.eval() == [5, 10])


@pytest.mark.parametrize("lower, upper", [(-1, np.inf), (-1, 1.5), (-np.inf, 1.5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_continuous_random(op_type, lower, upper):
    loc = 0.15
    scale = 10
    normal_op = icdf_normal if op_type == "icdf" else rejection_normal
    x = normal_op(loc, scale, name="x", size=100)

    rng = aesara.shared(np.random.RandomState())
    xt, xt_update = truncate(x, lower=lower, upper=upper, rng=rng)
    assert isinstance(xt.owner.op, TruncatedRV)
    assert xt.owner.inputs[-1] is rng
    assert xt.type == x.type

    # Check that original op can be used on its own
    assert x.eval().shape == (100,)

    xt_fn = aesara.function([], xt, updates=xt_update)
    xt_draws = np.array([xt_fn() for _ in range(5)])
    assert np.all(xt_draws >= lower)
    assert np.all(xt_draws <= upper)
    assert np.unique(xt_draws).size == xt_draws.size

    # Compare with reference
    ref_xt = scipy.stats.truncnorm(
        (lower - loc) / scale,
        (upper - loc) / scale,
        loc,
        scale,
    )
    assert scipy.stats.cramervonmises(xt_draws.ravel(), ref_xt.cdf).pvalue > 0.001

    # Test max_n_steps
    xt, xt_update = truncate(x, lower=lower, upper=upper, max_n_steps=1)
    xt_fn = aesara.function([], xt, updates=xt_update)
    if op_type == "icdf":
        xt_draws = xt_fn()
        assert np.all(xt_draws >= lower)
        assert np.all(xt_draws <= upper)
        assert np.unique(xt_draws).size == xt_draws.size
    else:
        with pytest.raises(TruncationError, match="^Truncation did not converge"):
            xt_fn()


@pytest.mark.parametrize("lower, upper", [(-1, np.inf), (-1, 1.5), (-np.inf, 1.5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_continuous_logp(op_type, lower, upper):
    loc = 0.15
    scale = 10
    op = icdf_normal if op_type == "icdf" else rejection_normal

    x = op(loc, scale, name="x")
    xt, _ = truncate(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)

    xt_vv = xt.clone()
    xt_logp_fn = aesara.function([xt_vv], joint_logprob({xt: xt_vv}))

    ref_xt = scipy.stats.truncnorm(
        (lower - loc) / scale,
        (upper - loc) / scale,
        loc,
        scale,
    )
    for bound in (lower, upper):
        if np.isinf(bound):
            return
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logp_fn(test_xt_v), ref_xt.logpdf(test_xt_v))


@pytest.mark.parametrize("lower, upper", [(2, np.inf), (2, 5), (-np.inf, 5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_discrete_random(op_type, lower, upper):
    p = 0.2
    geometric_op = icdf_geometric if op_type == "icdf" else rejection_geometric

    x = geometric_op(p, name="x", size=500)
    xt, xt_update = truncate(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)
    assert xt.type == x.type

    xt_draws = aesara.function([], xt, updates=xt_update)()
    assert np.all(xt_draws >= lower)
    assert np.all(xt_draws <= upper)
    assert np.any(xt_draws == (max(1, lower)))
    if upper != np.inf:
        assert np.any(xt_draws == upper)

    # Test max_n_steps
    xt, xt_update = truncate(x, lower=lower, upper=upper, max_n_steps=3)
    xt_fn = aesara.function([], xt, updates=xt_update)
    if op_type == "icdf":
        xt_draws = xt_fn()
        assert np.all(xt_draws >= lower)
        assert np.all(xt_draws <= upper)
        assert np.any(xt_draws == (max(1, lower)))
        if upper != np.inf:
            assert np.any(xt_draws == upper)
    else:
        with pytest.raises(TruncationError, match="^Truncation did not converge"):
            xt_fn()


@pytest.mark.parametrize("lower, upper", [(2, np.inf), (2, 5), (-np.inf, 5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_discrete_logp(op_type, lower, upper):
    p = 0.7
    op = icdf_geometric if op_type == "icdf" else rejection_geometric

    x = op(p, name="x")
    xt, _ = truncate(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)

    xt_vv = xt.clone()
    xt_logp_fn = aesara.function([xt_vv], logprob(xt, xt_vv))

    ref_xt = st.geom(p)
    log_norm = np.log(ref_xt.cdf(upper) - ref_xt.cdf(lower - 1))

    def ref_xt_logpmf(value):
        if value < lower or value > upper:
            return -np.inf
        return ref_xt.logpmf(value) - log_norm

    for bound in (lower, upper):
        if np.isinf(bound):
            continue
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logp_fn(test_xt_v), ref_xt_logpmf(test_xt_v))

    # Check that it integrates to 1
    log_integral = scipy.special.logsumexp(
        [xt_logp_fn(v) for v in range(min(upper + 1, 20))]
    )
    assert np.isclose(log_integral, 0.0, atol=1e-5)


def test_truncation_exceptions():
    with pytest.raises(ValueError, match="lower and upper cannot both be None"):
        truncate(at.random.normal())

    with pytest.raises(NotImplementedError, match="Truncation is only implemented for"):
        truncate(at.clip(at.random.normal(), -1, 1), -1, 1)

    with pytest.raises(NotImplementedError, match="Truncation is only implemented for"):
        truncate(at.random.dirichlet([1, 1, 1]), -1, 1)


def test_truncation_bound_check():
    x = at.random.normal(name="x")
    xt, _ = truncate(x, lower=5, upper=-5)
    xt_vv = xt.clone()
    with pytest.raises(ParameterValueError):
        logprob(xt, xt_vv).eval({xt_vv: 0})
