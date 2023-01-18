import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats
import scipy.stats as st
from aesara.tensor.random.basic import GeometricRV, NormalRV, UniformRV

from aeppl import joint_logprob, logprob
from aeppl.logprob import ParameterValueError, _icdf
from aeppl.truncation import TruncatedRV, TruncationError, _truncated, truncate


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

    srng = at.random.RandomStream()
    xt, _ = truncate(x, lower=5, upper=15, srng=srng)
    assert isinstance(xt.owner.op, UniformRV)
    assert xt.owner.inputs[0] is srng.updates()[0][0]

    lower_upper = at.stack(xt.owner.inputs[3:])
    assert np.all(lower_upper.eval() == [5, 10])


@pytest.mark.filterwarnings("ignore:Rewrite warning")
@pytest.mark.parametrize("lower, upper", [(-1, np.inf), (-1, 1.5), (-np.inf, 1.5)])
@pytest.mark.parametrize("op_type", ["icdf", "rejection"])
def test_truncation_continuous_random(op_type, lower, upper):
    loc = 0.15
    scale = 10
    normal_op = icdf_normal if op_type == "icdf" else rejection_normal
    x = normal_op(loc, scale, name="x", size=100)

    srng = at.random.RandomStream()
    xt, xt_update = truncate(x, lower=lower, upper=upper, srng=srng)
    assert isinstance(xt.owner.op, TruncatedRV)
    assert xt.owner.inputs[-1] is srng.updates()[1 if op_type == "icdf" else 2][0]
    assert xt.type.dtype == x.type.dtype
    assert xt.type.ndim == x.type.ndim

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
    xt, xt_update = truncate(x, lower=lower, upper=upper, max_n_steps=2)
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

    logp, (xt_vv,) = joint_logprob(xt)
    xt_logp_fn = aesara.function([xt_vv], logp)

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
    assert xt.type.is_super(x.type)

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
