import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy as sp
import scipy.special
from aesara.graph.fg import FunctionGraph
from numdifftools import Derivative, Jacobian

from aeppl.joint_logprob import DensityNotFound, conditional_logprob, joint_logprob
from aeppl.transforms import (
    DEFAULT_TRANSFORM,
    ChainedTransform,
    ExpTransform,
    IntervalTransform,
    LocTransform,
    LogOddsTransform,
    LogTransform,
    RVTransform,
    ScaleTransform,
    TransformValuesMapping,
    TransformValuesRewrite,
    _default_transformed_rv,
)
from tests.utils import assert_no_rvs


class DirichletScipyDist:
    def __init__(self, alphas):
        self.alphas = alphas

    def rvs(self, size=None, random_state=None):
        if size is None:
            size = ()
        samples_shape = tuple(np.atleast_1d(size)) + self.alphas.shape
        samples = np.empty(samples_shape)
        alphas_bcast = np.broadcast_to(self.alphas, samples_shape)

        for index in np.ndindex(*samples_shape[:-1]):
            samples[index] = random_state.dirichlet(alphas_bcast[index])

        return samples

    def logpdf(self, value):
        res = np.sum(
            scipy.special.xlogy(self.alphas - 1, value)
            - scipy.special.gammaln(self.alphas),
            axis=-1,
        ) + scipy.special.gammaln(np.sum(self.alphas, axis=-1))
        return res


@pytest.mark.parametrize(
    "at_dist, dist_params, sp_dist, size",
    [
        (at.random.uniform, (0, 1), sp.stats.uniform, ()),
        (
            at.random.pareto,
            (1.5, 10.5),
            lambda b, scale: sp.stats.pareto(b, scale=scale),
            (),
        ),
        (
            at.random.triangular,
            (1.5, 3.0, 10.5),
            lambda lower, mode, upper: sp.stats.triang(
                (mode - lower) / (upper - lower), loc=lower, scale=upper - lower
            ),
            (),
        ),
        (
            at.random.halfnormal,
            (0, 1),
            sp.stats.halfnorm,
            (),
        ),
        (
            at.random.wald,
            (1.5, 10.5),
            lambda mean, scale: sp.stats.invgauss(mean / scale, scale=scale),
            (),
        ),
        (
            at.random.exponential,
            (1.5,),
            lambda mu: sp.stats.expon(scale=mu),
            (),
        ),
        pytest.param(
            at.random.lognormal,
            (-1.5, 10.5),
            lambda mu, sigma: sp.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu)),
            (),
        ),
        (
            at.random.lognormal,
            (-1.5, 1.5),
            lambda mu, sigma: sp.stats.lognorm(s=sigma, scale=np.exp(mu)),
            (),
        ),
        (
            at.random.halfcauchy,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.halfcauchy(loc=alpha, scale=beta),
            (),
        ),
        (
            at.random.gamma,
            (1.5, 10.5),
            lambda alpha, inv_beta: sp.stats.gamma(alpha, scale=1.0 / inv_beta),
            (),
        ),
        (
            at.random.invgamma,
            (1.5, 10.5),
            lambda alpha, beta: sp.stats.invgamma(alpha, scale=beta),
            (),
        ),
        (
            at.random.chisquare,
            (1.5,),
            lambda df: sp.stats.chi2(df),
            (),
        ),
        (
            at.random.weibull,
            (1.5,),
            lambda c: sp.stats.weibull_min(c),
            (),
        ),
        (
            at.random.beta,
            (1.5, 1.5),
            lambda alpha, beta: sp.stats.beta(alpha, beta),
            (),
        ),
        (
            at.random.vonmises,
            (1.5, 10.5),
            lambda mu, kappa: sp.stats.vonmises(kappa, loc=mu),
            (),
        ),
        (
            at.random.dirichlet,
            (np.array([0.7, 0.3]),),
            lambda alpha: sp.stats.dirichlet(alpha),
            (),
        ),
        (
            at.random.dirichlet,
            (np.array([[0.7, 0.3], [0.9, 0.1]]),),
            lambda alpha: DirichletScipyDist(alpha),
            (),
        ),
        pytest.param(
            at.random.dirichlet,
            (np.array([0.3, 0.7]),),
            lambda alpha: DirichletScipyDist(alpha),
            (3, 2),
        ),
    ],
)
def test_transformed_logprob(at_dist, dist_params, sp_dist, size):
    """
    This test takes a `RandomVariable` type, plus parameters, and uses it to
    construct a variable ``a`` that's used in the graph ``b =
    at.random.normal(a, 1.0)``.  The transformed log-probability is then
    computed for ``b``.  We then test that the log-probability of ``a`` is
    properly transformed, as well as any instances of ``a`` that are used
    elsewhere in the graph (i.e. in ``b``), by comparing the graph for the
    transformed log-probability with the SciPy-derived log-probability--using a
    numeric approximation to the Jacobian term.
    """

    a = at_dist(*dist_params, size=size)
    a.name = "a"

    b = at.random.normal(a, 1.0)
    b.name = "b"

    transform_rewrite = TransformValuesRewrite({a: DEFAULT_TRANSFORM})
    res, (b_value_var, a_value_var) = joint_logprob(
        b, a, extra_rewrites=transform_rewrite
    )

    test_val_rng = np.random.RandomState(3238)

    logp_vals_fn = aesara.function([a_value_var, b_value_var], res)

    a_trans_op = _default_transformed_rv(a.owner.op, a.owner).op
    transform = a_trans_op.transform

    # Remove the static shape assumptions from the value variable so that it's
    # easier to construct the numerical Jacobian reference values in higher
    # dimensions
    a_value_var_gen = at.tensor(
        dtype=a_value_var.type.dtype, shape=(None,) * a_value_var.type.ndim
    )
    a_forward_fn = aesara.function(
        [a_value_var_gen], transform.forward(a_value_var_gen, *a.owner.inputs)
    )
    a_backward_fn = aesara.function(
        [a_value_var_gen], transform.backward(a_value_var_gen, *a.owner.inputs)
    )
    log_jac_fn = aesara.function(
        [a_value_var],
        transform.log_jac_det(a_value_var, *a.owner.inputs),
        on_unused_input="ignore",
    )

    for i in range(10):
        a_dist = sp_dist(*dist_params)
        a_val = a_dist.rvs(size=size, random_state=test_val_rng).astype(
            a_value_var.dtype
        )
        b_dist = sp.stats.norm(a_val, 1.0)
        b_val = b_dist.rvs(random_state=test_val_rng).astype(b_value_var.dtype)

        a_trans_value = a_forward_fn(a_val)

        if a_val.ndim > 0:

            def jacobian_estimate_novec(value):

                dim_diff = a_val.ndim - value.ndim
                if dim_diff > 0:
                    # Make sure the dimensions match the expected input
                    # dimensions for the compiled backward transform function
                    def a_backward_fn_(x):
                        x_ = np.expand_dims(x, axis=list(range(dim_diff)))
                        return a_backward_fn(x_).squeeze()

                else:
                    a_backward_fn_ = a_backward_fn

                jacobian_val = Jacobian(a_backward_fn_)(value)

                n_missing_dims = jacobian_val.shape[0] - jacobian_val.shape[1]
                if n_missing_dims > 0:
                    missing_bases = np.eye(jacobian_val.shape[0])[..., -n_missing_dims:]
                    jacobian_val = np.concatenate(
                        [jacobian_val, missing_bases], axis=-1
                    )

                return np.linalg.slogdet(jacobian_val)[-1]

            jacobian_estimate = np.vectorize(
                jacobian_estimate_novec, signature="(n)->()"
            )

            exp_log_jac_val = jacobian_estimate(a_trans_value)
        else:
            jacobian_val = np.atleast_2d(Derivative(a_backward_fn)(a_trans_value))
            exp_log_jac_val = np.linalg.slogdet(jacobian_val)[-1]

        log_jac_val = log_jac_fn(a_trans_value)
        np.testing.assert_almost_equal(exp_log_jac_val, log_jac_val, decimal=4)

        exp_logprob_val = a_dist.logpdf(a_val).sum()
        exp_logprob_val += exp_log_jac_val.sum()
        exp_logprob_val += b_dist.logpdf(b_val).sum()

        logprob_val = logp_vals_fn(a_trans_value, b_val)

        np.testing.assert_almost_equal(exp_logprob_val, logprob_val, decimal=4)


@pytest.mark.parametrize("use_jacobian", [True, False])
def test_simple_transformed_logprob_nojac(use_jacobian):
    X_rv = at.random.halfnormal(0, 3, name="X")

    transform_rewrite = TransformValuesRewrite({X_rv: DEFAULT_TRANSFORM})
    tr_logp, (x_vv,) = joint_logprob(
        X_rv,
        extra_rewrites=transform_rewrite,
        use_jacobian=use_jacobian,
    )

    assert np.isclose(
        tr_logp.eval({x_vv: np.log(2.5)}),
        sp.stats.halfnorm(0, 3).logpdf(2.5) + (np.log(2.5) if use_jacobian else 0.0),
    )


@pytest.mark.parametrize("ndim", (0, 1))
def test_fallback_log_jac_det(ndim):
    """
    Test fallback log_jac_det in RVTransform produces correct the graph for a
    simple transformation: x**2 -> -log(2*x)
    """

    class SquareTransform(RVTransform):
        name = "square"

        def forward(self, value, *inputs):
            return at.power(value, 2)

        def backward(self, value, *inputs):
            return at.sqrt(value)

    square_tr = SquareTransform()

    value = at.TensorType("float64", (None,) * ndim)("value")
    value_tr = square_tr.forward(value)
    log_jac_det = square_tr.log_jac_det(value_tr)

    test_value = np.full((2,) * ndim, 3)
    expected_log_jac_det = -np.log(6) * test_value.size
    assert np.isclose(log_jac_det.eval({value: test_value}), expected_log_jac_det)


def test_hierarchical_uniform_transform():
    """
    This model requires rv-value replacements in the backward transformation of
    the value var `x`
    """

    lower_rv = at.random.uniform(0, 1, name="lower")
    upper_rv = at.random.uniform(9, 10, name="upper")
    x_rv = at.random.uniform(lower_rv, upper_rv, name="x")

    transform_rewrite = TransformValuesRewrite(
        {
            lower_rv: DEFAULT_TRANSFORM,
            upper_rv: DEFAULT_TRANSFORM,
            x_rv: DEFAULT_TRANSFORM,
        }
    )
    logp, (lower, upper, x) = joint_logprob(
        lower_rv,
        upper_rv,
        x_rv,
        extra_rewrites=transform_rewrite,
    )

    assert_no_rvs(logp)
    assert not np.isinf(logp.eval({lower: -10, upper: 20, x: -20}))


def test_nondefault_transforms():
    loc_rv = at.random.uniform(-10, 10, name="loc")
    scale_rv = at.random.uniform(-1, 1, name="scale")
    x_rv = at.random.normal(loc_rv, scale_rv, name="x")

    transform_rewrite = TransformValuesRewrite(
        {
            loc_rv: None,
            scale_rv: LogOddsTransform(),
            x_rv: LogTransform(),
        }
    )

    logp, (loc, scale, x) = joint_logprob(
        loc_rv,
        scale_rv,
        x_rv,
        extra_rewrites=transform_rewrite,
    )

    # Check numerical evaluation matches with expected transforms
    loc_val = 0
    scale_val_tr = -1
    x_val_tr = -1

    scale_val = sp.special.expit(scale_val_tr)
    x_val = np.exp(x_val_tr)

    exp_logp = 0
    exp_logp += sp.stats.uniform(-10, 20).logpdf(loc_val)
    exp_logp += sp.stats.uniform(-1, 2).logpdf(scale_val)
    exp_logp += np.log(scale_val) + np.log1p(-scale_val)  # logodds log_jac_det
    exp_logp += sp.stats.norm(loc_val, scale_val).logpdf(x_val)
    exp_logp += x_val_tr  # log log_jac_det

    assert np.isclose(
        logp.eval({loc: loc_val, scale: scale_val_tr, x: x_val_tr}),
        exp_logp,
    )


def test_default_transform_multiout():
    r"""Make sure that `Op`\s with multiple outputs are handled correctly."""

    # This SVD value is necessarily `1`, but it's generated by an `Op` with
    # multiple outputs and no default output.
    sd = at.linalg.svd(at.eye(1))[1][0]
    x_rv = at.random.normal(0, sd, name="x")

    transform_rewrite = TransformValuesRewrite({x_rv: DEFAULT_TRANSFORM})

    logp, (x,) = joint_logprob(
        x_rv,
        extra_rewrites=transform_rewrite,
    )

    assert np.isclose(
        logp.eval({x: 1}),
        sp.stats.norm(0, 1).logpdf(1),
    )


def test_nonexistent_default_transform():
    """
    Test that setting `DEFAULT_TRANSFORM` to a variable that has no default
    transform does not fail
    """
    x_rv = at.random.normal(name="x")

    transform_rewrite = TransformValuesRewrite({x_rv: DEFAULT_TRANSFORM})

    logp, (x,) = joint_logprob(
        x_rv,
        extra_rewrites=transform_rewrite,
    )

    assert np.isclose(
        logp.eval({x: 1}),
        sp.stats.norm(0, 1).logpdf(1),
    )


def test_TransformValuesMapping():
    x = at.vector()
    fg = FunctionGraph(outputs=[x])

    tvm = TransformValuesMapping({}, {})
    fg.attach_feature(tvm)

    tvm2 = TransformValuesMapping({}, {})
    fg.attach_feature(tvm2)

    assert fg._features[-1] is tvm


def test_original_values_output_dict():
    """
    Test that the original unconstrained value variable appears an the key of
    the logprob factor
    """
    p_rv = at.random.beta(1, 1, name="p")

    tr = TransformValuesRewrite({p_rv: DEFAULT_TRANSFORM})
    logp_dict, _ = conditional_logprob(p_rv, extra_rewrites=tr)

    assert p_rv in logp_dict


def test_mixture_transform():
    """Make sure that non-`RandomVariable` `MeasurableVariable`s can be transformed.

    This test is specific to `MixtureRV`, which is derived from an `OpFromGraph`.
    """

    I_rv = at.random.bernoulli(0.5, name="I")
    Y_1_rv = at.random.beta(100, 1, name="Y_1")
    Y_2_rv = at.random.beta(1, 100, name="Y_2")

    # A `MixtureRV`, which is an `OpFromGraph` subclass, will replace this
    # `at.stack` in the graph
    Y_rv = at.stack([Y_1_rv, Y_2_rv])[I_rv]
    Y_rv.name = "Y"

    logp, (y_vv, i_vv) = joint_logprob(
        Y_rv,
        I_rv,
    )

    transform_rewrite = TransformValuesRewrite({Y_rv: LogOddsTransform()})

    logp_trans, (y_vv_trans, i_vv_trans) = joint_logprob(
        Y_rv,
        I_rv,
        extra_rewrites=transform_rewrite,
        use_jacobian=False,
    )

    logp_fn = aesara.function((i_vv, y_vv), logp)
    logp_trans_fn = aesara.function((i_vv_trans, y_vv_trans), logp_trans)
    np.isclose(logp_trans_fn(0, np.log(0.1 / 0.9)), logp_fn(0, 0.1))
    np.isclose(logp_trans_fn(1, np.log(0.1 / 0.9)), logp_fn(1, 0.1))


def test_invalid_interval_transform():
    x_rv = at.random.normal(0, 1)
    x_vv = x_rv.clone()

    msg = "Both edges of IntervalTransform cannot be None"
    tr = IntervalTransform(lambda *inputs: (None, None))
    with pytest.raises(ValueError, match=msg):
        tr.forward(x_vv, *x_rv.owner.inputs)

    tr = IntervalTransform(lambda *inputs: (None, None))
    with pytest.raises(ValueError, match=msg):
        tr.backward(x_vv, *x_rv.owner.inputs)

    tr = IntervalTransform(lambda *inputs: (None, None))
    with pytest.raises(ValueError, match=msg):
        tr.log_jac_det(x_vv, *x_rv.owner.inputs)


def test_chained_transform():
    loc = 5
    scale = 0.1

    ch = ChainedTransform(
        transform_list=[
            ScaleTransform(
                transform_args_fn=lambda *inputs: at.constant(scale),
            ),
            ExpTransform(),
            LocTransform(
                transform_args_fn=lambda *inputs: at.constant(loc),
            ),
        ],
        base_op=at.random.multivariate_normal,
    )

    x = at.random.multivariate_normal(np.zeros(3), np.eye(3))
    x_val = x.eval()

    x_val_forward = ch.forward(x_val, *x.owner.inputs).eval()
    assert np.allclose(
        x_val_forward,
        np.exp(x_val * scale) + loc,
    )

    x_val_backward = ch.backward(x_val_forward, *x.owner.inputs, scale, loc).eval()
    assert np.allclose(
        x_val_backward,
        x_val,
    )

    log_jac_det = ch.log_jac_det(x_val_forward, *x.owner.inputs, scale, loc)
    assert np.isclose(
        log_jac_det.eval(),
        -np.log(scale) - np.sum(np.log(x_val_forward - loc)),
    )


def test_exp_transform_rv():
    base_rv = at.random.normal(0, 1, size=2, name="base_rv")
    y_rv = at.exp(base_rv)
    y_rv.name = "y"

    logps, (y_vv,) = conditional_logprob(y_rv)
    logp = logps[y_rv]
    logp_fn = aesara.function([y_vv], logp)

    y_val = [0.1, 0.3]
    np.testing.assert_allclose(
        logp_fn(y_val),
        sp.stats.lognorm(s=1).logpdf(y_val),
    )


def test_log_transform_rv():
    base_rv = at.random.lognormal(0, 1, size=2, name="base_rv")
    y_rv = at.log(base_rv)
    y_rv.name = "y"

    logps, (y_vv,) = conditional_logprob(y_rv)
    logp = logps[y_rv]
    logp_fn = aesara.function([y_vv], logp)

    y_val = [0.1, 0.3]
    np.testing.assert_allclose(
        logp_fn(y_val),
        sp.stats.norm().logpdf(y_val),
    )


@pytest.mark.parametrize(
    "rv_size, loc_type",
    [
        (None, at.scalar),
        (2, at.vector),
        ((2, 1), at.col),
    ],
)
@pytest.mark.parametrize("right", [True, False])
def test_transform_measurable_add(rv_size, loc_type, right):

    loc = loc_type("loc")
    X_rv = at.random.normal(0, 1, size=rv_size, name="X")
    if right:
        Z_rv = loc + X_rv
    else:
        Z_rv = X_rv + loc

    logps, (z_vv,) = conditional_logprob(Z_rv)
    logp = logps[Z_rv]
    assert_no_rvs(logp)
    logp_fn = aesara.function([loc, z_vv], logp)

    loc_test_val = np.full(rv_size or (), 4.0)
    z_test_val = np.full(rv_size or (), 1.0)

    np.testing.assert_allclose(
        logp_fn(loc_test_val, z_test_val),
        sp.stats.norm(loc_test_val, 1).logpdf(z_test_val),
    )


@pytest.mark.parametrize(
    "rv_size, scale_type",
    [
        (None, at.scalar),
        (1, at.TensorType("floatX", (True,))),
        ((2, 3), at.matrix),
    ],
)
@pytest.mark.parametrize("right", [True, False])
def test_scale_transform_rv(rv_size, scale_type, right):

    scale = scale_type("scale")
    X_rv = at.random.normal(0, 1, size=rv_size, name="X")
    if right:
        Z_rv = scale * X_rv
    else:
        # Z_rv = at.random.normal(0, 1, size=rv_size, name="base_rv") * scale
        Z_rv = X_rv / at.reciprocal(scale)

    logps, (z_vv,) = conditional_logprob(Z_rv)
    logp = logps[Z_rv]
    assert_no_rvs(logp)
    logp_fn = aesara.function([scale, z_vv], logp)

    scale_test_val = np.full(rv_size or (), 4.0)
    z_val = np.full(rv_size or (), 1.0)

    np.testing.assert_allclose(
        logp_fn(scale_test_val, z_val),
        sp.stats.norm(0, scale_test_val).logpdf(z_val),
    )


def test_transformed_rv_and_value():
    y_rv = at.random.halfnormal(-1, 1, name="base_rv") + 1
    y_rv.name = "y"

    transform_rewrite = TransformValuesRewrite({y_rv: LogTransform()})

    logp, (y_vv,) = joint_logprob(y_rv, extra_rewrites=transform_rewrite)
    assert_no_rvs(logp)
    logp_fn = aesara.function([y_vv], logp)

    y_test_val = -5

    assert np.isclose(
        logp_fn(y_test_val),
        sp.stats.halfnorm(0, 1).logpdf(np.exp(y_test_val)) + y_test_val,
    )


def test_loc_transform_multiple_rvs_fails1():
    x_rv1 = at.random.normal(name="x_rv1")
    x_rv2 = at.random.normal(name="x_rv2")
    y_rv = x_rv1 + x_rv2

    with pytest.raises(DensityNotFound):
        joint_logprob(y_rv)


def test_nested_loc_transform_multiple_rvs_fails2():
    x_rv1 = at.random.normal(name="x_rv1")
    x_rv2 = at.cos(at.random.normal(name="x_rv2"))
    y_rv = x_rv1 + x_rv2

    with pytest.raises(DensityNotFound):
        joint_logprob(y_rv)


def test_discrete_rv_unary_transform_fails():
    y_rv = at.exp(at.random.poisson(1))
    with pytest.raises(DensityNotFound):
        joint_logprob(y_rv)


def test_discrete_rv_multinary_transform_fails():
    y_rv = 5 + at.random.poisson(1)
    with pytest.raises(DensityNotFound):
        joint_logprob(y_rv)


def test_invalid_broadcasted_transform_rv_fails():
    loc = at.vector("loc")
    y_rv = loc + at.random.normal(0, 1, size=2, name="base_rv")
    y_rv.name = "y"

    logp, (y_vv,) = joint_logprob(y_rv)

    with pytest.raises(TypeError):
        logp.eval({y_vv: [0, 0, 0, 0], loc: [0, 0, 0, 0]})


@pytest.mark.parametrize("a", (1.0, 2.0))
def test_transform_measurable_true_div(a):
    shape, scale = 3, 5
    X_rv = at.random.gamma(shape, scale, name="X")

    Z_rv = a / X_rv

    logp, (z_vv,) = joint_logprob(Z_rv)
    z_logp_fn = aesara.function([z_vv], logp)

    z_test_val = 1.5
    assert np.isclose(
        z_logp_fn(z_test_val),
        sp.stats.invgamma(shape, scale=scale * a).logpdf(z_test_val),
    )

    Z_rv = X_rv / a

    logp, (z_vv,) = joint_logprob(Z_rv)
    z_logp_fn = aesara.function([z_vv], logp)

    z_test_val = 1.5
    assert np.isclose(
        z_logp_fn(z_test_val),
        sp.stats.gamma(shape, scale=1 / (scale * a)).logpdf(z_test_val),
    )


def test_transform_measurable_neg():
    X_rv = at.random.halfnormal(name="X")
    Z_rv = -X_rv

    logp, (z_vv,) = joint_logprob(Z_rv)
    z_logp_fn = aesara.function([z_vv], logp)

    assert np.isclose(z_logp_fn(-1.5), sp.stats.halfnorm.logpdf(1.5))


def test_transform_measurable_sub():
    # We use a base RV that is asymmetric around zero
    X_rv = at.random.normal(1.0, name="X")

    Z_rv = 5.0 - X_rv

    logp, (z_vv,) = joint_logprob(Z_rv)
    z_logp_fn = aesara.function([z_vv], logp)
    assert np.isclose(z_logp_fn(7.3), sp.stats.norm.logpdf(5.0 - 7.3, 1.0))

    Z_rv = X_rv - 5.0

    logp, (z_vv,) = joint_logprob(Z_rv)
    z_logp_fn = aesara.function([z_vv], logp)
    assert np.isclose(z_logp_fn(7.3), sp.stats.norm.logpdf(7.3, loc=-4.0))


def test_transform_reused_measurable():

    srng = at.random.RandomStream(0)

    X_rv = srng.normal(0, 1, name="X")
    Z_tr = at.exp(X_rv)

    z_vv = at.dscalar(name="z_vv")

    logprob, vvs = joint_logprob(realized={Z_tr: z_vv, X_rv: z_vv})

    logp_fn = aesara.function([z_vv], logprob)

    z_val = 0.1

    exp_res = sp.stats.lognorm(s=1).logpdf(z_val) + sp.stats.norm().logpdf(z_val)

    np.testing.assert_allclose(logp_fn(z_val), exp_res)
