import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats.distributions as sp
from aesara.graph.basic import ancestors, applys_between, equal_computations
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)

from aeppl.abstract import ValuedVariable
from aeppl.joint_logprob import conditional_logprob, joint_logprob
from aeppl.logprob import logprob
from aeppl.utils import rvs_to_value_vars, walk_model
from tests.utils import assert_no_rvs


def test_joint_logprob_basic():
    # A simple check for when `joint_logprob` is the same as `logprob`
    a = at.random.uniform(0.0, 1.0)
    a.name = "a"

    logps, (a_vv,) = conditional_logprob(a)
    a_logp = logps[a]
    a_logp_exp = logprob(a, a_vv)

    assert equal_computations([a_logp], [a_logp_exp])

    # Let's try a hierarchical model
    sigma = at.random.invgamma(0.5, 0.5)
    Y = at.random.normal(0.0, sigma)

    lls, (Y_vv, sigma_vv) = conditional_logprob(Y, sigma)
    total_ll = lls[sigma] + lls[Y]

    # We need to replace the reference to `sigma` in `Y` with its value
    # variable
    ll_Y = logprob(Y, Y_vv)
    (ll_Y,), _ = rvs_to_value_vars(
        [ll_Y],
        initial_replacements={sigma: sigma_vv},
    )
    total_ll_exp = logprob(sigma, sigma_vv) + ll_Y

    assert equal_computations([total_ll], [total_ll_exp])

    # Now, make sure we can compute a joint log-probability for a hierarchical
    # model with some non-`RandomVariable` nodes
    c = at.random.normal()
    c.name = "c"
    b_l = c * a + 2.0
    b = at.random.uniform(b_l, b_l + 1.0)
    b.name = "b"

    b_logp, (a_vv, b_vv, c_vv) = joint_logprob(a, b, c)

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert_no_rvs(b_logp)

    res_ancestors = list(walk_model((b_logp,), walk_past_rvs=True))
    assert a_vv in res_ancestors
    assert b_vv in res_ancestors
    assert c_vv in res_ancestors


def test_joint_logprob_multi_obs():

    a = at.random.uniform(0.0, 1.0)
    b = at.random.normal(0.0, 1.0)

    logps, (a_vv, b_vv) = conditional_logprob(a, b)
    logp = logps[a] + logps[b]
    logp_exp = logprob(a, a_vv) + logprob(b, b_vv)

    assert equal_computations([logp], [logp_exp])

    x = at.random.normal(0, 1)
    y = at.random.normal(x, 1)

    exp_logp, (x_vv, y_vv) = joint_logprob(x, y)
    logp, _ = joint_logprob(realized={x: x_vv, y: y_vv})

    assert equal_computations([logp], [exp_logp])


def test_joint_logprob_diff_dims():
    M = at.matrix("M")
    x = at.random.normal(0, 1, size=M.shape[1], name="X")
    y = at.random.normal(M.dot(x), 1, name="Y")

    logp, (x_vv, y_vv) = joint_logprob(x, y)

    M_val = np.random.normal(size=(10, 3))
    x_val = np.random.normal(size=(3,))
    y_val = np.random.normal(size=(10,))

    point = {M: M_val, x_vv: x_val, y_vv: y_val}
    logp_val = logp.eval(point)

    exp_logp_val = (
        sp.norm.logpdf(x_val, 0, 1).sum()
        + sp.norm.logpdf(y_val, M_val.dot(x_val), 1).sum()
    )
    assert exp_logp_val == pytest.approx(logp_val)


@pytest.mark.parametrize(
    "indices, size",
    [
        (slice(0, 2), 5),
        (np.r_[True, True, False, False, True], 5),
        (np.r_[0, 1, 4], 5),
        ((np.array([0, 1, 4]), np.array([0, 1, 4])), (5, 5)),
    ],
)
def test_joint_logprob_incsubtensor(indices, size):
    """Make sure we can compute a joint log-probability for ``Y[idx] = data`` where ``Y`` is univariate."""

    rng = np.random.RandomState(232)
    mu = np.power(10, np.arange(np.prod(size))).reshape(size)
    sigma = 0.001
    data = rng.normal(mu[indices], 1.0)
    y_val = rng.normal(mu, sigma, size=size)

    Y_base_rv = at.random.normal(mu, sigma, size=size)
    Y_rv = at.set_subtensor(Y_base_rv[indices], data)
    Y_rv.name = "Y"

    assert isinstance(
        Y_rv.owner.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)
    )

    logp, (Y_vv,) = conditional_logprob(Y_rv)

    obs_logps = logp[Y_rv].eval({Y_vv: y_val})

    y_val_idx = y_val.copy()
    y_val_idx[indices] = data
    exp_obs_logps = sp.norm.logpdf(y_val_idx, mu, sigma)

    np.testing.assert_almost_equal(obs_logps, exp_obs_logps)


def test_incsubtensor_original_values_output_dict():
    """
    Test that the original un-incsubtensor value variable appears an the key of
    the logprob factor
    """

    base_rv = at.random.normal(0, 1, size=2)
    rv = at.set_subtensor(base_rv[0], 5)

    logp_dict, _ = conditional_logprob(rv)
    assert rv in logp_dict


def test_joint_logprob_subtensor():
    """Make sure we can compute a joint log-probability for ``Y[I]`` where ``Y`` and ``I`` are random variables."""

    size = 5

    mu_base = np.power(10, np.arange(np.prod(size))).reshape(size)
    mu = np.stack([mu_base, -mu_base])
    sigma = 0.001
    rng = aesara.shared(np.random.RandomState(232), borrow=True)

    A_rv = at.random.normal(mu, sigma, rng=rng)
    A_rv.name = "A"

    p = 0.5

    I_rv = at.random.bernoulli(p, size=size, rng=rng)
    I_rv.name = "I"

    A_idx = A_rv[I_rv, at.ogrid[A_rv.shape[-1] :]]

    assert isinstance(
        A_idx.owner.op, (Subtensor, AdvancedSubtensor, AdvancedSubtensor1)
    )

    A_idx_logps, (A_idx_vv, I_vv) = conditional_logprob(A_idx, I_rv)
    A_idx_logp = at.add(*A_idx_logps.values())

    logp_vals_fn = aesara.function([A_idx_vv, I_vv], A_idx_logp)

    # The compiled graph should not contain any `RandomVariables`
    assert_no_rvs(logp_vals_fn.maker.fgraph.outputs[0])

    decimals = 6 if aesara.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    for i in range(10):
        bern_sp = sp.bernoulli(p)
        I_value = bern_sp.rvs(size=size, random_state=test_val_rng).astype(I_rv.dtype)

        norm_sp = sp.norm(mu[I_value, np.ogrid[mu.shape[1] :]], sigma)
        A_idx_value = norm_sp.rvs(random_state=test_val_rng).astype(A_idx.dtype)

        exp_obs_logps = norm_sp.logpdf(A_idx_value)
        exp_obs_logps += bern_sp.logpmf(I_value)

        logp_vals = logp_vals_fn(A_idx_value, I_value)

        np.testing.assert_almost_equal(logp_vals, exp_obs_logps, decimal=decimals)


def test_persist_inputs():
    """Make sure we don't unnecessarily clone input variables.

    In earlier versions, we preserved the identity of entire value-variable
    graphs in the log-probability output, and not just the inputs of those
    graphs.  Now, since everything is canonicalized, "realized" value-variable
    graphs may not be identical in the output of calls to `conditional_logprob`
    and related functions.

    """
    x = at.scalar("x")
    beta_rv = at.random.normal(0, 1, name="beta")
    Y_rv = at.random.normal(beta_rv * x, 1, name="y")

    logp, (beta_vv, y_vv) = joint_logprob(beta_rv, Y_rv)

    # Make sure an standard input variable is preserved in the output (i.e.  we
    # can't reasonably replace inputs held by the caller).
    assert x in ancestors([logp])

    # Now, we do the same for inputs within a "realized" value-variable graph.
    y_vv_2 = y_vv * 2
    y_vv_2.name = "y_vv_2"

    logp_2, (beta_vv,) = joint_logprob(beta_rv, realized={Y_rv: y_vv_2})

    logp_2_ancestors = tuple(ancestors([logp_2]))
    assert y_vv in logp_2_ancestors
    # The entire "realized" value-variable graph may not be preserved in the
    # output, as is the case here when `y_vv * 2` is canonicalized to `2 *
    # y_vv`.
    # assert y_vv_2 in logp_2_ancestors


def test_random_in_logprob():
    """Make sure we can have `RandomVariable`s in log-probabilities."""

    x_rv = at.random.normal(name="x")
    y_rv = at.random.normal(x_rv, 1, name="y")

    logps, vvars = conditional_logprob(y_rv)

    assert len(vvars) == 1
    assert any(
        var.owner.op == at.random.normal
        for var in ancestors([logps[y_rv]])
        if var.owner
    )


def test_multiple_rvs_with_same_value():
    """Make sure we can use the same value for two different measurable terms."""
    x_rv1 = at.random.normal(name="x1")
    x_rv2 = at.random.normal(name="x2")
    x = x_rv1.clone()
    x.name = "x"

    logps, vvars = conditional_logprob(realized={x_rv1: x, x_rv2: x})

    assert not vvars
    assert equal_computations([logps[x_rv1]], [logps[x_rv2]])


def test_deprecations():
    X = at.random.normal(name="X")
    x = X.clone()
    x.name = "x"

    with pytest.warns(DeprecationWarning):
        conditional_logprob(realized={X: x}, warn_missing_rvs=True)


def test_no_output_ValuedVariables():
    srng = at.random.RandomStream(0)

    X_at = at.matrix("X")
    tau_rv = srng.halfcauchy(1)
    beta_rv = srng.normal(0, tau_rv, size=X_at.shape[-1])

    eta = X_at @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.bernoulli(p)

    logdensity, vvs = joint_logprob(Y_rv, beta_rv, tau_rv)

    assert not any(
        isinstance(node.op, ValuedVariable)
        for node in applys_between(ins=vvs, outs=(logdensity,))
    )
