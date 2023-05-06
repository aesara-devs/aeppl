import aesara
import aesara.tensor as at
import numpy as np
import scipy.stats as st

from aeppl import conditional_logprob, joint_logprob
from aeppl.censoring import MeasurableClip
from aeppl.rewriting import construct_ir_fgraph
from tests.utils import assert_no_rvs


def test_scalar_clipped_mixture():
    srng = at.random.RandomStream(0)

    x = at.clip(srng.normal(loc=1), 0.5, 1.5)
    x.name = "x"
    y = srng.beta(1, 2, name="y")

    comps = at.stack([x, y])
    comps.name = "comps"
    idxs = srng.bernoulli(0.4, name="idxs")
    mix = comps[idxs]
    mix.name = "mix"

    logp, (idxs_vv, mix_vv) = joint_logprob(idxs, mix)

    logp_fn = aesara.function([idxs_vv, mix_vv], logp)
    assert logp_fn(0, 0.4) == -np.inf
    assert np.isclose(logp_fn(0, 0.5), st.norm.logcdf(0.5, 1) + np.log(0.6))
    assert np.isclose(logp_fn(0, 1.3), st.norm.logpdf(1.3, 1) + np.log(0.6))
    assert np.isclose(logp_fn(1, 0.4), st.beta.logpdf(0.4, 1, 2) + np.log(0.4))


def test_nested_scalar_mixtures():
    srng = at.random.RandomStream(0)

    x = srng.normal(loc=-50, name="x")
    y = srng.normal(loc=50, name="y")
    comps1 = at.stack([x, y])
    comps1.name = "comps1"
    idxs1 = srng.bernoulli(0.5, name="idxs1")
    mix1 = comps1[idxs1]
    mix1.name = "mix1"

    w = srng.normal(loc=-100, name="w")
    z = srng.normal(loc=100, name="z")
    comps2 = at.stack([w, z])
    comps2.name = "comps2"
    idxs2 = srng.bernoulli(0.5, name="idxs2")
    mix2 = comps2[idxs2]
    mix2.name = "mix2"

    comps12 = at.stack([mix1, mix2])
    comps12.name = "comps12"
    idxs12 = srng.bernoulli(0.5, name="idxs12")
    mix12 = comps12[idxs12]
    mix12.name = "mix12"

    logp, (idxs1_vv, idxs2_vv, idxs12_vv, mix12_vv) = joint_logprob(
        idxs1, idxs2, idxs12, mix12
    )
    logp_fn = aesara.function([idxs1_vv, idxs2_vv, idxs12_vv, mix12_vv], logp)

    expected_mu_logpdf = st.norm.logpdf(0) + np.log(0.5) * 3
    assert np.isclose(logp_fn(0, 0, 0, -50), expected_mu_logpdf)
    assert np.isclose(logp_fn(0, 1, 0, -50), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 0, 0, 50), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 1, 0, 50), expected_mu_logpdf)
    assert np.isclose(logp_fn(0, 0, 1, -100), expected_mu_logpdf)
    assert np.isclose(logp_fn(0, 1, 1, 100), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 0, 1, -100), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 1, 1, 100), expected_mu_logpdf)

    assert np.isclose(logp_fn(0, 0, 0, 50), st.norm.logpdf(100) + np.log(0.5) * 3)
    assert np.isclose(logp_fn(0, 0, 1, 50), st.norm.logpdf(150) + np.log(0.5) * 3)


def test_unvalued_ir_reversion():
    """Make sure that un-valued IR rewrites are reverted."""
    srng = at.random.RandomStream(0)

    x_rv = srng.normal(name="X")
    y_rv = at.clip(x_rv, 0, 1)
    y_rv.name = "Y"
    z_rv = srng.normal(y_rv, 1, name="Z")
    z_vv = z_rv.clone()
    z_vv.name = "z"

    # Only the `z_rv` is "valued", so `y_rv` doesn't need to be converted into
    # measurable IR.
    rv_values = {z_rv: z_vv}

    z_fgraph, new_rvs_to_values = construct_ir_fgraph(rv_values)

    assert not any(isinstance(node.op, MeasurableClip) for node in z_fgraph.apply_nodes)


def test_shifted_cumsum():
    srng = at.random.RandomStream(0)

    x = srng.normal(size=(5,), name="x")
    y = 5 + at.cumsum(x)
    y.name = "y"

    y_vv = y.clone()
    logp, (y_vv,) = joint_logprob(y)
    assert np.isclose(
        logp.eval({y_vv: np.arange(5) + 1 + 5}),
        st.norm.logpdf(1) * 5,
    )


def test_double_log_transform_rv():
    srng = at.random.RandomStream(0)

    base_rv = srng.normal(0, 1)
    y_rv = at.log(at.log(base_rv))
    y_rv.name = "y"

    logp, (y_vv,) = conditional_logprob(y_rv)
    logp_fn = aesara.function([y_vv], logp[y_rv])

    log_log_y_val = np.asarray(0.5)
    log_y_val = np.exp(log_log_y_val)
    y_val = np.exp(log_y_val)
    np.testing.assert_allclose(
        logp_fn(log_log_y_val),
        st.norm().logpdf(y_val) + log_y_val + log_log_y_val,
    )


def test_affine_transform_rv():
    srng = at.random.RandomStream(0)

    loc = at.scalar("loc")
    scale = at.vector("scale")
    rv_size = 3

    y_rv = loc + srng.normal(0, 1, size=rv_size, name="base_rv") * scale
    y_rv.name = "y"

    logp, (y_vv,) = conditional_logprob(y_rv)
    assert_no_rvs(logp[y_rv])
    logp_fn = aesara.function([loc, scale, y_vv], logp[y_rv])

    loc_test_val = 4.0
    scale_test_val = np.full(rv_size, 0.5)
    y_test_val = np.full(rv_size, 1.0)

    np.testing.assert_allclose(
        logp_fn(loc_test_val, scale_test_val, y_test_val),
        st.norm(loc_test_val, scale_test_val).logpdf(y_test_val),
    )


def test_affine_log_transform_rv():
    srng = at.random.RandomStream(0)

    a, b = at.scalars("a", "b")
    base_rv = srng.lognormal(0, 1, name="base_rv", size=(1, 2))
    y_rv = a + at.log(base_rv) * b
    y_rv.name = "y"

    y_vv = y_rv.clone()

    logps, (y_vv,) = conditional_logprob(y_rv)
    logp = logps[y_rv]
    logp_fn = aesara.function([a, b, y_vv], logp)

    a_val = -1.5
    b_val = 3.0
    y_val = [[0.1, 0.1]]

    assert np.allclose(
        logp_fn(a_val, b_val, y_val),
        st.norm(a_val, b_val).logpdf(y_val),
    )
