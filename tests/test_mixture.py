import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats.distributions as sp
from aesara.graph.basic import Variable, equal_computations
from aesara.tensor.random.basic import CategoricalRV
from aesara.tensor.shape import shape_tuple
from aesara.tensor.subtensor import as_index_constant

from aeppl.dists import dirac_delta
from aeppl.joint_logprob import conditional_logprob, joint_logprob
from aeppl.mixture import MixtureRV, expand_indices
from aeppl.rewriting import construct_ir_fgraph
from tests.test_logprob import scipy_logprob
from tests.utils import assert_no_rvs


def test_mixture_basics():
    srng = at.random.RandomStream(29833)

    def create_mix_model(size, axis):
        X_rv = srng.normal(0, 1, size=size, name="X")
        Y_rv = srng.gamma(0.5, 0.5, size=size, name="Y")

        p_at = at.scalar("p")
        p_at.tag.test_value = 0.5

        I_rv = srng.bernoulli(p_at, size=size, name="I")

        if isinstance(axis, Variable):
            M_rv = at.join(axis, X_rv, Y_rv)[I_rv]
        else:
            M_rv = at.stack([X_rv, Y_rv], axis=axis)[I_rv]

        M_rv.name = "M"

        return locals()

    axis_at = at.lscalar("axis")
    axis_at.tag.test_value = 0
    env = create_mix_model((2,), axis_at)
    I_rv = env["I_rv"]
    M_rv = env["M_rv"]

    with pytest.raises(NotImplementedError):
        joint_logprob(M_rv, I_rv)


@aesara.config.change_flags(compute_test_value="warn")
@pytest.mark.parametrize(
    "op_constructor",
    [
        lambda _I, _X, _Y: at.stack([_X, _Y])[_I],
        lambda _I, _X, _Y: at.switch(_I, _X, _Y),
    ],
)
def test_compute_test_value(op_constructor):
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(0, 1, name="X")
    Y_rv = srng.gamma(0.5, 0.5, name="Y")

    p_at = at.scalar("p")
    p_at.tag.test_value = 0.3

    I_rv = srng.bernoulli(p_at, name="I")

    M_rv = op_constructor(I_rv, X_rv, Y_rv)
    M_rv.name = "M"

    M_logps, _ = conditional_logprob(M_rv, I_rv)

    del M_rv.tag.test_value

    M_logp = at.add(*M_logps.values())

    assert isinstance(M_logp.tag.test_value, np.ndarray)


@pytest.mark.parametrize(
    "p_val, size",
    [
        (np.array(0.0, dtype=aesara.config.floatX), ()),
        (np.array(1.0, dtype=aesara.config.floatX), ()),
        (np.array(0.0, dtype=aesara.config.floatX), (2,)),
        (np.array(1.0, dtype=aesara.config.floatX), (2, 1)),
        (np.array(1.0, dtype=aesara.config.floatX), (2, 3)),
        (np.array([0.1, 0.9], dtype=aesara.config.floatX), (2, 3)),
    ],
)
def test_hetero_mixture_binomial(p_val, size):
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(0, 1, size=size, name="X")
    Y_rv = srng.gamma(0.5, 0.5, size=size, name="Y")

    if np.ndim(p_val) == 0:
        p_at = at.scalar("p")
        p_at.tag.test_value = p_val
        I_rv = srng.bernoulli(p_at, size=size, name="I")
        p_val_1 = p_val
    else:
        p_at = at.vector("p")
        p_at.tag.test_value = np.array(p_val, dtype=aesara.config.floatX)
        I_rv = srng.categorical(p_at, size=size, name="I")
        p_val_1 = p_val[1]

    M_rv = at.stack([X_rv, Y_rv])[I_rv]
    M_rv.name = "M"

    M_logps, (M_vv, I_vv) = conditional_logprob(M_rv, I_rv)
    M_logp = at.add(*M_logps.values())

    M_logp_fn = aesara.function([p_at, M_vv, I_vv], M_logp)

    assert_no_rvs(M_logp_fn.maker.fgraph.outputs[0])

    decimals = 6 if aesara.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    bern_sp = sp.bernoulli(p_val_1)
    norm_sp = sp.norm(loc=0, scale=1)
    gamma_sp = sp.gamma(0.5, scale=1.0 / 0.5)

    for i in range(10):
        i_val = bern_sp.rvs(size=size, random_state=test_val_rng)
        x_val = norm_sp.rvs(size=size, random_state=test_val_rng)
        y_val = gamma_sp.rvs(size=size, random_state=test_val_rng)

        component_logps = np.stack([norm_sp.logpdf(x_val), gamma_sp.logpdf(y_val)])[
            i_val
        ]
        exp_obs_logps = component_logps + bern_sp.logpmf(i_val)

        m_val = np.stack([x_val, y_val])[i_val]
        logp_vals = M_logp_fn(p_val, m_val, i_val)

        np.testing.assert_almost_equal(logp_vals, exp_obs_logps, decimal=decimals)


@pytest.mark.parametrize(
    "X_args, Y_args, Z_args, p_val, comp_size, idx_size, extra_indices, join_axis",
    [
        # Scalar mixture components, scalar index
        (
            (
                np.array(0, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array(0.5, dtype=aesara.config.floatX),
                np.array(0.5, dtype=aesara.config.floatX),
            ),
            (
                np.array(100, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=aesara.config.floatX),
            (),
            (),
            (),
            0,
        ),
        # Scalar mixture components, vector index
        (
            (
                np.array(0, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array(0.5, dtype=aesara.config.floatX),
                np.array(0.5, dtype=aesara.config.floatX),
            ),
            (
                np.array(100, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=aesara.config.floatX),
            (),
            (6,),
            (),
            0,
        ),
        (
            (
                np.array([0, -100], dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array([0.5, 1], dtype=aesara.config.floatX),
                np.array([0.5, 1], dtype=aesara.config.floatX),
            ),
            (
                np.array([100, 1000], dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([[0.1, 0.5, 0.4], [0.4, 0.1, 0.5]], dtype=aesara.config.floatX),
            (2,),
            (2,),
            (),
            0,
        ),
        (
            (
                np.array([0, -100], dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array([0.5, 1], dtype=aesara.config.floatX),
                np.array([0.5, 1], dtype=aesara.config.floatX),
            ),
            (
                np.array([100, 1000], dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([[0.1, 0.5, 0.4], [0.4, 0.1, 0.5]], dtype=aesara.config.floatX),
            None,
            None,
            (),
            0,
        ),
        (
            (
                np.array(0, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array(0.5, dtype=aesara.config.floatX),
                np.array(0.5, dtype=aesara.config.floatX),
            ),
            (
                np.array(100, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=aesara.config.floatX),
            (),
            (),
            (),
            0,
        ),
        (
            (
                np.array(0, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array(0.5, dtype=aesara.config.floatX),
                np.array(0.5, dtype=aesara.config.floatX),
            ),
            (
                np.array(100, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=aesara.config.floatX),
            (2,),
            (2,),
            (),
            0,
        ),
        (
            (
                np.array(0, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array(0.5, dtype=aesara.config.floatX),
                np.array(0.5, dtype=aesara.config.floatX),
            ),
            (
                np.array(100, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=aesara.config.floatX),
            (2, 3),
            (2, 3),
            (),
            0,
        ),
        (
            (
                np.array(0, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array(0.5, dtype=aesara.config.floatX),
                np.array(0.5, dtype=aesara.config.floatX),
            ),
            (
                np.array(100, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=aesara.config.floatX),
            (2, 3),
            (),
            (),
            0,
        ),
        (
            (
                np.array(0, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array(0.5, dtype=aesara.config.floatX),
                np.array(0.5, dtype=aesara.config.floatX),
            ),
            (
                np.array(100, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=aesara.config.floatX),
            (3,),
            (3,),
            (slice(None),),
            1,
        ),
        (
            (
                np.array(0, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array(0.5, dtype=aesara.config.floatX),
                np.array(0.5, dtype=aesara.config.floatX),
            ),
            (
                np.array(100, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=aesara.config.floatX),
            (5,),
            (5,),
            (np.arange(5),),
            0,
        ),
        (
            (
                np.array(0, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            (
                np.array(0.5, dtype=aesara.config.floatX),
                np.array(0.5, dtype=aesara.config.floatX),
            ),
            (
                np.array(100, dtype=aesara.config.floatX),
                np.array(1, dtype=aesara.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=aesara.config.floatX),
            (5,),
            (5,),
            (np.arange(5), None),
            0,
        ),
    ],
)
def test_hetero_mixture_categorical(
    X_args, Y_args, Z_args, p_val, comp_size, idx_size, extra_indices, join_axis
):
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(*X_args, size=comp_size, name="X")
    Y_rv = srng.gamma(*Y_args, size=comp_size, name="Y")
    Z_rv = srng.normal(*Z_args, size=comp_size, name="Z")

    p_at = at.as_tensor(p_val).type()
    p_at.name = "p"
    p_at.tag.test_value = np.array(p_val, dtype=aesara.config.floatX)
    I_rv = srng.categorical(p_at, size=idx_size, name="I")

    indices_at = list(extra_indices)
    indices_at.insert(join_axis, I_rv)
    indices_at = tuple(indices_at)

    M_rv = at.stack([X_rv, Y_rv, Z_rv], axis=join_axis)[indices_at]
    M_rv.name = "M"

    logp_parts, (M_vv, I_vv) = conditional_logprob(M_rv, I_rv)

    I_logp_fn = aesara.function([p_at, I_vv], logp_parts[I_rv])
    M_logp_fn = aesara.function([M_vv, I_vv], logp_parts[M_rv])

    assert_no_rvs(I_logp_fn.maker.fgraph.outputs[0])
    assert_no_rvs(M_logp_fn.maker.fgraph.outputs[0])

    decimals = 6 if aesara.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    norm_1_sp = sp.norm(loc=X_args[0], scale=X_args[1])
    gamma_sp = sp.gamma(Y_args[0], scale=1 / Y_args[1])
    norm_2_sp = sp.norm(loc=Z_args[0], scale=Z_args[1])

    for i in range(10):
        i_val = CategoricalRV.rng_fn(test_val_rng, p_val, idx_size)

        indices_val = list(extra_indices)
        indices_val.insert(join_axis, i_val)
        indices_val = tuple(indices_val)

        x_val = norm_1_sp.rvs(size=comp_size, random_state=test_val_rng)
        y_val = gamma_sp.rvs(size=comp_size, random_state=test_val_rng)
        z_val = norm_2_sp.rvs(size=comp_size, random_state=test_val_rng)

        component_logps = np.stack(
            [norm_1_sp.logpdf(x_val), gamma_sp.logpdf(y_val), norm_2_sp.logpdf(z_val)],
            axis=join_axis,
        )[indices_val]
        index_logps = scipy_logprob(i_val, p_val)
        exp_obs_logps = component_logps + index_logps[(Ellipsis,) + (None,) * join_axis]

        m_val = np.stack([x_val, y_val, z_val], axis=join_axis)[indices_val]

        I_logp_vals = I_logp_fn(p_val, i_val)
        M_logp_vals = M_logp_fn(m_val, i_val)

        logp_vals = M_logp_vals + I_logp_vals[(Ellipsis,) + (None,) * join_axis]

        np.testing.assert_almost_equal(logp_vals, exp_obs_logps, decimal=decimals)


@pytest.mark.parametrize(
    "A_parts, indices",
    [
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.array([[0, 1], [2, 2]]), slice(2, 3)),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), np.array([[0, 1], [2, 2]])),
        ),
        (
            (
                np.random.normal(size=(5, 4, 3)),
                np.random.normal(size=(5, 4, 3)),
                np.random.normal(size=(5, 4, 3)),
            ),
            (
                np.array([[0], [2], [1]]),
                slice(None),
                np.array([2, 1]),
                slice(2, 3),
            ),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), np.array([0, 1, 2])),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.array([[0, 1], [2, 2]]), np.array([[0, 1], [2, 2]])),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (
                np.array([[0, 1], [2, 2]]),
                np.array([[0, 1], [2, 2]]),
                np.array([[0, 1], [2, 2]]),
            ),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.array([[0, 1], [2, 2]]), np.array([[0, 1], [2, 2]]), 1),
        ),
        (
            (
                np.random.normal(size=(5, 4, 3)),
                np.random.normal(size=(5, 4, 3)),
            ),
            (slice(0, 2),),
        ),
        (
            (
                np.random.normal(size=(5, 4, 3)),
                np.random.normal(size=(5, 4, 3)),
            ),
            (slice(0, 2), np.random.randint(3, size=(2, 3))),
        ),
    ],
)
def test_expand_indices_basic(A_parts, indices):
    A = at.stack(A_parts)
    at_indices = [as_index_constant(idx) for idx in indices]
    full_indices = expand_indices(at_indices, shape_tuple(A))
    assert len(full_indices) == A.ndim
    exp_res = A[indices].eval()
    res = A[full_indices].eval()
    assert np.array_equal(res, exp_res)


@pytest.mark.parametrize(
    "A_parts, indices",
    [
        (
            (
                np.random.normal(size=(6, 5, 4, 3)),
                np.random.normal(size=(6, 5, 4, 3)),
                np.random.normal(size=(6, 5, 4, 3)),
            ),
            (
                slice(None),
                np.array([[0], [2], [1]]),
                slice(None),
                np.array([2, 1]),
                slice(2, 3),
            ),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.array([[0, 1], [2, 2]]), slice(None), np.array([[0, 1], [2, 2]])),
        ),
    ],
)
def test_expand_indices_moved_subspaces(A_parts, indices):
    A = at.stack(A_parts)
    at_indices = [as_index_constant(idx) for idx in indices]
    full_indices = expand_indices(at_indices, shape_tuple(A))
    assert len(full_indices) == A.ndim
    exp_res = A[indices].eval()
    res = A[full_indices].eval()
    assert np.array_equal(res, exp_res)


@pytest.mark.parametrize(
    "A_parts, indices",
    [
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), np.array([0, 1, 2]), 1),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), 1, np.array([0, 1, 2])),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (1, slice(2, 3), np.array([0, 1, 2])),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.random.randint(2, size=(4, 3)), 1, 0),
        ),
    ],
)
def test_expand_indices_single_indices(A_parts, indices):
    A = at.stack(A_parts)
    at_indices = [as_index_constant(idx) for idx in indices]
    full_indices = expand_indices(at_indices, shape_tuple(A))
    assert len(full_indices) == A.ndim
    exp_res = A[indices].eval()
    res = A[full_indices].eval()
    assert np.array_equal(res, exp_res)


@pytest.mark.parametrize(
    "A_parts, indices",
    [
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (None,),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (None, None, None),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (None, 1, None, 0, None),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), None, 1, None, 0, None),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), None, 1, 0, None),
        ),
    ],
)
def test_expand_indices_newaxis(A_parts, indices):
    A = at.stack(A_parts)
    at_indices = [as_index_constant(idx) for idx in indices]
    full_indices = expand_indices(at_indices, shape_tuple(A))
    assert len(full_indices) == A.ndim
    exp_res = A[indices].eval()
    res = A[full_indices].eval()
    assert np.array_equal(res, exp_res)


def test_mixture_with_DiracDelta():
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(0, 1, name="X")
    Y_rv = dirac_delta(0.0)
    Y_rv.name = "Y"

    I_rv = srng.categorical([0.5, 0.5], size=4)

    M_rv = at.stack([X_rv, Y_rv])[I_rv]
    M_rv.name = "M"

    logp_res, _ = conditional_logprob(M_rv, I_rv)

    assert M_rv in logp_res


def test_switch_mixture():
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(-10.0, 0.1, name="X")
    Y_rv = srng.normal(10.0, 0.1, name="Y")

    I_rv = srng.bernoulli(0.5, name="I")
    i_vv = I_rv.clone()
    i_vv.name = "i"

    Z1_rv = at.switch(I_rv, X_rv, Y_rv)
    z_vv = Z1_rv.clone()
    z_vv.name = "z1"

    fgraph, *_ = construct_ir_fgraph({Z1_rv: z_vv, I_rv: i_vv})

    out_rv = fgraph.outputs[0].owner.inputs[0]
    assert isinstance(out_rv.owner.op, MixtureRV)
    assert not hasattr(
        fgraph.outputs[0].tag, "test_value"
    )  # aesara.config.compute_test_value == "off"
    assert fgraph.outputs[0].name is None

    Z1_rv.name = "Z1"

    fgraph, *_ = construct_ir_fgraph({Z1_rv: z_vv, I_rv: i_vv})

    out_rv = fgraph.outputs[0].owner.inputs[0]
    assert out_rv.name == "Z1-mixture"

    # building the identical graph but with a stack to check that mixture computations are identical

    Z2_rv = at.stack((X_rv, Y_rv))[I_rv]

    fgraph2, *_ = construct_ir_fgraph({Z2_rv: z_vv, I_rv: i_vv})

    assert equal_computations(fgraph.outputs, fgraph2.outputs)

    z1_logp, _ = joint_logprob(realized={Z1_rv: z_vv, I_rv: i_vv})
    z2_logp, _ = joint_logprob(realized={Z2_rv: z_vv, I_rv: i_vv})

    # below should follow immediately from the equal_computations assertion above
    assert equal_computations([z1_logp], [z2_logp])

    np.testing.assert_almost_equal(0.69049938, z1_logp.eval({z_vv: -10, i_vv: 0}))
    np.testing.assert_almost_equal(0.69049938, z2_logp.eval({z_vv: -10, i_vv: 0}))


def test_dirac_delta_mixture_dtype():
    srng = at.random.RandomStream()

    p = at.scalar("p")
    I_rv = srng.bernoulli(p)

    X_rv = dirac_delta(at.as_tensor(0))
    assert X_rv.type.dtype.startswith("int")

    Y_rv = srng.poisson(1.0)

    Z_rv = at.stack([X_rv, Y_rv])[I_rv]
    logprob, (
        i_vv,
        z_vv,
    ) = joint_logprob(I_rv, Z_rv)

    assert logprob.type.dtype.startswith("float")
