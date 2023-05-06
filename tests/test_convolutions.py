import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.random.basic import NormalRV

from aeppl.rewriting import construct_ir_fgraph
from aeppl.transforms import MeasurableElemwiseTransform


@pytest.mark.parametrize(
    "mu_x, mu_y, sigma_x, sigma_y, x_shape, y_shape",
    [
        (
            np.array([1, 10, 100]),
            np.array(2),
            np.array(0.03),
            np.tile(0.04, 3),
            (),
            (),
        ),
        (
            np.array([1, 10, 100]),
            np.array(2),
            np.array(0.03),
            np.full((5, 1), 0.04),
            (),
            (5, 3),
        ),
        (
            np.array([[1, 10, 100]]),
            np.array([[0.2], [2], [20], [200], [2000]]),
            np.array(0.03),
            np.array(0.04),
            (),
            (),
        ),
        (
            np.broadcast_to(np.array([1, 10, 100]), (5, 3)),
            np.array([2, 20, 200]),
            np.array(0.03),
            np.array(0.04),
            (2, 5, 3),
            (),
        ),
        (
            np.array([[1, 10, 100]]),
            np.array([[0.2], [2], [20], [200], [2000]]),
            np.array([[0.5], [5], [50], [500], [5000]]),
            np.array([[0.4, 4, 40]]),
            (2, 5, 3),
            (),
        ),
        (
            np.array(1),
            np.array(2),
            np.array(3),
            np.array(4),
            (5, 1),
            (1,),
        ),
    ],
)
@pytest.mark.parametrize("sub", [False, True])
def test_add_independent_normals(mu_x, mu_y, sigma_x, sigma_y, x_shape, y_shape, sub):
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(mu_x, sigma_x, size=x_shape)
    X_rv.name = "X"

    Y_rv = srng.normal(mu_y, sigma_y, size=y_shape)
    Y_rv.name = "Y"

    Z_rv = X_rv + Y_rv if not sub else X_rv - Y_rv
    Z_rv.name = "Z"
    z_vv = Z_rv.clone()

    fgraph, *_ = construct_ir_fgraph({Z_rv: z_vv})

    (valued_var_out_node) = fgraph.outputs[0].owner
    # The convolution should be applied, and not the transform
    assert isinstance(valued_var_out_node.inputs[0].owner.op, NormalRV)

    new_rv = fgraph.outputs[0].owner.inputs[0]

    new_rv_mu = mu_x + mu_y if not sub else mu_x - mu_y
    new_rv_sigma = np.sqrt(sigma_x**2 + sigma_y**2)

    new_rv_shape = np.broadcast_shapes(
        new_rv_mu.shape, new_rv_sigma.shape, x_shape, y_shape
    )

    new_rv_mu = np.broadcast_to(new_rv_mu, new_rv_shape)
    new_rv_sigma = np.broadcast_to(new_rv_sigma, new_rv_shape)

    assert isinstance(new_rv.owner.op, NormalRV)
    assert np.allclose(new_rv.owner.inputs[3].eval(), new_rv_mu)
    assert np.allclose(new_rv.owner.inputs[4].eval(), new_rv_sigma)


def test_normal_add_input_valued():
    """Test the case when one of the normal inputs to the add `Op` is a `ValuedVariable`."""
    srng = at.random.RandomStream(0)

    X_rv = srng.normal(1.0, name="X")
    x_vv = X_rv.clone()
    Y_rv = srng.normal(1.0, name="Y")
    Z_rv = X_rv + Y_rv
    Z_rv.name = "Z"
    z_vv = Z_rv.clone()

    fgraph, *_ = construct_ir_fgraph({Z_rv: z_vv, X_rv: x_vv})

    valued_var_out_node = fgraph.outputs[0].owner
    # We should not expect the convolution to be applied; instead, the
    # transform should be (for now)
    assert isinstance(
        valued_var_out_node.inputs[0].owner.op, MeasurableElemwiseTransform
    )


def test_normal_add_three_inputs():
    """Test the case when there are more than two inputs in the sum."""
    srng = at.random.RandomStream(0)

    mu_x = at.vector("mu_x")
    sigma_x = at.vector("sigma_x")
    X_rv = srng.normal(mu_x, sigma_x, name="X")
    mu_y = at.vector("mu_y")
    sigma_y = at.vector("sigma_y")
    Y_rv = srng.normal(mu_y, sigma_y, size=(2, 1), name="Y")
    mu_w = at.vector("mu_w")
    sigma_w = at.vector("sigma_w")
    W_rv = srng.normal(mu_w, sigma_w, name="W")

    Z_rv = X_rv + Y_rv + W_rv
    Z_rv.name = "Z"
    z_vv = Z_rv.clone()

    fgraph, *_ = construct_ir_fgraph({Z_rv: z_vv})

    valued_var_out_node = fgraph.outputs[0].owner
    # The convolution should be applied, and not the transform
    assert isinstance(valued_var_out_node.inputs[0].owner.op, NormalRV)
