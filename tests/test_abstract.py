import aesara
import aesara.tensor as at
import numpy as np
from aesara.gradient import NullTypeGradError, grad
from pytest import raises

from aeppl.abstract import valued_variable


def test_observed():
    rv_var = at.random.normal(0, 1, size=3)
    obs_var = valued_variable(
        rv_var, np.array([0.2, 0.1, -2.4], dtype=aesara.config.floatX)
    )

    assert obs_var.owner.inputs[0] is rv_var

    with raises(TypeError):
        valued_variable(rv_var, np.array([1, 2], dtype=int))

    with raises(TypeError):
        valued_variable(rv_var, np.array([[1.0, 2.0]], dtype=rv_var.dtype))

    # obs_rv = valued_variable(None, np.array([0.2, 0.1, -2.4], dtype=aesara.config.floatX))
    #
    # assert isinstance(obs_rv.owner.inputs[0].type, NoneTypeT)

    rv_val = at.vector()
    rv_val.tag.test_value = np.array([0.2, 0.1, -2.4], dtype=aesara.config.floatX)

    obs_var = valued_variable(rv_var, rv_val)

    with raises(NullTypeGradError):
        grad(obs_var.sum(), [rv_val])
