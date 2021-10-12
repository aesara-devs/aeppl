import aesara
import aesara.tensor as at
import pytest
from aesara.graph.opt import EquilibriumOptimizer, in2out
from aesara.graph.opt_utils import optimize_graph
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.extra_ops import BroadcastTo
from aesara.tensor.subtensor import Subtensor

from aeppl.abstract import valued_variable
from aeppl.dists import DiracDelta, dirac_delta
from aeppl.opt import local_lift_DiracDelta, valued_var_bcast_lift

bcast_lift_opt = EquilibriumOptimizer(
    [valued_var_bcast_lift], ignore_newtrees=False, max_use_ratio=1000
)


@pytest.mark.parametrize(
    "rv_params, rv_size, bcast_shape, should_rewrite",
    [
        # The `BroadcastTo` shouldn't be lifted, because it would imply that there
        # are 10 independent samples, when there's really only one
        # pytest.param(
        #     (0, 1),
        #     None,
        #     (10,),
        #     False,
        #     marks=pytest.mark.xfail(reason="Not implemented"),
        # ),
        # These should work, under the assumption that `size == 10`, of course.
        ((0, 1), at.iscalar("size"), (10,), True),
        ((0, 1), at.iscalar("size"), (1, 10, 1), True),
        ((at.zeros((at.iscalar("size"),)), 1), None, (10,), True),
    ],
)
def test_valued_var_bcast_lift(rv_params, rv_size, bcast_shape, should_rewrite):
    rv_var = at.random.normal(*rv_params, size=rv_size)
    vv_var = rv_var.clone()
    valed_var = valued_variable(rv_var, vv_var)
    graph = at.broadcast_to(valed_var, bcast_shape)

    assert isinstance(graph.owner.op, BroadcastTo)

    new_graph = optimize_graph(graph, custom_opt=bcast_lift_opt)

    if should_rewrite:
        assert not isinstance(new_graph.owner.op, BroadcastTo)
    else:
        assert isinstance(new_graph.owner.op, BroadcastTo)


def test_valued_var_bcast_lift_empty():
    r"""Make sure `valued_var_bcast_lift` can handle useless scalar `BroadcastTo`\s."""
    X_rv = at.random.normal()
    x_vv = X_rv.clone()
    Y_at = valued_variable(X_rv, x_vv)
    Z_at = at.broadcast_to(Y_at, ())

    # Make sure we're testing what we intend to test
    assert isinstance(Z_at.owner.op, BroadcastTo)

    res = optimize_graph(Z_at, custom_opt=in2out(valued_var_bcast_lift), clone=False)
    assert res is Y_at


def test_local_lift_DiracDelta():
    c_at = at.vector()
    dd_at = dirac_delta(c_at)

    Z_at = at.cast(dd_at, "int64")

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Elemwise)

    Z_at = dd_at.dimshuffle("x", 0)

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, DimShuffle)

    Z_at = dd_at[0]

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Subtensor)

    # Don't lift multi-output `Op`s
    c_at = at.matrix()
    dd_at = dirac_delta(c_at)
    Z_at = at.nlinalg.svd(dd_at)[0]

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert res is Z_at


def test_local_remove_DiracDelta():
    c_at = at.vector()
    dd_at = dirac_delta(c_at)

    fn = aesara.function([c_at], dd_at)
    assert not any(
        isinstance(node.op, DiracDelta) for node in fn.maker.fgraph.toposort()
    )
