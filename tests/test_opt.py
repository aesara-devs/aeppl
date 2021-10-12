import aesara.tensor as at
import pytest
from aesara.graph.opt import EquilibriumOptimizer
from aesara.graph.opt_utils import optimize_graph
from aesara.tensor.extra_ops import BroadcastTo

from aeppl.opt import naive_bcast_rv_lift

bcast_lift_opt = EquilibriumOptimizer(
    [naive_bcast_rv_lift], ignore_newtrees=False, max_use_ratio=1000
)


@pytest.mark.parametrize(
    "rv_params, rv_size, bcast_shape, should_rewrite",
    [
        # The `BroadcastTo` shouldn't be lifted, because it would imply that there
        # are 10 independent samples, when there's really only one
        pytest.param(
            (0, 1),
            None,
            (10,),
            False,
            marks=pytest.mark.xfail(reason="Not implemented"),
        ),
        # These should work, under the assumption that `size == 10`, of course.
        ((0, 1), at.iscalar("size"), (10,), True),
        ((0, 1), at.iscalar("size"), (1, 10, 1), True),
        ((at.zeros((at.iscalar("size"),)), 1), None, (10,), True),
    ],
)
def test_naive_bcast_rv_lift(rv_params, rv_size, bcast_shape, should_rewrite):
    graph = at.broadcast_to(at.random.normal(*rv_params, size=rv_size), bcast_shape)

    assert isinstance(graph.owner.op, BroadcastTo)

    new_graph = optimize_graph(graph, custom_opt=bcast_lift_opt)

    if should_rewrite:
        assert not isinstance(new_graph.owner.op, BroadcastTo)
    else:
        assert isinstance(new_graph.owner.op, BroadcastTo)
