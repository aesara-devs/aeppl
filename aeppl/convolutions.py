import aesara.tensor as at
from aesara.graph.rewriting.basic import node_rewriter
from aesara.tensor.random.basic import NormalRV, normal

from aeppl.rewriting import measurable_ir_rewrites_db


@node_rewriter((at.sub, at.add))
def add_independent_normals(fgraph, node):
    """Replace a sum of un-valued independent normal RVs with a single normal RV."""

    if len(node.inputs) < 2:
        return None

    if node.op == at.add:
        sub = False
    else:
        sub = True

    # This also implicitly checks that the RVs are un-valued (i.e. they're not
    # `ValuedVariable`s)
    if not all(inp.owner and isinstance(inp.owner.op, NormalRV) for inp in node.inputs):
        return None

    new_size = at.broadcast_shape(
        *(tuple(inp.owner.inputs[1]) for inp in node.inputs), arrays_are_shapes=True
    )

    means = [inp.owner.inputs[-2] for inp in node.inputs]
    covs = [inp.owner.inputs[-1] ** 2 for inp in node.inputs]

    new_rng = node.inputs[0].owner.inputs[0].clone()

    new_node = normal.make_node(
        new_rng,
        new_size,
        node.outputs[0].dtype,
        at.add(*means) if not sub else at.sub(*means),
        at.sqrt(at.add(*covs)),
    )

    fgraph.add_input(new_rng)

    # new_rng must be updated with values of the RNGs output by `new_node
    new_rng.default_update = new_node.outputs[0]
    new_normal_rv = new_node.default_output()

    return [new_normal_rv]


measurable_ir_rewrites_db.register(
    "add_independent_normals",
    add_independent_normals,
    "basic",
)
