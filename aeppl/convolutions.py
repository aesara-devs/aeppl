import aesara.tensor as at
from aesara.graph.rewriting.basic import node_rewriter
from aesara.tensor.random.basic import NormalRV, normal

from aeppl.rewriting import measurable_ir_rewrites_db


@node_rewriter((at.sub, at.add))
def add_independent_normals(fgraph, node):
    """Replace a sum of two un-valued independent normal RVs with a single normal RV."""

    if node.op == at.add:
        sub = False
    else:
        sub = True

    X_rv, Y_rv = node.inputs

    if not (X_rv.owner and Y_rv.owner) or not (
        # This also checks that the RVs are un-valued (i.e. they're not
        # `ValuedVariable`s)
        isinstance(X_rv.owner.op, NormalRV)
        and isinstance(Y_rv.owner.op, NormalRV)
    ):
        return None

    old_rv = node.outputs[0]

    mu_x, sigma_x, mu_y, sigma_y, _ = at.broadcast_arrays(
        *(X_rv.owner.inputs[-2:] + Y_rv.owner.inputs[-2:] + [old_rv])
    )

    new_rng = X_rv.owner.inputs[0].clone()

    new_node = normal.make_node(
        new_rng,
        old_rv.shape,
        old_rv.dtype,
        mu_x + mu_y if not sub else mu_x - mu_y,
        at.sqrt(sigma_x**2 + sigma_y**2),
    )

    fgraph.add_input(new_rng)

    # new_rng must be updated with values of the RNGs output by `new_node
    new_rng.default_update = new_node.outputs[0]
    new_normal_rv = new_node.default_output()

    if old_rv.name:
        new_normal_rv.name = old_rv.name

    return [new_normal_rv]


measurable_ir_rewrites_db.register(
    "add_independent_normals",
    add_independent_normals,
    "basic",
)
