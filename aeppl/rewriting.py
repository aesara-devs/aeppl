from typing import Dict, Optional, Tuple

import aesara.tensor as at
from aesara.compile.mode import optdb
from aesara.graph.basic import Apply, Variable
from aesara.graph.features import Feature
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import GraphRewriter, in2out, node_rewriter
from aesara.graph.rewriting.db import EquilibriumDB, RewriteDatabaseQuery, SequenceDB
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.extra_ops import BroadcastTo
from aesara.tensor.random.rewriting import local_subtensor_rv_lift
from aesara.tensor.rewriting.basic import register_canonicalize, register_useless
from aesara.tensor.rewriting.shape import ShapeFeature
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)

from aeppl.abstract import MeasurableVariable, ValuedVariable, valued_variable
from aeppl.dists import DiracDelta
from aeppl.utils import indices_from_subtensor

inc_subtensor_ops = (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)
subtensor_ops = (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)


class NoCallbackEquilibriumDB(EquilibriumDB):
    r"""This `EquilibriumDB` doesn't hide its exceptions.

    By setting `failure_callback` to ``None`` in the `EquilibriumGraphRewriter`\s
    that `EquilibriumDB` generates, we're able to directly emit the desired
    exceptions from within the `NodeRewriter`\s themselves.
    """

    def query(self, *tags, **kwtags):
        res = super().query(*tags, **kwtags)
        res.failure_callback = None
        return res


class MeasurableConversionTracker(Feature):
    r"""Keeps track of variables that were converted to `MeasurableVariable`\s.

    A `measurable_conversions` map is tracked that holds
    information about un-valued and un-measurable variables that were replaced
    with measurable variables.  This information can be used to revert these
    rewrites.

    """

    def on_attach(self, fgraph):
        if hasattr(fgraph, "preserve_rv_mappings"):
            raise ValueError(
                f"{fgraph} already has the `PreserveRVMappings` feature attached."
            )

        fgraph.measurable_conversions = {}

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        if (
            r.owner
            and new_r.owner
            and not isinstance(r.owner.op, (ValuedVariable, MeasurableVariable))
            and isinstance(new_r.owner.op, MeasurableVariable)
            and not any(
                isinstance(node.op, ValuedVariable)
                for node, idx in fgraph.clients[new_r]
                if isinstance(node, Apply)
            )
        ):
            fgraph.measurable_conversions[r] = new_r


@register_canonicalize
@node_rewriter((Elemwise, BroadcastTo, DimShuffle) + subtensor_ops)
def local_lift_DiracDelta(fgraph, node):
    r"""Lift basic `Op`\s through `DiracDelta`\s."""

    if len(node.outputs) > 1:
        return

    # Only handle scalar `Elemwise` `Op`s
    if isinstance(node.op, Elemwise) and len(node.inputs) != 1:
        return

    dd_inp = node.inputs[0]

    if dd_inp.owner is None or not isinstance(dd_inp.owner.op, DiracDelta):
        return

    dd_val = dd_inp.owner.inputs[0]

    new_value_node = node.op.make_node(dd_val, *node.inputs[1:])
    new_node = dd_inp.owner.op.make_node(new_value_node.outputs[0])
    return new_node.outputs


@register_useless
@node_rewriter((DiracDelta,))
def local_remove_DiracDelta(fgraph, node):
    r"""Remove `DiracDelta`\s."""
    dd_val = node.inputs[0]
    return [dd_val]


@node_rewriter([ValuedVariable])
def incsubtensor_rv_replace(fgraph, node):
    r"""Replace `*IncSubtensor*` `Op`\s and their value variables for log-probability calculations.

    This is used to derive the log-probability graph for ``Y[idx] = data``, where
    ``Y`` is a `RandomVariable`, ``idx`` indices, and ``data`` some arbitrary data.

    To compute the log-probability of a statement like ``Y[idx] = data``, we must
    first realize that our objective is equivalent to computing ``logprob(Y, z)``,
    where ``z = at.set_subtensor(y[idx], data)`` and ``y`` is the value variable
    for ``Y``.

    In other words, the log-probability for an `*IncSubtensor*` is the log-probability
    of the underlying `RandomVariable` evaluated at ``data`` for the indices
    given by ``idx`` and at the value variable for ``~idx``.

    This provides a means of specifying "missing data", for instance.
    """
    incsubtensor_var, value_var = node.inputs
    subtensor_node = incsubtensor_var.owner

    if subtensor_node is None or not isinstance(subtensor_node.op, inc_subtensor_ops):
        return None  # pragma: no cover

    base_rv_var = subtensor_node.inputs[0]

    if not (
        base_rv_var.owner
        and isinstance(base_rv_var.owner.op, MeasurableVariable)
        and not isinstance(base_rv_var, ValuedVariable)
    ):
        return None  # pragma: no cover

    data = subtensor_node.inputs[1]
    idx = indices_from_subtensor(
        getattr(subtensor_node.op, "idx_list", None), subtensor_node.inputs[2:]
    )

    # Create a new value variable with the indices `idx` set to `data`
    new_value_var = at.set_subtensor(value_var[idx], data)

    new_base_rv_var = valued_variable(base_rv_var, new_value_var)

    return [new_base_rv_var]


logprob_rewrites_db = SequenceDB()
logprob_rewrites_db.name = "logprob_rewrites_db"
logprob_rewrites_db.register("pre-canonicalize", optdb.query("+canonicalize"), "basic")

# These rewrites convert un-measurable variables into their measurable forms,
# but they need to be reapplied, because some of the measurable forms require
# their inputs to be measurable.
measurable_ir_rewrites_db = NoCallbackEquilibriumDB()
measurable_ir_rewrites_db.name = "measurable_ir_rewrites_db"

logprob_rewrites_db.register(
    "measurable_ir_rewrites", measurable_ir_rewrites_db, "basic"
)

# These rewrites push random/measurable variables "down", making them closer to
# (or eventually) the graph outputs.  Often this is done by lifting other `Op`s
# "up" through the random/measurable variables and into their inputs.
measurable_ir_rewrites_db.register("subtensor_lift", local_subtensor_rv_lift, "basic")
measurable_ir_rewrites_db.register(
    "incsubtensor_lift", incsubtensor_rv_replace, "basic"
)

logprob_rewrites_db.register("post-canonicalize", optdb.query("+canonicalize"), "basic")


def construct_ir_fgraph(
    rv_values: Dict[Variable, Variable],
    ir_rewriter: Optional[GraphRewriter] = None,
) -> Tuple[FunctionGraph, Dict[Variable, Variable], Dict[Variable, Variable]]:
    r"""Construct a `FunctionGraph` in measurable IR form for the keys in `rv_values`.

    A custom IR rewriter can be specified. By default,
    ``logprob_rewrites_db.query(RewriteDatabaseQuery(include=["basic"]))`` is used.

    Our measurable IR takes the form of an Aesara graph that is more-or-less
    equivalent to a given Aesara graph (i.e. the keys of `rv_values`) but
    contains `Op`s that are subclasses of the `MeasurableVariable` type in
    place of ones that do not inherit from `MeasurableVariable` in the original
    graph but are nevertheless measurable.

    `MeasurableVariable`\s are mapped to log-probabilities, so this IR is how
    non-trivial log-probabilities are constructed, especially when the
    "measurability" of a term depends on the measurability of its inputs
    (e.g. a mixture).

    In some cases, entire sub-graphs in the original graph are replaced with a
    single measurable node.  In other cases, the relevant nodes are already
    measurable and there is no difference between the resulting measurable IR
    graph and the original.  In general, some changes will be present,
    because--at the very least--canonicalization is always performed and the
    measurable IR includes manipulations that are not applicable to outside of
    the context of measurability/log-probabilities.

    For instance, some `Op`s will be lifted through `MeasurableVariable`\s in
    this IR, and the resulting graphs will not be computationally sound,
    because they wouldn't produce independent samples when the original graph
    would.  See https://github.com/aesara-devs/aeppl/pull/78.

    Returns
    -------
    A `FunctionGraph` of the measurable IR, a copy of `rv_values` containing
    the new, cloned versions of the original variables in `rv_values`, and
    a ``dict`` mapping all the original variables to their cloned values in
    the `FunctionGraph`.
    """

    # We're going to create a `FunctionGraph` that effectively represents the
    # joint log-probability of all the random variables assigned values in
    # `rv_values`.
    # The `FunctionGraph`'s outputs will be `ValuedVariable`s.  This serves to
    # associate a random variable with its value and facilitate
    # rewrites/transformations of both simultaneously.
    # The `FunctionGraph` will be transformed by rewrites so that
    # log-probabilities can be assigned to/derived for its outputs.

    # We clone the random variables that we'll use as `FunctionGraph` outputs
    # so that they're distinct nodes in the graph.  This allows us to replace
    # all instances of the original random variables with their value
    # variables, while leaving the output clones untouched.
    rv_value_clones = {}
    measured_outputs = {}
    memo = {}
    for rv, val in rv_values.items():
        rv_node_clone = rv.owner.clone()
        rv_clone = rv_node_clone.outputs[rv.owner.outputs.index(rv)]
        rv_value_clones[rv_clone] = val
        measured_outputs[rv] = valued_variable(rv_clone, val)
        # Prevent value variables from being cloned
        memo[val] = val

    # We add `ShapeFeature` because it will get rid of references to the old
    # `RandomVariable`s that have been lifted; otherwise, it will be difficult
    # to give good warnings when an unaccounted for `RandomVariable` is
    # encountered
    fgraph = FunctionGraph(
        outputs=tuple(measured_outputs.values()),
        features=[ShapeFeature(), MeasurableConversionTracker()],
        clone=True,
        memo=memo,
        copy_orphans=False,
        copy_inputs=False,
    )

    # Update `rv_values` so that it uses the new cloned variables
    rv_value_clones = {memo[k]: v for k, v in rv_value_clones.items()}

    # Replace valued non-output variables with their values
    fgraph.replace_all(
        [(memo[rv], val) for rv, val in measured_outputs.items() if rv in memo],
        reason="valued-non-outputs-replace",
        import_missing=True,
    )

    if ir_rewriter is None:
        ir_rewriter = logprob_rewrites_db.query(RewriteDatabaseQuery(include=["basic"]))

    ir_rewriter.rewrite(fgraph)

    # Undo un-valued measurable IR rewrites
    new_to_old = tuple((v, k) for k, v in fgraph.measurable_conversions.items())
    fgraph.replace_all(new_to_old, reason="undo-unvalued-measurables")

    return fgraph, rv_value_clones, memo


@register_useless
@node_rewriter([ValuedVariable])
def remove_ValuedVariable(fgraph, node):
    return [node.inputs[1]]


ir_cleanup_db = SequenceDB()
ir_cleanup_db.name = "ir_cleanup_db"
ir_cleanup_db.register(
    "remove-intermediate-ir",
    in2out(local_remove_DiracDelta, remove_ValuedVariable),
    "basic",
)
