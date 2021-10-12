import abc
from copy import copy
from functools import singledispatch
from typing import Callable, List

from aesara.gradient import grad_undefined
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import Op
from aesara.tensor.random.op import RandomVariable


class MeasurableVariable(abc.ABC):
    """A variable that can be assigned a measure/log-probability"""


MeasurableVariable.register(RandomVariable)


def get_measurable_outputs(op: Op, node: Apply) -> List[Variable]:
    """Return only the outputs that are measurable."""
    if isinstance(op, MeasurableVariable):
        return _get_measurable_outputs(op, node)
    else:
        return []


@singledispatch
def _get_measurable_outputs(op, node):
    return node.outputs


@_get_measurable_outputs.register(RandomVariable)
def _get_measurable_outputs_RandomVariable(op, node):
    return node.outputs[1:]


def noop_measurable_outputs_fn(*args, **kwargs):
    return None


def assign_custom_measurable_outputs(
    node: Apply,
    measurable_outputs_fn: Callable = noop_measurable_outputs_fn,
    type_prefix: str = "Unmeasurable",
) -> Apply:
    """Assign a custom ``_get_measurable_outputs`` dispatch function to a measurable variable instance.

    The node is cloned and a custom `Op` that's a copy of the original node's
    `Op` is created.  That custom `Op` replaces the old `Op` in the cloned
    node, and then a custom dispatch implementation is created for the clone
    `Op` in `_get_measurable_outputs`.

    If `measurable_outputs_fn` isn't specified, a no-op is used; the result is
    a clone of `node` that will effectively be ignored by
    `factorized_joint_logprob`.

    Parameters
    ==========
    node
        The node to recreate with a new cloned `Op`.
    measurable_outputs_fn
        The function that will be assigned to the new cloned `Op` in the
        `_get_measurable_outputs` dispatcher.
        The default is a no-op function (i.e. no measurable outputs)
    type_prefix
        The prefix used for the new type's name.
        The default is ``"Unmeasurable"``, which matches the default
        ``"measurable_outputs_fn"``.
    """

    new_node = node.clone()
    op_type = type(new_node.op)
    new_op_type = type(
        f"{type_prefix}{op_type.__name__}", (op_type,), op_type.__dict__.copy()
    )

    new_node.op = copy(new_node.op)
    new_node.op.__class__ = new_op_type

    # TODO: The above could be a stand-alone utility function for all sorts of
    # instance-based dispatching

    _get_measurable_outputs.register(new_op_type)(measurable_outputs_fn)

    return new_node


class ValuedVariable(Op):
    r"""Represents the association of a measurable variable and its value.

    A `ValuedVariable` node represents the pair :math:`(Y, y)`, where
    :math:`Y` is a random variable and :math:`y \sim Y`.

    Log-probability (densities) are functions over these pairs, which makes
    these nodes in a graph an intermediate form that serves to construct a
    log-probability from a model graph.

    This intermediate form can be used as the target for rewrites that
    otherwise wouldn't make sense to apply to--say--a random variable node
    directly.  An example is `BroadcastTo` lifting through `RandomVariable`\s.
    """

    default_output = 0
    view_map = {0: [0]}

    def make_node(self, rv, value):
        # TODO: We assume the `Type` of `rv`, but that might not be the best
        # approach
        output = rv.type()
        return Apply(self, [rv, value], [output])

    def perform(self, node, inputs, out):
        out[0][0] = inputs[1]

    def grad(self, inputs, outputs):
        return [
            grad_undefined(self, k, inp, "No gradient defined for `ValuedVariable`")
            for k, inp in enumerate(inputs)
        ]

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]


valued_variable = ValuedVariable()
