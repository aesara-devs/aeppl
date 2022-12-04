import abc
from copy import copy
from functools import partial, singledispatch
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import aesara.tensor as at
from aesara.gradient import DisconnectedType, jacobian
from aesara.graph.basic import Apply, Variable, walk
from aesara.graph.features import AlreadyThere, Feature
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.rewriting.basic import GraphRewriter, in2out, node_rewriter
from aesara.tensor.math import add, exp, log, mul, reciprocal, sub, true_div
from aesara.tensor.rewriting.basic import register_useless
from aesara.tensor.var import TensorVariable
from typing_extensions import Protocol

from aeppl.abstract import (
    MeasurableElemwise,
    MeasurableVariable,
    ValuedVariable,
    _get_measurable_outputs,
    assign_custom_measurable_outputs,
    valued_variable,
)
from aeppl.logprob import _logprob, logprob
from aeppl.rewriting import measurable_ir_rewrites_db

if TYPE_CHECKING:
    from aesara.graph.rewriting.basic import NodeRewriter

    class TransformFnType(Protocol):
        def __call__(
            self, measurable_input: MeasurableVariable, *other_inputs: Variable
        ) -> Tuple["RVTransform", Tuple[TensorVariable, ...]]:
            pass


def register_measurable_ir(
    node_rewriter: "NodeRewriter",
    *tags: str,
    **kwargs,
):
    name = kwargs.pop("name", None) or node_rewriter.__name__
    measurable_ir_rewrites_db.register(
        name, node_rewriter, "basic", "transform", *tags, **kwargs
    )
    return node_rewriter


@singledispatch
def _default_transformed_rv(
    op: Op,
    node: Apply,
) -> Optional[Apply]:
    """Create a node for a transformed log-probability of a `MeasurableVariable`.

    This function dispatches on the type of `op`.  If you want to implement
    new transforms for a `MeasurableVariable`, register a function on this
    dispatcher.

    """
    return None


class TransformedVariable(Op):
    r"""A no-op that identifies a transformed value variable and its un-transformed input.

    This transformed/untransformed-value pairing is primarily used as a means to
    obtain the untransformed value for use with the `RVTransform.log_jac_det` when
    log-probabilities are generated.

    It is expected that all occurrences of these `Op`\s will be removed during
    compilation by `remove_TransformedVariables`.

    """

    view_map = {0: [0]}

    def make_node(self, tran_value: TensorVariable, value: TensorVariable):
        return Apply(self, [tran_value, value], [tran_value.type()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            "These `Op`s should be removed from graphs used for computation."
        )

    def connection_pattern(self, node):
        return [[True], [False]]

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]

    def grad(self, args, g_outs):
        return g_outs[0], DisconnectedType()()


transformed_variable = TransformedVariable()


@register_useless
@node_rewriter([TransformedVariable])
def remove_TransformedVariables(fgraph, node):
    if isinstance(node.op, TransformedVariable):
        return [node.inputs[0]]


class RVTransform(abc.ABC):
    @abc.abstractmethod
    def forward(self, value: TensorVariable, *inputs: Variable) -> TensorVariable:
        """Apply the transformation."""

    @abc.abstractmethod
    def backward(self, value: TensorVariable, *inputs: Variable) -> TensorVariable:
        """Invert the transformation."""

    def log_jac_det(self, value: TensorVariable, *inputs) -> TensorVariable:
        """Construct the log of the absolute value of the Jacobian determinant."""
        # jac = at.reshape(
        #     gradient(at.sum(self.backward(value, *inputs)), [value]), value.shape
        # )
        # return at.log(at.abs(jac))
        phi_inv = self.backward(value, *inputs)
        return at.log(
            at.abs(at.nlinalg.det(at.atleast_2d(jacobian(phi_inv, [value])[0])))
        )


class DefaultTransformSentinel:
    pass


DEFAULT_TRANSFORM = DefaultTransformSentinel()


@node_rewriter([ValuedVariable])
def transform_values(fgraph: FunctionGraph, node: Apply):
    r"""Apply transforms to the values of value-bound measurable variables.

    It is assumed that the input value variables correspond to forward
    transformations, usually chosen in such a way that the values are
    unconstrained on the real line.

    For example, if ``Y = halfnormal(...)``, we assume the respective value
    variable is specified on the log scale and back-transform it to obtain
    ``Y`` on the natural scale.

    The main steps of this rewrite are as follows:

    1. Obtain a `RVTransform` for a `ValuedVariable`.
    2. Replace all occurrences of the original value variable with the
    "backward"-transform of a new value variable in the "forward"-transformed
    space.
    3. Signify that the original value variable is to be replaced by the new
    one.
    4. Replace the old `ValuedVariable` with a new one containing a
    `TransformedVariable` value.

    Step 3. is currently accomplished by updating the `memo` dictionary
    associated with the `FunctionGraph`.  Our main entry-point,
    `conditional_logprob`, checks this dictionary for value variable changes.

    TODO: This approach is less than ideal, because it puts awkward demands on
    users/callers of this rewrite to check with `memo`; let's see if we can do
    something better.

    The new value variable mentioned in Step 2. may be of a different `Type`
    (e.g. extra/fewer dimensions) than the original value variable; this is why
    we must replace the corresponding original value variables before we
    construct log-probability graphs.  This is due to the fact that we
    generally cannot replace variables with new ones that have different
    `Type`\s.

    """

    values_to_transforms = getattr(fgraph, "values_to_transforms", None)

    if values_to_transforms is None:
        return None  # pragma: no cover

    rv_var, value_var = node.inputs

    rv_node = rv_var.owner

    try:
        rv_var = rv_node.default_output()
        rv_var_out_idx = rv_node.outputs.index(rv_var)
    except ValueError:
        return None

    transform = values_to_transforms.get(value_var, None)

    if transform is None:
        return None
    elif transform is DEFAULT_TRANSFORM:
        trans_node = _default_transformed_rv(rv_node.op, rv_node)
        if trans_node is None:
            return None
        transform = trans_node.op.transform
    else:
        # This automatically constructs a `_logprob` dispatch for the
        # transformed node, so look there for the log-probability
        # implementation
        new_op = _create_transformed_rv_op(rv_node.op, transform)
        # Create a new `Apply` node and outputs
        trans_node = rv_node.clone()
        trans_node.op = new_op
        trans_node.outputs[rv_var_out_idx].name = rv_node.outputs[rv_var_out_idx].name

    # We now assume that the old value variable represents the *transformed space*.

    # We normally initialize value variables as copies of the random variables,
    # thus, in the untransformed space, we need to apply the forward
    # transformation to get value variables with the correct `Type`s in the
    # transformed space.
    trans_value_var: TensorVariable = transform.forward(
        value_var, *trans_node.inputs
    ).type()

    if value_var.name:
        trans_value_var.name = f"{value_var.name}-trans"

    fgraph.add_input(trans_value_var)

    # We need to replace all instances of the old value variables
    # with "inversely/un-" transformed versions of themselves.
    untrans_value_var = transformed_variable(
        transform.backward(trans_value_var, *trans_node.inputs), trans_value_var
    )

    # This effectively lets the caller know that a value variable has been
    # replaced (i.e. they should filter all their old value variables through
    # the memo/replacements map).
    fgraph.memo[value_var] = trans_value_var

    trans_var = trans_node.outputs[rv_var_out_idx]
    new_var = valued_variable(trans_var, untrans_value_var)

    return {value_var: untrans_value_var, node.outputs[0]: new_var}


class TransformValuesMapping(Feature):
    r"""A `Feature` that maintains a map between value variables and their transforms as
    well as between value variables and their transformed counterparts.

    This is exclusively/primarily used by `TransformValuesRewrite`.

    """

    def __init__(self, values_to_transforms, memo):
        """
        Parameters
        ==========
        values_to_transforms
            Mapping between value variables and their transformations. Each
            value variable can be assigned one of `RVTransform`,
            `DEFAULT_TRANSFORM`, or ``None``. Random variables with no
            transform specified remain unchanged.
        memo
            Mapping from variables to their clones.  This is updated
            in-place whenever a value variable is transformed.

        """
        self.values_to_transforms = values_to_transforms
        self.memo = memo

    def on_attach(self, fgraph):
        if hasattr(fgraph, "values_to_transforms"):
            raise AlreadyThere()

        fgraph.values_to_transforms = self.values_to_transforms
        fgraph.memo = self.memo


class TransformValuesRewrite(GraphRewriter):
    r"""Transforms measurable variables.

    This transformation can be used to replace all occurrences of a measurable
    variable and its values within a model graph with a transformed version of
    itself (e.g. for the purposes of sampling on "unconstrained" ranges).

    Example
    =======

    .. code::

        A_rv = at.random.normal(0, 1.0, name="A")
        B_rv = at.random.normal(A_rv, 1.0, name="B")

        logp, (a_vv, b_vv) = joint_logprob(
            A_rv,
            B_rv,
            extra_rewrites=TransformValuesRewrite(
                {A_rv: ExpTransform()}
            ),
        )

        >>> print(aesara.pprint(logp))
        sum([((-0.9189385332046727 + (-0.5 * (log(A_vv-trans) ** 2))) + (-1.0 * log(A_vv-trans))),
             (-0.9189385332046727 + (-0.5 * ((B_vv - log(A_vv-trans)) ** 2)))], axis=None)

    """

    default_transform_rewrite = in2out(transform_values, ignore_newtrees=True)

    def __init__(
        self,
        rvs_to_transforms: Dict[
            TensorVariable, Union[RVTransform, DefaultTransformSentinel, None]
        ],
    ):
        """
        Parameters
        ==========
        rvs_to_transforms
            Mapping between measurable variables and their transformations. Each
            measurable variable can be assigned an `RVTransform` instance,
            `DEFAULT_TRANSFORM`, or ``None``. Measurable variables with no
            transform specified remain unchanged.

        """

        self.rvs_to_transforms = rvs_to_transforms

    def add_requirements(
        self,
        fgraph,
        rv_to_values: Dict[TensorVariable, TensorVariable],
        memo: Dict[TensorVariable, TensorVariable],
    ):
        values_to_transforms = {
            rv_to_values[rv]: transform
            for rv, transform in self.rvs_to_transforms.items()
        }
        values_transforms_feature = TransformValuesMapping(values_to_transforms, memo)
        fgraph.attach_feature(values_transforms_feature)

    def apply(self, fgraph: FunctionGraph):
        return self.default_transform_rewrite.rewrite(fgraph)


class MeasurableElemwiseTransform(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a transformed `Elemwise`."""

    # Cannot use `transform` as name because it would clash with the property added by
    # the `TransformValuesRewrite`
    transform_elemwise: RVTransform
    measurable_input_idx: int

    def __init__(
        self, *args, transform: RVTransform, measurable_input_idx: int, **kwargs
    ):
        self.transform_elemwise = transform
        self.measurable_input_idx = measurable_input_idx
        super().__init__(*args, **kwargs)


@_get_measurable_outputs.register(MeasurableElemwiseTransform)
def _get_measurable_outputs_ElemwiseTransform(op, node):
    return [node.default_output()]


@_logprob.register(MeasurableElemwiseTransform)
def measurable_elemwise_logprob(
    op: MeasurableElemwiseTransform, values, *inputs, **kwargs
):
    """Compute the log-probability graph for a `MeasurableElemwiseTransform`."""
    # TODO: Could other rewrites affect the order of inputs?
    (value,) = values
    other_inputs = list(inputs)
    measurable_input = other_inputs.pop(op.measurable_input_idx)

    # The value variable must still be back-transformed to be on the natural support of
    # the respective measurable input.
    backward_value = op.transform_elemwise.backward(value, *other_inputs)
    input_logprob = logprob(measurable_input, backward_value, **kwargs)

    jacobian = op.transform_elemwise.log_jac_det(value, *other_inputs)

    return input_logprob + jacobian


@register_measurable_ir
@node_rewriter([true_div])
def measurable_true_div(fgraph, node):
    r"""Rewrite a `true_div` node to a `MeasurableVariable`.

    TODO FIXME: We need update/clarify the canonicalization situation so that
    these can be reliably rewritten as products of reciprocals.

    """
    numerator, denominator = node.inputs

    reciprocal_denominator = at.reciprocal(denominator)
    # `denominator` is measurable
    res = measurable_reciprocal.transform(fgraph, reciprocal_denominator.owner)
    if res:
        return measurable_mul.transform(fgraph, at.mul(numerator, res[0]).owner)

    # `numerator` is measurable
    return measurable_mul.transform(
        fgraph, at.mul(numerator, reciprocal_denominator).owner
    )


@register_measurable_ir
@node_rewriter([sub])
def measurable_sub(fgraph, node):
    r"""Rewrite a `sub` node to a `MeasurableVariable`.

    TODO FIXME: We need update/clarify the canonicalization situation so that
    these can be reliably rewritten as products of reciprocals.

    """
    minuend, subtrahend = node.inputs

    mul_subtrahend = at.mul(-1, subtrahend)

    # `subtrahend` is measurable
    res = measurable_mul.transform(fgraph, mul_subtrahend.owner)
    if res:
        return measurable_add.transform(fgraph, at.add(minuend, res[0]).owner)

    # TODO FIXME: `local_add_canonizer` will unreliably rewrite expressions like
    # `x - y` to `-y + x` (e.g. apparently when `y` is a constant?) and, as a result,
    # this will not be reached.  We're leaving this in just in case, but we
    # ultimately need to fix Aesara's canonicalizations.

    # `minuend` is measurable
    return measurable_add.transform(fgraph, at.add(minuend, mul_subtrahend).owner)


@register_measurable_ir
@node_rewriter([exp])
def measurable_exp(fgraph, node):
    """Rewrite an `exp` node to a `MeasurableVariable`."""

    def transform(measurable_input, *args):
        return ExpTransform(), (measurable_input,)

    return construct_elemwise_transform(fgraph, node, transform)


@register_measurable_ir
@node_rewriter([log])
def measurable_log(fgraph, node):
    """Rewrite a `log` node to a `MeasurableVariable`."""

    def transform(measurable_input, *args):
        return LogTransform(), (measurable_input,)

    return construct_elemwise_transform(fgraph, node, transform)


@register_measurable_ir
@node_rewriter([add])
def measurable_add(fgraph, node):
    """Rewrite an `add` node to a `MeasurableVariable`."""

    def transform(measurable_input, *other_inputs):
        transform_inputs = (
            measurable_input,
            at.add(*other_inputs) if len(other_inputs) > 1 else other_inputs[0],
        )
        transform = LocTransform(
            transform_args_fn=lambda *inputs: inputs[-1],
        )
        return transform, transform_inputs

    return construct_elemwise_transform(fgraph, node, transform)


@register_measurable_ir
@node_rewriter([mul])
def measurable_mul(fgraph, node):
    """Rewrite a `mul` node to a `MeasurableVariable`."""

    def transform(measurable_input, *other_inputs):
        transform_inputs = (
            measurable_input,
            at.mul(*other_inputs) if len(other_inputs) > 1 else other_inputs[0],
        )
        return (
            ScaleTransform(
                transform_args_fn=lambda *inputs: inputs[-1],
            ),
            transform_inputs,
        )

    return construct_elemwise_transform(fgraph, node, transform)


@register_measurable_ir
@node_rewriter([reciprocal])
def measurable_reciprocal(fgraph, node):
    """Rewrite a `reciprocal` node to a `MeasurableVariable`."""

    def transform(measurable_input, *other_inputs):
        return ReciprocalTransform(), (measurable_input,)

    return construct_elemwise_transform(fgraph, node, transform)


def construct_elemwise_transform(
    fgraph: FunctionGraph,
    node: Apply,
    transform_fn: "TransformFnType",
) -> Optional[List[Variable]]:
    """Construct a measurable transformation for an `Elemwise` node.

    Parameters
    ----------
    fgraph
        The `FunctionGraph` in which `node` resides.
    node
        The `Apply` node to be converted.
    transform_fn
        A function that takes a single measurable input and all the remaining
        inputs and returns a transform object and transformed inputs.

    Returns
    -------
    A new variable with an `Apply` node with a `MeasurableElemwiseTransform`
    that replaces `node`.

    """
    scalar_op = node.op.scalar_op

    # Check that we have a single source of measurement
    measurable_inputs = [
        inp
        for idx, inp in enumerate(node.inputs)
        if inp.owner
        and isinstance(inp.owner.op, MeasurableVariable)
        and not isinstance(inp, ValuedVariable)
    ]

    if len(measurable_inputs) != 1:
        return None

    measurable_input: TensorVariable = measurable_inputs[0]

    # Do not apply rewrite to discrete variables
    # TODO: Formalize this restriction better.
    if measurable_input.type.dtype.startswith("int"):
        return None

    # Check that other inputs are not potentially measurable, in which case this rewrite
    # would be invalid
    # TODO FIXME: This is rather costly and redundant; find a way to avoid it
    # or make it cheaper.
    other_inputs = tuple(inp for inp in node.inputs if inp is not measurable_input)

    def expand(var: TensorVariable) -> List[TensorVariable]:
        new_vars: List[TensorVariable] = []

        if (
            var.owner
            and not isinstance(var.owner.op, MeasurableVariable)
            and not isinstance(var, ValuedVariable)
        ):
            new_vars.extend(reversed(var.owner.inputs))

        return new_vars

    if any(
        ancestor_node
        for ancestor_node in walk(other_inputs, expand, False)
        if (
            ancestor_node.owner
            and isinstance(ancestor_node.owner.op, MeasurableVariable)
            and not isinstance(ancestor_node, ValuedVariable)
        )
    ):
        return None

    # Make base_measure outputs unmeasurable
    # This seems to be the only thing preventing nested rewrites from being erased
    # TODO: Is this still needed?
    if not isinstance(measurable_input.owner.op, ValuedVariable):
        measurable_input = assign_custom_measurable_outputs(measurable_input.owner)

    measurable_input_idx = 0
    transform, transform_inputs = transform_fn(measurable_input, *other_inputs)

    transform_op = MeasurableElemwiseTransform(
        scalar_op=scalar_op,
        transform=transform,
        measurable_input_idx=measurable_input_idx,
    )
    transform_out = transform_op.make_node(*transform_inputs).default_output()
    transform_out.name = node.outputs[0].name

    return [transform_out]


class LocTransform(RVTransform):
    name = "loc"

    def __init__(self, transform_args_fn):
        self.transform_args_fn = transform_args_fn

    def forward(self, value, *inputs):
        loc = self.transform_args_fn(*inputs)
        return value + loc

    def backward(self, value, *inputs):
        loc = self.transform_args_fn(*inputs)
        return value - loc

    def log_jac_det(self, value, *inputs):
        return at.zeros_like(value)


class ScaleTransform(RVTransform):
    name = "scale"

    def __init__(self, transform_args_fn):
        self.transform_args_fn = transform_args_fn

    def forward(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return value * scale

    def backward(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return value / scale

    def log_jac_det(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return -at.log(at.abs(scale))


class LogTransform(RVTransform):
    name = "log"

    def forward(self, value, *inputs):
        return at.log(value)

    def backward(self, value, *inputs):
        return at.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


class ExpTransform(RVTransform):
    name = "exp"

    def forward(self, value, *inputs):
        return at.exp(value)

    def backward(self, value, *inputs):
        return at.log(value)

    def log_jac_det(self, value, *inputs):
        return -at.log(value)


class ReciprocalTransform(RVTransform):
    name = "reciprocal"

    def forward(self, value, *inputs):
        return at.reciprocal(value)

    def backward(self, value, *inputs):
        return at.reciprocal(value)

    def log_jac_det(self, value, *inputs):
        return -2 * at.log(value)


class IntervalTransform(RVTransform):
    name = "interval"

    def __init__(
        self, args_fn: Callable[..., Tuple[Optional[Variable], Optional[Variable]]]
    ):
        """

        Parameters
        ==========
        args_fn
            Function that expects inputs of RandomVariable and returns the lower
            and upper bounds for the interval transformation. If one of these is
            None, the RV is considered to be unbounded on the respective edge.
        """
        self.args_fn = args_fn

    def forward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            return at.log(value - a) - at.log(b - value)
        elif a is not None:
            return at.log(value - a)
        elif b is not None:
            return at.log(b - value)
        else:
            raise ValueError("Both edges of IntervalTransform cannot be None")

    def backward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            sigmoid_x = at.sigmoid(value)
            return sigmoid_x * b + (1 - sigmoid_x) * a
        elif a is not None:
            return at.exp(value) + a
        elif b is not None:
            return b - at.exp(value)
        else:
            raise ValueError("Both edges of IntervalTransform cannot be None")

    def log_jac_det(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            s = at.softplus(-value)
            return at.log(b - a) - 2 * s - value
        elif a is None and b is None:
            raise ValueError("Both edges of IntervalTransform cannot be None")
        else:
            return value


class LogOddsTransform(RVTransform):
    name = "logodds"

    def backward(self, value, *inputs):
        return at.expit(value)

    def forward(self, value, *inputs):
        return at.log(value / (1 - value))

    def log_jac_det(self, value, *inputs):
        sigmoid_value = at.sigmoid(value)
        return at.log(sigmoid_value) + at.log1p(-sigmoid_value)


class SimplexTransform(RVTransform):
    name = "simplex"

    def forward(self, value, *inputs):
        log_value = at.log(value)
        shift = at.sum(log_value, -1, keepdims=True) / value.shape[-1]
        return log_value[..., :-1] - shift

    def backward(self, value, *inputs):
        value = at.concatenate([value, -at.sum(value, -1, keepdims=True)], axis=-1)
        exp_value_max = at.exp(value - at.max(value, -1, keepdims=True))
        return exp_value_max / at.sum(exp_value_max, -1, keepdims=True)

    def log_jac_det(self, value, *inputs):
        N = value.shape[-1] + 1
        sum_value = at.sum(value, -1, keepdims=True)
        value_sum_expanded = value + sum_value
        value_sum_expanded = at.concatenate(
            [value_sum_expanded, at.zeros(sum_value.shape)], -1
        )
        logsumexp_value_expanded = at.logsumexp(value_sum_expanded, -1, keepdims=True)
        res = at.log(N) + (N * sum_value) - (N * logsumexp_value_expanded)
        return at.sum(res, -1)


class CircularTransform(RVTransform):
    name = "circular"

    def backward(self, value, *inputs):
        return at.arctan2(at.sin(value), at.cos(value))

    def forward(self, value, *inputs):
        return at.as_tensor_variable(value)

    def log_jac_det(self, value, *inputs):
        return at.zeros(value.shape)


class ChainedTransform(RVTransform):
    name = "chain"

    def __init__(self, transform_list, base_op):
        self.transform_list = transform_list
        self.base_op = base_op

    def forward(self, value, *inputs):
        for transform in self.transform_list:
            value = transform.forward(value, *inputs)
        return value

    def backward(self, value, *inputs):
        for transform in reversed(self.transform_list):
            value = transform.backward(value, *inputs)
        return value

    def log_jac_det(self, value, *inputs):
        value = at.as_tensor_variable(value)
        det_list = []
        ndim0 = value.ndim
        for transform in reversed(self.transform_list):
            det_ = transform.log_jac_det(value, *inputs)
            det_list.append(det_)
            ndim0 = min(ndim0, det_.ndim)
            value = transform.backward(value, *inputs)
        # match the shape of the smallest jacobian_det
        det = 0.0
        for det_ in det_list:
            if det_.ndim > ndim0:
                det += det_.sum(axis=-1)
            else:
                det += det_
        return det


def _create_transformed_rv_op(
    rv_op: Op,
    transform: RVTransform,
    *,
    default: bool = False,
    cls_dict_extra: Optional[Dict] = None,
) -> Op:
    """Create a new transformed variable instance given a base `RandomVariable` `Op`.

    This will essentially copy the `type` of the given `Op` instance, create a
    copy of said `Op` instance and change it's `type` to the new one.

    In the end, we have an `Op` instance that will map to a `RVTransform` while
    also behaving exactly as it did before.

    Parameters
    ==========
    rv_op
        The `RandomVariable` for which we want to construct a `TransformedRV`.
    transform
        The `RVTransform` for `rv_op`.
    default
        If ``False`` do not make `transform` the default transform for `rv_op`.
    cls_dict_extra
        Additional class members to add to the constructed `TransformedRV`.

    """

    trans_name = getattr(transform, "name", "transformed")
    rv_op_type = type(rv_op)
    rv_type_name = rv_op_type.__name__
    cls_dict = rv_op_type.__dict__.copy()
    rv_name = cls_dict.get("name", "")
    if rv_name:
        cls_dict["name"] = f"{rv_name}_{trans_name}"
    cls_dict["transform"] = transform

    if cls_dict_extra is not None:
        cls_dict.update(cls_dict_extra)

    new_op_type = type(f"Transformed{rv_type_name}", (rv_op_type,), cls_dict)

    MeasurableVariable.register(new_op_type)

    @_logprob.register(new_op_type)
    def transformed_logprob(op, values, *inputs, use_jacobian=True, **kwargs):
        """Compute the log-likelihood graph for a `TransformedRV`.

        We assume that the value variable was back-transformed to be on the natural
        support of the respective random variable.
        """
        (value,) = values

        assert value.owner.op == transformed_variable

        backward_val, trans_val = value.owner.inputs

        logprob = _logprob(rv_op, (backward_val,), *inputs, **kwargs)

        if use_jacobian:
            jacobian = op.transform.log_jac_det(trans_val, *inputs)
            logprob += jacobian

        return logprob

    transform_op = rv_op_type if default else new_op_type

    @_default_transformed_rv.register(transform_op)
    def class_transformed_rv(op, node):
        new_op = new_op_type()
        res = new_op.make_node(*node.inputs)
        res.outputs[1].name = node.outputs[1].name
        return res

    new_op = copy(rv_op)
    new_op.__class__ = new_op_type

    return new_op


create_default_transformed_rv_op = partial(_create_transformed_rv_op, default=True)


TransformedUniformRV = create_default_transformed_rv_op(
    at.random.uniform,
    # inputs[3] = lower; inputs[4] = upper
    IntervalTransform(lambda *inputs: (inputs[3], inputs[4])),
)
TransformedParetoRV = create_default_transformed_rv_op(
    at.random.pareto,
    # inputs[3] = alpha
    IntervalTransform(lambda *inputs: (inputs[3], None)),
)
TransformedTriangularRV = create_default_transformed_rv_op(
    at.random.triangular,
    # inputs[3] = lower; inputs[5] = upper
    IntervalTransform(lambda *inputs: (inputs[3], inputs[5])),
)
TransformedHalfNormalRV = create_default_transformed_rv_op(
    at.random.halfnormal,
    # inputs[3] = loc
    IntervalTransform(lambda *inputs: (inputs[3], None)),
)
TransformedWaldRV = create_default_transformed_rv_op(
    at.random.wald,
    LogTransform(),
)
TransformedExponentialRV = create_default_transformed_rv_op(
    at.random.exponential,
    LogTransform(),
)
TransformedLognormalRV = create_default_transformed_rv_op(
    at.random.lognormal,
    LogTransform(),
)
TransformedHalfCauchyRV = create_default_transformed_rv_op(
    at.random.halfcauchy,
    LogTransform(),
)
TransformedGammaRV = create_default_transformed_rv_op(
    at.random.gamma,
    LogTransform(),
)
TransformedInvGammaRV = create_default_transformed_rv_op(
    at.random.invgamma,
    LogTransform(),
)
TransformedChiSquareRV = create_default_transformed_rv_op(
    at.random.chisquare,
    LogTransform(),
)
TransformedWeibullRV = create_default_transformed_rv_op(
    at.random.weibull,
    LogTransform(),
)
TransformedBetaRV = create_default_transformed_rv_op(
    at.random.beta,
    LogOddsTransform(),
)
TransformedVonMisesRV = create_default_transformed_rv_op(
    at.random.vonmises,
    CircularTransform(),
)
TransformedDirichletRV = create_default_transformed_rv_op(
    at.random.dirichlet,
    SimplexTransform(),
)
