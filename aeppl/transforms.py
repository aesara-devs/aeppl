import abc
from functools import singledispatch
from typing import Dict, List, Optional, Type, Union

import aesara.tensor as at
from aesara.gradient import jacobian
from aesara.graph.basic import Node, Variable
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import Feature, in2out, local_optimizer
from aesara.graph.utils import MetaType
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable

from aeppl.logprob import _logprob


class Transform(abc.ABC):
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
        # return at.log(at.abs_(jac))
        phi_inv = self.backward(value, *inputs)
        return at.log(at.nlinalg.det(at.atleast_2d(jacobian(phi_inv, [value]))))


class TransformedRVMeta(MetaType):
    def __new__(cls, name, bases, clsdict):
        cls_res = super().__new__(cls, name, bases, clsdict)
        base_op = clsdict.get("base_op", None)
        default = clsdict.get("default", False)

        if default and base_op is not None:
            # Create dispatch functions
            @_default_transformed_rv_op.register(type(base_op))
            def default_transformed_rv_op(op):
                return cls_res()

        return cls_res


class TransformedRV(RandomVariable, metaclass=TransformedRVMeta):
    r"""A base class for transformed `RandomVariable`\s."""


def create_transformed_rv_op(
    rv_op: Op,
    transform: Transform,
    *,
    default: bool = False,
    cls_dict_extra: Optional[Dict] = None,
) -> Type[TransformedRV]:
    """Create a new ``TransformedRV`` given a base ``RandomVariable`` ``Op``

    Pass `default = True` to specify that the returned ``TransformedRV`` should
    be the default transformation for the base ``RandomVariable``

    """

    trans_name = getattr(transform, "name", "transformed")
    rv_type_name = type(rv_op).__name__
    cls_dict = type(rv_op).__dict__.copy()
    rv_name = cls_dict.get("name", "")
    if rv_name:
        cls_dict["name"] = f"{rv_name}_{trans_name}"
    cls_dict["base_op"] = rv_op
    cls_dict["transform"] = transform
    cls_dict["default"] = default

    if cls_dict_extra is not None:
        cls_dict.update(cls_dict_extra)

    new_op_type = type(f"Transformed{rv_type_name}", (TransformedRV,), cls_dict)

    return new_op_type


@singledispatch
def _default_transformed_rv_op(
    op: Op,
) -> Optional[Type[TransformedRV]]:
    """Return a default ``TransformedRV`` for a given ``RandomVariable`` ``Op``.

    This function dispatches on the type of ``Op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new transforms for a
    ``RandomVariable``, register a function on this dispatcher.

    """
    return None


DEFAULT_TRANSFORM = object()


class TransformValuesMapping(Feature):
    r"""A `Feature` that maintains a map between value variables and their transforms."""

    def __init__(self, values_to_transforms):
        self.values_to_transforms = values_to_transforms

    def on_attach(self, fgraph):
        if hasattr(fgraph, "transform_values_mapping"):
            raise ValueError(
                f"{fgraph} already has the `TransformValuesMapping` feature attached."
            )

        fgraph.transform_values_mapping = self

    def on_detach(self, fgraph):
        if self.values_to_transforms:
            raise RuntimeError(
                "The following value variables could not be transformed: {}".format(
                    *self.values_to_transforms
                )
            )
        del fgraph.transform_values_mapping


class TransformValuesOpt:
    # TODO: Type hints, how to specify the DEFAULT_TRANSFORM instead of just `object`?
    def __init__(
        self,
        values_to_transforms: Dict[TensorVariable, Union[Transform, object, None]],
    ):
        """
        Implements transformation rewrite of value variables, by combining the
        `TransformValuesMapping` Feature with the `transform_values` local
        optimizer.

        Parameters
        ==========
        values_to_transforms
            Mapping between value variables and their transformations.
            Each value variable can be assigned one of `RVTransform`,
            `DEFAULT_TRANSFORM`, or `None`. If a transform is not specified for
            a specific value variable it will not be transformed.

        Raises
        ======
        RuntimeError
            If any of the specified value variables is not successfully transformed

        """

        self.values_to_transforms = values_to_transforms

    def optimize(self, fgraph: FunctionGraph):
        values_transforms_feature = TransformValuesMapping(
            self.values_to_transforms.copy()
        )
        fgraph.attach_feature(values_transforms_feature)

        transform_values_opt = in2out(transform_values, ignore_newtrees=True)
        transform_values_opt.optimize(fgraph)

        fgraph.remove_feature(values_transforms_feature)


@local_optimizer(tracks=None)
def transform_values(
    fgraph: FunctionGraph, node: Node
) -> Optional[List[TransformedRV]]:
    """
    Apply transforms to the value variables specified in a `TransformValuesMapping`
    Feature. It is assumed that the input value variables correspond to forward
    transformations, usually chosen in such a way that the values are unconstrained
    on the real line.

    e.g., if Y ~ HalfNormal, we assume the respective value variable is specified
    on the log scale and back-transform it to obtain Y on the natural scale.
    """

    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)
    transform_map_feature = getattr(fgraph, "transform_values_mapping", None)

    if rv_map_feature is None or transform_map_feature is None:
        return None  # pragma: no cover

    rv_var = node.default_output()

    value_var = rv_map_feature.rv_values.get(rv_var, None)
    if value_var is None:
        return None

    transform = transform_map_feature.values_to_transforms.pop(value_var, None)

    if transform is None:
        return None
    elif transform is DEFAULT_TRANSFORM:
        transformed_rv_op = _default_transformed_rv_op(node.op)
        if transformed_rv_op is None:
            return None
        transform = transformed_rv_op.transform
    else:
        transformed_rv_op = create_transformed_rv_op(node.op, transform)()

    # Create TransformedRV
    transformed_rv_node = transformed_rv_op.make_node(*node.inputs)
    new_rv_var = transformed_rv_node.outputs[1]
    new_rv_var.name = rv_var.name

    # We now assume that the original value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    new_value_var = transform.backward(value_var, *node.inputs)

    if value_var.name and getattr(transform, "name", None):
        new_value_var.name = f"{value_var.name}_{transform.name}"

    # Map old and new transformed RVs to new value_var
    rv_map_feature.rv_values[new_rv_var] = new_value_var
    rv_map_feature.rv_values[rv_var] = new_value_var

    return transformed_rv_node.outputs


@_logprob.register(TransformedRV)
def transformed_logprob(op, value, *inputs, name=None, **kwargs):
    """
    Compute logp graph for a TransformedRV whose value variable was already
    back-transformed to be on the natural support of the base random variable.
    """

    logprob = _logprob(op.base_op, value, *inputs, name=name, **kwargs)

    # TODO: Make sure the backward and forward functions for standard transforms
    #  are optimized away by Aesara (e.g, Sigmoid is not), otherwise our graphs
    #  are more complex than what they need to be
    original_forward_value = op.transform.forward(value, *inputs)
    jacobian = op.transform.log_jac_det(original_forward_value, *inputs)

    if name:
        logprob.name = f"{name}_logprob"
        jacobian.name = f"{name}_logprob_jac"

    return logprob + jacobian


class LogTransform(Transform):
    name = "log"

    def forward(self, value, *inputs):
        return at.log(value)

    def backward(self, value, *inputs):
        return at.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


class IntervalTransform(Transform):
    name = "interval"

    def __init__(self, args_fn):
        self.args_fn = args_fn

    def forward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            return at.log(value - a) - at.log(b - value)
        elif a is not None:
            return at.log(value - a)
        elif b is not None:
            return at.log(b - value)

    def backward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            sigmoid_x = at.sigmoid(value)
            return sigmoid_x * b + (1 - sigmoid_x) * a
        elif a is not None:
            return at.exp(value) + a
        elif b is not None:
            return b - at.exp(value)

    def log_jac_det(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            s = at.softplus(-value)
            return at.log(b - a) - 2 * s - value
        else:
            return value


class LogOddsTransform(Transform):
    name = "logodds"

    def backward(self, value, *inputs):
        return at.expit(value)

    def forward(self, value, *inputs):
        return at.log(value / (1 - value))

    def log_jac_det(self, value, *inputs):
        sigmoid_value = at.sigmoid(value)
        return at.log(sigmoid_value) + at.log1p(-sigmoid_value)


class StickBreaking(Transform):
    name = "stickbreaking"

    def forward(self, value, *inputs):
        log_value = at.log(value)
        shift = at.sum(log_value, -1, keepdims=True) / value.shape[-1]
        return log_value[..., :-1] - shift

    def backward(self, value, *inputs):
        value = at.concatenate([value, -at.sum(value, -1, keepdims=True)])
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


class CircularTransform(Transform):
    name = "circular"

    def backward(self, value, *inputs):
        return at.arctan2(at.sin(value), at.cos(value))

    def forward(self, value, *inputs):
        return at.as_tensor_variable(value)

    def log_jac_det(self, value, *inputs):
        return at.zeros(value.shape)


TransformedUniformRV = create_transformed_rv_op(
    at.random.uniform,
    # inputs[3] = lower; inputs[4] = upper
    IntervalTransform(lambda *inputs: (inputs[3], inputs[4])),
    default=True,
)
TransformedParetoRV = create_transformed_rv_op(
    at.random.pareto,
    # inputs[3] = alpha
    IntervalTransform(lambda *inputs: (inputs[3], None)),
    default=True,
)
TransformedTriangularRV = create_transformed_rv_op(
    at.random.triangular,
    # inputs[3] = lower; inputs[5] = upper
    IntervalTransform(lambda *inputs: (inputs[3], inputs[5])),
    default=True,
)
TransformedHalfNormalRV = create_transformed_rv_op(
    at.random.halfnormal,
    # inputs[3] = loc
    IntervalTransform(lambda *inputs: (inputs[3], None)),
    default=True,
)
TransformedWaldRV = create_transformed_rv_op(
    at.random.wald,
    LogTransform(),
    default=True,
)
TransformedExponentialRV = create_transformed_rv_op(
    at.random.exponential,
    LogTransform(),
    default=True,
)
TransformedLognormalRV = create_transformed_rv_op(
    at.random.lognormal,
    LogTransform(),
    default=True,
)
TransformedHalfCauchyRV = create_transformed_rv_op(
    at.random.halfcauchy,
    LogTransform(),
    default=True,
)
TransformedGammaRV = create_transformed_rv_op(
    at.random.gamma,
    LogTransform(),
    default=True,
)
TransformedInvGammaRV = create_transformed_rv_op(
    at.random.invgamma,
    LogTransform(),
    default=True,
)
TransformedChiSquareRV = create_transformed_rv_op(
    at.random.chisquare,
    LogTransform(),
    default=True,
)
TransformedWeibullRV = create_transformed_rv_op(
    at.random.weibull,
    LogTransform(),
    default=True,
)
TransformedBetaRV = create_transformed_rv_op(
    at.random.beta,
    LogOddsTransform(),
    default=True,
)
TransformedVonMisesRV = create_transformed_rv_op(
    at.random.vonmises,
    CircularTransform(),
    default=True,
)
TransformedDirichletRV = create_transformed_rv_op(
    at.random.dirichlet,
    StickBreaking(),
    default=True,
)
