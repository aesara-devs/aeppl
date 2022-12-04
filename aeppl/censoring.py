from typing import TYPE_CHECKING, List, Optional

import aesara.tensor as at
import numpy as np
from aesara.graph.basic import Node
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import node_rewriter
from aesara.scalar.basic import ceil as scalar_ceil
from aesara.scalar.basic import clip as scalar_clip
from aesara.scalar.basic import floor as scalar_floor
from aesara.scalar.basic import round_half_to_even as scalar_round_half_to_even
from aesara.tensor.math import ceil, clip, floor, round_half_to_even
from aesara.tensor.var import TensorConstant

from aeppl.abstract import (
    MeasurableElemwise,
    MeasurableVariable,
    ValuedVariable,
    assign_custom_measurable_outputs,
)
from aeppl.logprob import CheckParameterValue, _logcdf, _logprob, logdiffexp
from aeppl.rewriting import measurable_ir_rewrites_db

if TYPE_CHECKING:
    from aesara.graph.basic import Variable
    from aesara.graph.op import Op


class MeasurableClip(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a clipped RV sub-graph."""


measurable_clip = MeasurableClip(scalar_clip)


@node_rewriter([clip])
def find_measurable_clips(
    fgraph: FunctionGraph, node: Node
) -> Optional[List["Variable"]]:
    # TODO: Canonicalize x[x>ub] = ub -> clip(x, x, ub)

    if isinstance(node.op, MeasurableClip):
        return None  # pragma: no cover

    clipped_var = node.outputs[0]
    base_var, lower_bound, upper_bound = node.inputs

    if not (
        base_var.owner
        and isinstance(base_var.owner.op, MeasurableVariable)
        and not isinstance(base_var, ValuedVariable)
    ):
        return None

    # Replace bounds by `+-inf` if `y = clip(x, x, ?)` or `y=clip(x, ?, x)`
    # This is used in `clip_logprob` to generate a more succint logprob graph
    # for one-sided clipped random variables
    lower_bound = lower_bound if (lower_bound is not base_var) else at.constant(-np.inf)
    upper_bound = upper_bound if (upper_bound is not base_var) else at.constant(np.inf)

    # Make base_var unmeasurable
    unmeasurable_base_var = assign_custom_measurable_outputs(base_var.owner)
    clipped_rv_node = measurable_clip.make_node(
        unmeasurable_base_var, lower_bound, upper_bound
    )
    clipped_rv = clipped_rv_node.outputs[0]

    clipped_rv.name = clipped_var.name

    return [clipped_rv]


measurable_ir_rewrites_db.register(
    "find_measurable_clips",
    find_measurable_clips,
    "basic",
    "censoring",
)


@_logprob.register(MeasurableClip)
def clip_logprob(op, values, base_rv, lower_bound, upper_bound, **kwargs):
    r"""Logprob of a clipped censored distribution

    The probability is given by

    .. math::

        \begin{cases}
            0 & \text{for } x < lower, \\
            \text{CDF}(lower, dist) & \text{for } x = lower, \\
            \text{P}(x, dist) & \text{for } lower < x < upper, \\
            1-\text{CDF}(upper, dist) & \text {for} x = upper, \\
            0 & \text{for } x > upper,
        \end{cases}

    """
    (value,) = values

    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs

    logprob = _logprob(base_rv_op, (value,), *base_rv_inputs, **kwargs)
    logcdf = _logcdf(base_rv_op, value, *base_rv_inputs, **kwargs)

    if base_rv_op.name:
        logprob.name = f"{base_rv_op}_logprob"
        logcdf.name = f"{base_rv_op}_logcdf"

    is_lower_bounded, is_upper_bounded = False, False
    if not (
        isinstance(upper_bound, TensorConstant) and np.all(np.isinf(upper_bound.value))
    ):
        is_upper_bounded = True

        logccdf = at.log1mexp(logcdf)
        # For right clipped discrete RVs, we need to add an extra term
        # corresponding to the pmf at the upper bound
        if base_rv.dtype.startswith("int"):
            logccdf = at.logaddexp(logccdf, logprob)

        logprob = at.switch(
            at.eq(value, upper_bound),
            logccdf,
            at.switch(at.gt(value, upper_bound), -np.inf, logprob),
        )
    if not (
        isinstance(lower_bound, TensorConstant)
        and np.all(np.isneginf(lower_bound.value))
    ):
        is_lower_bounded = True
        logprob = at.switch(
            at.eq(value, lower_bound),
            logcdf,
            at.switch(at.lt(value, lower_bound), -np.inf, logprob),
        )

    if is_lower_bounded and is_upper_bounded:
        logprob = CheckParameterValue("lower_bound <= upper_bound")(
            logprob, at.all(at.le(lower_bound, upper_bound))
        )

    return logprob


class MeasurableRound(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a clipped RV sub-graph."""


measurable_ceil = MeasurableRound(scalar_ceil)
measurable_floor = MeasurableRound(scalar_floor)
measurable_round_half_to_even = MeasurableRound(scalar_round_half_to_even)


@node_rewriter([ceil])
def find_measurable_ceil(fgraph: FunctionGraph, node: Node):
    return construct_measurable_rounding(fgraph, node, measurable_ceil)


@node_rewriter([floor])
def find_measurable_floor(fgraph: FunctionGraph, node: Node):
    return construct_measurable_rounding(fgraph, node, measurable_floor)


@node_rewriter([round_half_to_even])
def find_measurable_round_half_to_even(fgraph: FunctionGraph, node: Node):
    return construct_measurable_rounding(fgraph, node, measurable_round_half_to_even)


measurable_ir_rewrites_db.register(
    "find_measurable_ceil",
    find_measurable_ceil,
    "basic",
    "censoring",
)
measurable_ir_rewrites_db.register(
    "find_measurable_floor",
    find_measurable_floor,
    "basic",
    "censoring",
)
measurable_ir_rewrites_db.register(
    "find_measurable_round_half_to_even",
    find_measurable_round_half_to_even,
    "basic",
    "censoring",
)


def construct_measurable_rounding(
    fgraph: FunctionGraph, node: Node, rounded_op: "Op"
) -> Optional[List["Variable"]]:

    if isinstance(node.op, MeasurableRound):
        return None  # pragma: no cover

    (rounded_var,) = node.outputs
    (base_var,) = node.inputs

    if not (
        base_var.owner
        and isinstance(base_var.owner.op, MeasurableVariable)
        and not isinstance(base_var, ValuedVariable)
        # Rounding only makes sense for continuous variables
        and base_var.dtype.startswith("float")
    ):
        return None

    # Make base_var unmeasurable
    unmeasurable_base_var = assign_custom_measurable_outputs(base_var.owner)

    rounded_rv = rounded_op.make_node(unmeasurable_base_var).default_output()
    rounded_rv.name = rounded_var.name
    return [rounded_rv]


@_logprob.register(MeasurableRound)
def round_logprob(op, values, base_rv, **kwargs):
    r"""Logprob of a rounded censored distribution

    The probability of a distribution rounded to the nearest integer is given by

    .. math::

        \begin{cases}
            \text{CDF}(x+\frac{1}{2}, dist) - \text{CDF}(x-\frac{1}{2}, dist) & \text{for } x \in \mathbb{Z}, \\
            0 & \text{otherwise},
        \end{cases}

    The probability of a distribution rounded up is given by

    .. math::

        \begin{cases}
            \text{CDF}(x, dist) - \text{CDF}(x-1, dist) & \text{for } x \in \mathbb{Z}, \\
            0 & \text{otherwise},
        \end{cases}

    The probability of a distribution rounded down is given by

    .. math::

        \begin{cases}
            \text{CDF}(x+1, dist) - \text{CDF}(x, dist) & \text{for } x \in \mathbb{Z}, \\
            0 & \text{otherwise},
        \end{cases}

    """
    (value,) = values

    if op == measurable_round_half_to_even:
        value = at.round(value)
        value_upper = value + 0.5
        value_lower = value - 0.5
    elif op == measurable_floor:
        value = at.floor(value)
        value_upper = value + 1.0
        value_lower = value
    elif op == measurable_ceil:
        value = at.ceil(value)
        value_upper = value
        value_lower = value - 1.0
    else:
        raise TypeError(f"Unsupported scalar_op {op.scalar_op}")  # pragma: no cover

    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs

    logcdf_upper = _logcdf(base_rv_op, value_upper, *base_rv_inputs, **kwargs)
    logcdf_lower = _logcdf(base_rv_op, value_lower, *base_rv_inputs, **kwargs)

    if base_rv_op.name:
        logcdf_upper.name = f"{base_rv_op}_logcdf_upper"
        logcdf_lower.name = f"{base_rv_op}_logcdf_lower"

    return logdiffexp(logcdf_upper, logcdf_lower)
