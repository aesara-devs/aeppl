from functools import singledispatch
from typing import List, Optional, Tuple

import aesara.tensor as at
import aesara.tensor.random.basic as arb
import numpy as np
from aesara import scan, shared
from aesara.compile.builders import OpFromGraph
from aesara.graph.basic import Node
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import local_optimizer
from aesara.raise_op import CheckAndRaise
from aesara.scalar.basic import Clip
from aesara.scalar.basic import clip as scalar_clip
from aesara.scan import until
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorConstant, TensorVariable

from aeppl.abstract import (
    MeasurableVariable,
    _get_measurable_outputs,
    assign_custom_measurable_outputs,
)
from aeppl.logprob import (
    CheckParameterValue,
    _logcdf,
    _logprob,
    icdf,
    logcdf,
    logdiffexp,
)
from aeppl.opt import measurable_ir_rewrites_db


class CensoredRV(Elemwise):
    """A placeholder used to specify a log-likelihood for a censored RV sub-graph."""


MeasurableVariable.register(CensoredRV)


@local_optimizer(tracks=[Elemwise])
def find_censored_rvs(fgraph: FunctionGraph, node: Node) -> Optional[List[CensoredRV]]:
    # TODO: Canonicalize x[x>ub] = ub -> clip(x, x, ub)

    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    if isinstance(node.op, CensoredRV):
        return None  # pragma: no cover

    if not (isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, Clip)):
        return None

    clipped_var = node.outputs[0]
    base_var, lower_bound, upper_bound = node.inputs

    if not (
        base_var.owner
        and isinstance(base_var.owner.op, MeasurableVariable)
        and base_var not in rv_map_feature.rv_values
    ):
        return None

    # Replace bounds by `+-inf` if `y = clip(x, x, ?)` or `y=clip(x, ?, x)`
    # This is used in `censor_logprob` to generate a more succint logprob graph
    # for one-sided censored random variables
    lower_bound = lower_bound if (lower_bound is not base_var) else at.constant(-np.inf)
    upper_bound = upper_bound if (upper_bound is not base_var) else at.constant(np.inf)

    censored_op = CensoredRV(scalar_clip)
    # Make base_var unmeasurable
    unmeasurable_base_var = assign_custom_measurable_outputs(base_var.owner)
    censored_rv_node = censored_op.make_node(
        unmeasurable_base_var, lower_bound, upper_bound
    )
    censored_rv = censored_rv_node.outputs[0]

    censored_rv.name = clipped_var.name

    return [censored_rv]


measurable_ir_rewrites_db.register(
    "find_censored_rvs",
    find_censored_rvs,
    0,
    "basic",
    "censoring",
)


@_logprob.register(CensoredRV)
def censor_logprob(op, values, base_rv, lower_bound, upper_bound, **kwargs):
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

        logccdf = at.log(1 - at.exp(logcdf))
        # For right censored discrete RVs, we need to add an extra term
        # corresponding to the pmf at the upper bound
        if base_rv_op.dtype.startswith("int"):
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


class TruncatedRV(OpFromGraph):
    """An `Op` constructed from an Aesara graph that represents a truncated univariate random variable."""

    default_output = 0
    base_rv_op = None

    def __init__(self, base_rv_op: Op, *args, **kwargs):
        self.base_rv_op = base_rv_op
        super().__init__(*args, **kwargs)


MeasurableVariable.register(TruncatedRV)


@_get_measurable_outputs.register(TruncatedRV)
def _get_measurable_outputs_TruncatedRV(op, node):
    return [node.outputs[0]]


@singledispatch
def _truncated(op: Op, lower, upper, *params):
    """Return the truncated equivalent of another `RandomVariable`."""
    raise NotImplementedError(
        f"{op} does not have an equivalent truncated version implemented"
    )


class TruncationError(Exception):
    """Exception for errors generated from truncated graphs"""


class TruncationCheck(CheckAndRaise):
    """Implements a check in truncated graphs.

    Raises `TruncationError` if the check is not True.
    """

    def __init__(self, msg=""):
        super().__init__(TruncationError, msg)

    def __str__(self):
        return f"TruncationCheck{{{self.msg}}}"


def truncate(
    rv: TensorVariable, lower=None, upper=None, max_n_steps: int = 10_000, rng=None
) -> Tuple[TensorVariable, Tuple[TensorVariable, TensorVariable]]:
    """Truncate a univariate `RandomVariable` between lower and upper.

    If lower or upper is ``None``, the variable is not truncated on that side.

    Depending on dispatched implementations, this function returns either a specialized
    `Op`, or equivalent graph representing the truncation process, via inverse CDF
    sampling, or rejection sampling.

    The argument `max_n_steps` controls the maximum number of resamples that are
    attempted when performing rejection sampling. A `TruncationError` is raised if
    convergence is not reached after that many steps.

    Returns
    =======
    `TensorVariable` graph representing the truncated `RandomVariable` and respective updates
    """

    if lower is None and upper is None:
        raise ValueError("lower and upper cannot both be None")

    if not (isinstance(rv.owner.op, RandomVariable) and rv.owner.op.ndim_supp == 0):
        raise NotImplementedError(
            f"Truncation is only implemented for univariate random variables, got {rv.owner.op}"
        )

    lower = at.as_tensor_variable(lower) if lower is not None else at.constant(-np.inf)
    upper = at.as_tensor_variable(upper) if upper is not None else at.constant(np.inf)

    if rng is None:
        rng = shared(np.random.RandomState(), borrow=True)

    # Try to use specialized Op
    try:
        truncated_rv, updates = _truncated(
            rv.owner.op, lower, upper, rng, *rv.owner.inputs[1:]
        )
        return truncated_rv, updates
    except NotImplementedError:
        pass

    # Variables with `_` suffix identify dummy inputs for the OpFromGraph
    # We will use the Shared RNG variable directly because Scan demands it, even
    # though it would not be necessary for the icdf OpFromGraph
    graph_inputs = [*rv.owner.inputs[1:], lower, upper]
    graph_inputs_ = [inp.type() for inp in graph_inputs]
    *rv_inputs_, lower_, upper_ = graph_inputs_
    rv_ = rv.owner.op.make_node(rng, *rv_inputs_).default_output()

    # Try to use inverted cdf sampling
    try:
        # For left truncated discrete RVs, we need to include the whole lower bound.
        # This may result in draws below the truncation range, if any uniform == 0
        lower_value = lower_ - 1 if rv.owner.op.dtype.startswith("int") else lower_
        cdf_lower_ = at.exp(logcdf(rv_, lower_value))
        cdf_upper_ = at.exp(logcdf(rv_, upper_))
        uniform_ = at.random.uniform(
            cdf_lower_,
            cdf_upper_,
            rng=rng,
            size=rv_inputs_[0],
        )
        truncated_rv_ = icdf(rv_, uniform_)
        truncated_rv = TruncatedRV(
            base_rv_op=rv.owner.op,
            inputs=graph_inputs_,
            outputs=[truncated_rv_, uniform_.owner.outputs[0]],
            inline=True,
        )(*graph_inputs)
        updates = {truncated_rv.owner.inputs[-1]: truncated_rv.owner.outputs[-1]}
        return truncated_rv, updates
    except NotImplementedError:
        pass

    # Fallback to rejection sampling
    # TODO: Handle potential broadcast by lower / upper
    def loop_fn(truncated_rv, reject_draws, lower, upper, rng, *rv_inputs):
        next_rng, new_truncated_rv = rv.owner.op.make_node(rng, *rv_inputs).outputs
        truncated_rv = at.set_subtensor(
            truncated_rv[reject_draws],
            new_truncated_rv[reject_draws],
        )
        reject_draws = at.or_((truncated_rv < lower), (truncated_rv > upper))

        return (
            (truncated_rv, reject_draws),
            [(rng, next_rng)],
            until(~at.any(reject_draws)),
        )

    (truncated_rv_, reject_draws_), updates = scan(
        loop_fn,
        outputs_info=[
            at.zeros_like(rv_),
            at.ones_like(rv_, dtype=bool),
        ],
        non_sequences=[lower_, upper_, rng, *rv_inputs_],
        n_steps=max_n_steps,
        strict=True,
    )

    truncated_rv_ = truncated_rv_[-1]
    convergence_ = ~at.any(reject_draws_[-1])
    truncated_rv_ = TruncationCheck(
        f"Truncation did not converge in {max_n_steps} steps"
    )(truncated_rv_, convergence_)

    truncated_rv = TruncatedRV(
        base_rv_op=rv.owner.op,
        inputs=graph_inputs_,
        outputs=[truncated_rv_, tuple(updates.values())[0]],
        inline=True,
    )(*graph_inputs)
    updates = {truncated_rv.owner.inputs[-1]: truncated_rv.owner.outputs[-1]}
    return truncated_rv, updates


@_logprob.register(TruncatedRV)
def truncated_logprob(op, values, *inputs, **kwargs):
    (value,) = values

    *rv_inputs, lower_bound, upper_bound, rng = inputs
    rv_inputs = [rng, *rv_inputs]

    base_rv_op = op.base_rv_op
    logp = _logprob(base_rv_op, (value,), *rv_inputs, **kwargs)
    # For left truncated RVs, we don't want to include the lower bound in the
    # normalization term
    lower_bound_value = (
        lower_bound - 1 if base_rv_op.dtype.startswith("int") else lower_bound
    )
    lower_logcdf = _logcdf(base_rv_op, lower_bound_value, *rv_inputs, **kwargs)
    upper_logcdf = _logcdf(base_rv_op, upper_bound, *rv_inputs, **kwargs)

    if base_rv_op.name:
        logp.name = f"{base_rv_op}_logprob"
        lower_logcdf.name = f"{base_rv_op}_lower_logcdf"
        upper_logcdf.name = f"{base_rv_op}_upper_logcdf"

    is_lower_bounded = not (
        isinstance(lower_bound, TensorConstant)
        and np.all(np.isneginf(lower_bound.value))
    )
    is_upper_bounded = not (
        isinstance(upper_bound, TensorConstant) and np.all(np.isinf(upper_bound.value))
    )

    lognorm = 0
    if is_lower_bounded and is_upper_bounded:
        lognorm = logdiffexp(upper_logcdf, lower_logcdf)
    elif is_lower_bounded:
        lognorm = at.log1mexp(lower_logcdf)
    elif is_upper_bounded:
        lognorm = upper_logcdf

    logp = logp - lognorm

    if is_lower_bounded:
        logp = at.switch(value < lower_bound, -np.inf, logp)

    if is_upper_bounded:
        logp = at.switch(value <= upper_bound, logp, -np.inf)

    if is_lower_bounded and is_upper_bounded:
        logp = CheckParameterValue("lower_bound <= upper_bound")(
            logp, at.all(at.le(lower_bound, upper_bound))
        )

    return logp


@_truncated.register(arb.UniformRV)
def uniform_truncated(op, lower, upper, rng, size, dtype, lower_orig, upper_orig):
    truncated_uniform = at.random.uniform(
        at.max((lower_orig, lower)),
        at.min((upper_orig, upper)),
        rng=rng,
        size=size,
        dtype=dtype,
    )
    return truncated_uniform, {
        truncated_uniform.owner.inputs[0]: truncated_uniform.owner.outputs[0]
    }
