import warnings
from collections import deque
from typing import Dict, Optional, Union

import aesara.tensor as at
from aesara import config
from aesara.graph.basic import graph_inputs, io_toposort
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import compute_test_value
from aesara.graph.opt import GlobalOptimizer, LocalOptimizer
from aesara.graph.optdb import OptimizationQuery
from aesara.tensor.basic_opt import ShapeFeature
from aesara.tensor.var import TensorVariable

from aeppl.abstract import MeasurableVariable, get_measurable_outputs
from aeppl.logprob import _logprob
from aeppl.opt import PreserveRVMappings, logprob_rewrites_db
from aeppl.utils import rvs_to_value_vars


def factorized_joint_logprob(
    rv_values: Dict[TensorVariable, TensorVariable],
    warn_missing_rvs: bool = True,
    extra_rewrites: Optional[Union[GlobalOptimizer, LocalOptimizer]] = None,
    **kwargs,
) -> Dict[TensorVariable, TensorVariable]:
    r"""Create a map between variables and their log-probabilities such that the
    sum is their joint log-probability.

    The `rv_values` dictionary specifies a joint probability graph defined by
    pairs of random variables and respective measure-space input parameters

    For example, consider the following

    .. code-block:: python

        import aesara.tensor as at

        sigma2_rv = at.random.invgamma(0.5, 0.5)
        Y_rv = at.random.normal(0, at.sqrt(sigma2_rv))

    This graph for ``Y_rv`` is equivalent to the following hierarchical model:

    .. math::

        \sigma^2 \sim& \operatorname{InvGamma}(0.5, 0.5) \\
        Y \sim& \operatorname{N}(0, \sigma^2)

    If we create a value variable for ``Y_rv``, i.e. ``y_vv = at.scalar("y")``,
    the graph of ``factorized_joint_logprob({Y_rv: y_vv})`` is equivalent to the
    conditional probability :math:`\log p(Y = y \mid \sigma^2)`, with a stochastic
    ``sigma2_rv``. If we specify a value variable for ``sigma2_rv``, i.e.
    ``s_vv = at.scalar("s2")``, then ``factorized_joint_logprob({Y_rv: y_vv, sigma2_rv: s_vv})``
    yields the joint log-probability of the two variables.

    .. math::

        \log p(Y = y, \sigma^2 = s) =
            \log p(Y = y \mid \sigma^2 = s) + \log p(\sigma^2 = s)


    Parameters
    ==========
    rv_values
        A ``dict`` of variables that maps stochastic elements
        (e.g. `RandomVariable`\s) to symbolic `Variable`\s representing their
        values in a log-probability.
    warn_missing_rvs
        When ``True``, issue a warning when a `RandomVariable` is found in
        the graph and doesn't have a corresponding value variable specified in
        `rv_values`.
    extra_rewrites
        Extra rewrites to be applied (e.g. reparameterizations, transforms,
        etc.)

    Returns
    =======
    A ``dict`` that maps each value variable to the log-probability factor derived
    from the respective `RandomVariable`.

    """
    # Since we're going to clone the entire graph, we need to keep a map from
    # the old nodes to the new ones; otherwise, we won't be able to use
    # `rv_values`.
    # We start the `dict` with mappings from the value variables to themselves,
    # to prevent them from being cloned.
    memo = {v: v for v in rv_values.values()}

    # We add `ShapeFeature` because it will get rid of references to the old
    # `RandomVariable`s that have been lifted; otherwise, it will be difficult
    # to give good warnings when an unaccounted for `RandomVariable` is
    # encountered
    fgraph = FunctionGraph(
        outputs=list(rv_values.keys()),
        clone=True,
        memo=memo,
        copy_orphans=False,
        copy_inputs=False,
        features=[ShapeFeature()],
    )

    # Update `rv_values` so that it uses the new cloned variables
    rv_values = {memo[k]: v for k, v in rv_values.items()}

    # This `Feature` preserves the relationships between the original
    # random variables (i.e. keys in `rv_values`) and the new ones
    # produced when `Op`s are lifted through them.
    rv_remapper = PreserveRVMappings(rv_values)

    fgraph.attach_feature(rv_remapper)

    logprob_rewrites_db.query(OptimizationQuery(include=["basic"])).optimize(fgraph)

    if extra_rewrites is not None:
        extra_rewrites.optimize(fgraph)

    # This is the updated random-to-value-vars map with the
    # lifted variables
    lifted_rv_values = rv_remapper.rv_values
    replacements = lifted_rv_values.copy()

    # Walk the graph from its inputs to its outputs and construct the
    # log-probability
    q = deque(fgraph.toposort())

    logprob_vars = {}

    while q:
        node = q.popleft()

        if not any(o in lifted_rv_values for o in node.outputs):
            if (
                isinstance(node.op, MeasurableVariable)
                and not getattr(node.default_output().tag, "ignore_logprob", False)
                and warn_missing_rvs
            ):
                warnings.warn(
                    "Found a random variable that was neither among the observations "
                    f"nor the conditioned variables: {node}"
                )
            continue

        if isinstance(node.op, MeasurableVariable):

            outputs = get_measurable_outputs(node.op, node)

            q_rv_value_vars = [
                replacements[q_rv_var]
                for q_rv_var in outputs
                if not getattr(q_rv_var.tag, "ignore_logprob", False)
            ]

            if not q_rv_value_vars:
                continue

            # Replace `RandomVariable`s in the inputs with value variables.
            # Also, store the results in the `replacements` map so that we
            # don't need to redo these replacements.
            remapped_vars, _ = rvs_to_value_vars(
                q_rv_value_vars + list(node.inputs),
                initial_replacements=replacements,
            )
            q_rv_value_vars = remapped_vars[: len(q_rv_value_vars)]
            value_var_inputs = remapped_vars[len(q_rv_value_vars) :]

            q_logprob_var = _logprob(
                node.op,
                q_rv_value_vars,
                *value_var_inputs,
                **kwargs,
            )

            for q_rv_var in q_rv_value_vars:
                if q_rv_var.name:
                    q_logprob_var.name = f"{q_rv_var.name}_logprob"

                # Recompute test values for the changes introduced by the
                # replacements above.
                if config.compute_test_value != "off":
                    for node in io_toposort(
                        graph_inputs((q_logprob_var,)),
                        (q_logprob_var,),
                    ):
                        compute_test_value(node)

                if q_rv_var in logprob_vars:
                    raise ValueError(
                        f"More than one logprob factor was assigned to the value var {q_rv_var}"
                    )

                logprob_vars[q_rv_var] = q_logprob_var

        else:
            raise NotImplementedError(
                f"A measure/probability could not be derived for {node}"
            )

    return logprob_vars


def joint_logprob(*args, sum: bool = True, **kwargs) -> Optional[TensorVariable]:
    """Create a graph representing the joint log-probability/measure of a graph.

    This function calls `factorized_joint_logprob` and returns the combined
    log-probability factors as a single graph.

    Parameters
    ----------
    sum: bool
        If ``True`` each factor is collapsed to a scalar via ``sum`` before
        being joined with the remaining factors. This may be necessary to
        avoid incorrect broadcasting among independent factors.

    """
    logprob = factorized_joint_logprob(*args, **kwargs)
    if not logprob:
        return None
    elif len(logprob) == 1:
        logprob = tuple(logprob.values())[0]
        if sum:
            return at.sum(logprob)
        else:
            return logprob
    else:
        if sum:
            return at.sum([at.sum(factor) for factor in logprob.values()])
        else:
            return at.add(*logprob.values())
