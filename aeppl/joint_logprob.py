from typing import Dict, List, Optional, Union

import aesara.tensor as at
from aesara.graph.basic import graph_inputs, io_toposort
from aesara.graph.fg import FunctionGraph
from aesara.graph.opt import GlobalOptimizer, LocalOptimizer
from aesara.graph.optdb import OptimizationQuery
from aesara.tensor.basic_opt import ShapeFeature
from aesara.tensor.var import TensorVariable

from aeppl.abstract import ValuedVariable, get_measurable_outputs, valued_variable
from aeppl.logprob import _logprob
from aeppl.opt import logprob_rewrites_db


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
    for rv, val in rv_values.items():
        rv_node_clone = rv.owner.clone()
        rv_clone = rv_node_clone.outputs[rv.owner.outputs.index(rv)]
        rv_value_clones[rv_clone] = val

    # We topologically order the `FunctionGraph` outputs just in case
    # we want to perform operations on them in order (e.g. compute test values).
    # TODO: It's not clear that this is actually needed.
    outputs = []
    rv_vars = list(rv_value_clones.keys())
    sorted_rv_vars: List[TensorVariable] = sum(
        [
            [o for o in node.outputs if o in rv_vars]
            for node in io_toposort(graph_inputs(rv_vars), rv_vars)
        ],
        [],
    )
    measured_outputs = [
        valued_variable(rv, rv_value_clones[rv]) for rv in sorted_rv_vars
    ]

    # Since we're going to clone the entire graph, we need to keep a map from
    # the old nodes to the new ones; otherwise, we won't be able to use
    # `rv_values`.
    # We start the `dict` with mappings from the value variables to themselves,
    # to prevent them from being cloned.
    memo = {v: v for v in rv_value_clones.values()}

    # We add `ShapeFeature` because it will get rid of references to the old
    # `RandomVariable`s that have been lifted; otherwise, it will be difficult
    # to give good warnings when an unaccounted for `RandomVariable` is
    # encountered
    fgraph = FunctionGraph(
        outputs=measured_outputs,
        clone=True,
        memo=memo,
        copy_orphans=False,
        copy_inputs=False,
        features=[ShapeFeature()],
    )

    # Update `rv_values` so that it uses the new cloned variables
    rv_value_clones = {memo[k]: v for k, v in rv_value_clones.items()}

    # Replace valued non-output variables with their values
    fgraph.replace_all(
        [(memo[rv], val) for rv, val in rv_values.items() if rv in memo],
        import_missing=True,
    )

    logprob_rewrites_db.query(OptimizationQuery(include=["basic"])).optimize(fgraph)

    if extra_rewrites is not None:
        extra_rewrites.optimize(fgraph)

    logprob_vars = {}

    for out, orig_rv in zip(fgraph.outputs, sorted_rv_vars):
        node = out.owner

        assert isinstance(node.op, ValuedVariable)

        rv_var, val_var = node.inputs

        rv_node = rv_var.owner
        outputs = get_measurable_outputs(rv_node.op, rv_node)

        if not outputs:
            raise ValueError(f"Couldn't derive a log-probability for {out}")

        rv_logprob = _logprob(
            rv_node.op,
            [val_var],
            *rv_node.inputs,
            **kwargs,
        )

        if isinstance(rv_logprob, (tuple, list)):
            (rv_logprob,) = rv_logprob

        if orig_rv.name:
            rv_logprob.name = f"{orig_rv.name}_logprob"

        logprob_vars[orig_rv] = rv_logprob

        # # Recompute test values for the changes introduced by the
        # # replacements above.
        # if config.compute_test_value != "off":
        #     for node in io_toposort(graph_inputs([rv_logprob]), q_logprob_vars):
        #         compute_test_value(node)

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
