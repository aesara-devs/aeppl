import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import aesara.tensor as at
from aesara import config
from aesara.graph.basic import graph_inputs, io_toposort
from aesara.graph.op import compute_test_value
from aesara.graph.rewriting.basic import GraphRewriter, NodeRewriter
from aesara.tensor.var import TensorVariable

from aeppl.abstract import get_measurable_outputs
from aeppl.logprob import _logprob
from aeppl.rewriting import construct_ir_fgraph
from aeppl.utils import rvs_to_value_vars


def conditional_logprob(
    *random_variables: TensorVariable,
    realized: Dict[TensorVariable, TensorVariable] = {},
    warn_missing_rvs: bool = True,
    ir_rewriter: Optional[GraphRewriter] = None,
    extra_rewrites: Optional[Union[GraphRewriter, NodeRewriter]] = None,
    **kwargs,
) -> Tuple[Dict[TensorVariable, TensorVariable], List[TensorVariable]]:
    r"""Create a map between random variables and their conditional log-probabilities.

    The list of measurable variables implicitly defines a joint probability that
    factorizes according to the graphical model implemented by the Aesara model
    these variables belong to.

    For example, consider the following

    .. code-block:: python

        import aesara.tensor as at

        sigma2_rv = at.random.invgamma(0.5, 0.5, name="sigma2")
        Y_rv = at.random.normal(0, at.sqrt(sigma2_rv), name="Y")

    This graph for ``Y_rv`` is equivalent to the following hierarchical model:

    .. math::

        \sigma^2 \sim& \operatorname{InvGamma}(0.5, 0.5) \\
        Y \sim& \operatorname{N}(0, \sigma^2)

    The graph generated by ``conditional_logprob(Y_rv)`` corresponds to the
    conditional log-probability :math:`P(Y=y \mid \sigma^2)` where
    :math:`\sigma^2` is a stochastic variable. The graphs generated by
    ``conditional_logprob(sigma2_rv, Y_rv)`` represent :math:`\sigma^2` and
    :math:`Y`'s respective conditional log-probabilities, :math:`\log P(\sigma^2 = s)`
    and :math:`\log p(Y = y | \sigma^2 = s)`.

    To build the log-probability graphs, `conditional_logprob` must generate
    value variables associated with each input variable. They are returned along
    with the graphs:

    .. code-block:: python

        import aesara.tensor as at

        sigma2_rv = at.random.invgamma(0.5, 0.5)
        Y_rv = at.random.normal(0, at.sqrt(sigma2_rv))

        logprobs, value_variables = conditional_logprob(Y_rv, sigma2_rv)
        # print(logprobs.keys())
        # [Y, sigma2]
        # print(value_variables)
        # [Y_vv, sigma2_vv]


    Parameters
    ==========
    random_variables
        A ``list`` of  random variables for which we need to return a
        conditional log-probability graph.
    warn_missing_rvs
        When ``True``, issue a warning when a `RandomVariable` is found in
        the graph and doesn't have a corresponding value variable specified in
        `rv_values`.
    ir_rewriter
        Rewriter that produces the intermediate representation of Measurable Variables.
    extra_rewrites
        Extra rewrites to be applied (e.g. reparameterizations, transforms,
        etc.)

    Returns
    =======
    A ``dict`` that maps each random variable to the derived log-probability
    factor, and a list of the created valued variables in the same order as the
    order in which their corresponding random variables were passed as
    arguments.

    """

    # Create value variables by cloning the input measurable variables
    original_rv_values = {}
    for rv in random_variables:
        vv = rv.clone()
        if rv.name:
            vv.name = f"{rv.name}_vv"
        original_rv_values[rv] = vv

    # Value variables are not cloned when constructing the conditional log-probability
    # graphs. We can thus use them to recover the original random variables to index the
    # maps to the logprob graphs and value variables before returning them.
    rv_values = {**original_rv_values, **realized}
    vv_to_original_rvs = {vv: rv for rv, vv in rv_values.items()}

    fgraph, rv_values, _ = construct_ir_fgraph(rv_values, ir_rewriter=ir_rewriter)

    # The interface for transformations assumes that the value variables are in
    # the transformed space. To get the correct `shape` and `dtype` for the
    # value variables we return we need to apply the forward transformation to
    # our RV copies, and return the type of the resulting variable as a value
    # variable.
    vv_remapper = {}
    if extra_rewrites is not None:
        extra_rewrites.add_requirements(fgraph, {**original_rv_values, **realized})
        extra_rewrites.apply(fgraph)
        vv_remapper = fgraph.values_to_untransformed

    rv_remapper = fgraph.preserve_rv_mappings

    # This is the updated random-to-value-vars map with the lifted/rewritten
    # variables.  The rewrites are supposed to produce new
    # `MeasurableVariable`s that are amenable to `_logprob`.
    updated_rv_values = rv_remapper.rv_values

    # Some rewrites also transform the original value variables. This is the
    # updated map from the new value variables to the original ones, which
    # we want to use as the keys in the final dictionary output
    original_values = rv_remapper.original_values

    # When a `_logprob` has been produced for a `MeasurableVariable` node, all
    # other references to it need to be replaced with its value-variable all
    # throughout the `_logprob`-produced graphs.  The following `dict`
    # cumulatively maintains remappings for all the variables/nodes that needed
    # to be recreated after replacing `MeasurableVariable`s with their
    # value-variables.  Since these replacements work in topological order, all
    # the necessary value-variable replacements should be present for each
    # node.
    replacements = updated_rv_values.copy()

    # To avoid cloning the value variables, we map them to themselves in the
    # `replacements` `dict` (i.e. entries already existing in `replacements`
    # aren't cloned)
    replacements.update({v: v for v in rv_values.values()})

    # Walk the graph from its inputs to its outputs and construct the
    # log-probability
    q = deque(fgraph.toposort())

    logprob_vars = {}
    value_variables = {}

    while q:
        node = q.popleft()

        outputs = get_measurable_outputs(node.op, node)
        if not outputs:
            continue

        if any(o not in updated_rv_values for o in outputs):
            if warn_missing_rvs:
                warnings.warn(
                    "Found a random variable that is not assigned a value variable: "
                    f"{node.outputs}"
                )
            continue

        q_value_vars = [replacements[q_rv_var] for q_rv_var in outputs]

        if not q_value_vars:
            continue

        # Replace `RandomVariable`s in the inputs with value variables.
        # Also, store the results in the `replacements` map for the nodes
        # that follow.
        remapped_vars, _ = rvs_to_value_vars(
            q_value_vars + list(node.inputs),
            initial_replacements=replacements,
        )
        q_value_vars = remapped_vars[: len(q_value_vars)]
        q_rv_inputs = remapped_vars[len(q_value_vars) :]

        q_logprob_vars = _logprob(
            node.op,
            q_value_vars,
            *q_rv_inputs,
            **kwargs,
        )

        if not isinstance(q_logprob_vars, (list, tuple)):
            q_logprob_vars = [q_logprob_vars]

        for q_value_var, q_logprob_var in zip(q_value_vars, q_logprob_vars):

            q_value_var = original_values[q_value_var]
            q_rv = vv_to_original_rvs[q_value_var]

            if q_rv.name:
                q_logprob_var.name = f"{q_rv.name}_logprob"

            if q_rv in logprob_vars:
                raise ValueError(
                    f"More than one logprob factor was assigned to the random variable {q_rv}"
                )

            logprob_vars[q_rv] = q_logprob_var

            q_value_var = vv_remapper.get(q_value_var, q_value_var)
            value_variables[q_rv] = q_value_var

        # Recompute test values for the changes introduced by the
        # replacements above.
        if config.compute_test_value != "off":
            for node in io_toposort(graph_inputs(q_logprob_vars), q_logprob_vars):
                compute_test_value(node)

    missing_value_terms = set(vv_to_original_rvs.values()) - set(logprob_vars.keys())
    if missing_value_terms:
        raise RuntimeError(
            f"The logprob terms of the following random variables could not be derived: {missing_value_terms}"
        )

    return logprob_vars, [value_variables[rv] for rv in original_rv_values.keys()]


def joint_logprob(
    *random_variables: List[TensorVariable],
    realized: Dict[TensorVariable, TensorVariable] = {},
    **kwargs,
) -> Optional[Tuple[TensorVariable, List[TensorVariable]]]:
    """Create a graph representing the joint log-probability/measure of a graph.

    This function calls `factorized_joint_logprob` and returns the combined
    log-probability factors as a single graph.

    Parameters
    ==========
    random_variables
        A ``list`` of  random variables for which we need to return a
        conditional log-probability graph.
    realized
        A ``dict`` that maps  random variables to their realized value.

    Returns
    =======
    A ``TensorVariable`` that represents the joint log-probability of the graph
    implicitly defined by the random variables passed as arguments, and a list
    of the created valued variables in the same order as the order in which
    their corresponding random variables were passed as arguments.

    """
    logprob, value_variables = conditional_logprob(
        *random_variables, realized=realized, **kwargs
    )
    if not logprob:
        return None
    elif len(logprob) == 1:
        cond_logprob = tuple(logprob.values())[0]
        return at.sum(cond_logprob), value_variables
    else:
        joint_logprob: TensorVariable = at.sum(
            [at.sum(factor) for factor in logprob.values()]
        )
        return joint_logprob, value_variables
