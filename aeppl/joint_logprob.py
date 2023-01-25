import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import aesara.tensor as at
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import GraphRewriter, NodeRewriter
from aesara.tensor.var import TensorVariable

from aeppl.abstract import ValuedVariable, get_measurable_outputs
from aeppl.logprob import _logprob
from aeppl.rewriting import construct_ir_fgraph, ir_cleanup_db

if TYPE_CHECKING:
    from aesara.graph.basic import Apply, Variable


class DensityNotFound(Exception):
    """An exception raised when a density cannot be found."""


def conditional_logprob(
    *random_variables: TensorVariable,
    realized: Dict[TensorVariable, TensorVariable] = {},
    ir_rewriter: Optional[GraphRewriter] = None,
    extra_rewrites: Optional[Union[GraphRewriter, NodeRewriter]] = None,
    **kwargs,
) -> Tuple[Dict[TensorVariable, TensorVariable], Tuple[TensorVariable, ...]]:
    r"""Create a map between random variables and their conditional log-probabilities.

    Consider the following Aesara model:

    .. code-block:: python

        import aesara.tensor as at

        srng = at.random.RandomStream()

        sigma2_rv = srng.invgamma(0.5, 0.5)
        Y_rv = srng.normal(0, at.sqrt(sigma2_rv))

    Which represents the following mathematical model:

    .. math::

        \sigma^2 \sim& \operatorname{InvGamma}(0.5, 0.5) \\
        Y \sim& \operatorname{N}\left(0, \sigma^2\right)


    We can generate the graph that computes the conditional log-density associated
    with each random variable with:

    .. code-block:: python

        import aeppl

        logprobs, (sigma2_vv, Y_vv) = aeppl.conditional_logprob(sigma2_rv, Y_rv)

    The list of random variables passed to `conditional_logprob` implicitly
    defines a joint density that factorized according to the graphical model
    represented by the Aesara model. Here, ``logprobs[sigma2_rv]`` corresponds
    to the conditional log-density :math:`\operatorname{P}\left(sigma^2=s \mid Y\right)` and
    ``logprobs[Y_rv]`` to :math:`\operatorname{P}\left(Y=y \mid \sigma^2\right)`.

    To build the log-density graphs, `conditional_logprob` must generate the
    value variable associated with each random variable. They are returned along
    with the graph in the same order as the random variables were passed to
    `conditional_logprob`. Here, the value variables ``sigma2_vv`` and ``Y_vv``
    correspond to :math:`s` and :math:`y` in the previous expressions
    respectively.

    It is also possible to call `conditional_logprob` omitting some of the
    random variables in the graph:

    .. code-block:: python

        logprobs, (Y_vv,) = aeppl.conditional_logprob(Y_rv)

    In this case, ``logprobs[Y_rv]`` corresponds to the conditional log-density
    :math:`\operatorname{P}\left(Y=y \mid \sigma^2\right)` where
    :math:`\sigma^2` is a stochastic variable.

    Another important case is when one the variables is already realized. For
    instance, if `Y_rv` is observed we can include its realized value directly
    in the log-density graphs:

    .. code-block:: python

        y_obs = Y_rv.copy()
        logprobs, (sigma2_vv,) = aeppl.conditional_logprob(sigma2_rv, realized={Y_rv: y_obs})

    In this case, `conditional_logprob` uses the value variable passed in the
    conditional log-density graphs it produces.

    Parameters
    ==========
    random_variables
        A ``list`` of  random variables for which we need to return a
        conditional log-probability graph.
    realized
        A ``dict`` that maps realized random variables to their realized
        values. These values used in the generated conditional
        log-density graphs.
    ir_rewriter
        Rewriter that produces the intermediate representation of measurable
        variables.
    extra_rewrites
        Extra rewrites to be applied (e.g. reparameterizations, transforms,
        etc.)

    Returns
    =======
    conditional_logprobs
        A ``dict`` that maps each random variable to the graph that computes their
        conditional log-density implicitly defined by the random variables passed
        as arguments. ``None`` if a log-density cannot be computed.
    value_variables
        A ``list`` of the created valued variables in the same order as the
        order in which their corresponding random variables were passed as
        arguments. Empty if `random_variables` is empty.

    """

    deprecated_option = kwargs.pop("warn_missing_rvs", None)

    if deprecated_option:
        warnings.warn(
            "The `warn_missing_rvs` option is deprecated and has been removed.",
            DeprecationWarning,
        )

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

    fgraph, _, memo = construct_ir_fgraph(rv_values, ir_rewriter=ir_rewriter)

    if extra_rewrites is not None:
        extra_rewrites.add_requirements(fgraph, rv_values, memo)
        extra_rewrites.apply(fgraph)

    # We assign log-densities on a per-node basis, and not per-output/variable.
    realized_vars = set()
    new_to_old_rvs = {}
    nodes_to_vals: Dict["Apply", List[Tuple["Variable", "Variable"]]] = {}

    for bnd_var, (old_mvar, old_val) in zip(fgraph.outputs, rv_values.items()):
        mnode = bnd_var.owner
        assert mnode and isinstance(mnode.op, ValuedVariable)

        rv_var, val_var = mnode.inputs
        rv_node = rv_var.owner

        if rv_node is None:
            raise DensityNotFound(f"Couldn't derive a log-probability for {rv_var}")

        if old_mvar in realized:
            realized_vars.add(rv_var)

        # Do this just in case a value variable was changed.  (Some transforms
        # do this.)
        new_val = memo[old_val]

        nodes_to_vals.setdefault(rv_node, []).append((val_var, new_val))

        new_to_old_rvs[rv_var] = old_mvar

    value_vars: Tuple["Variable", ...] = ()
    logprob_vars = {}

    for rv_node, rv_val_pairs in nodes_to_vals.items():

        outputs = get_measurable_outputs(rv_node.op, rv_node)

        if not outputs:
            raise DensityNotFound(f"Couldn't derive a log-probability for {rv_node}")

        if len(outputs) < len(rv_val_pairs):
            raise ValueError(
                f"Too many values ({rv_val_pairs}) bound to node {rv_node}."
            )

        assert len(outputs) == len(rv_val_pairs)

        rv_vals, rv_base_vals = zip(*rv_val_pairs)

        rv_logprobs = _logprob(
            rv_node.op,
            rv_vals,
            *rv_node.inputs,
            **kwargs,
        )

        if not isinstance(rv_logprobs, (tuple, list)):
            rv_logprobs = (rv_logprobs,)

        for lp_out, rv_out, rv_base_val in zip(rv_logprobs, outputs, rv_base_vals):
            old_mvar = new_to_old_rvs[rv_out]

            if old_mvar.name:
                lp_out.name = f"{rv_out.name}_logprob"

            logprob_vars[old_mvar] = lp_out

            if rv_out not in realized_vars:
                value_vars += (rv_base_val,)

        # # Recompute test values for the changes introduced by the
        # # replacements above.
        # if config.compute_test_value != "off":
        #     for node in io_toposort(graph_inputs([rv_logprobs]), outputs):
        #         compute_test_value(node)

    # Remove unneeded IR elements from the graph
    rv_logprobs_fg = FunctionGraph(outputs=tuple(logprob_vars.values()), clone=False)
    ir_cleanup_db.query("+basic").rewrite(rv_logprobs_fg)

    return logprob_vars, value_vars


def joint_logprob(
    *random_variables: List[TensorVariable],
    realized: Dict[TensorVariable, TensorVariable] = {},
    **kwargs,
) -> Optional[Tuple[TensorVariable, Tuple[TensorVariable, ...]]]:
    r"""Build the graph of the joint log-density of an Aesara graph.

    Consider the following Aesara model:

    .. code-block:: python

        import aesara.tensor as at

        srng = at.random.RandomStream()
        sigma2_rv = srng.invgamma(0.5, 0.5)
        Y_rv = srng.normal(0, at.sqrt(sigma2_rv))

    Which represents the following mathematical model:

    .. math::

        \sigma^2 \sim& \operatorname{InvGamma}(0.5, 0.5) \\
        Y \sim& \operatorname{N}\left(0, \sigma^2\right)

    We can generate the graph that computes the joint log-density associated
    with this model:

    .. code-block:: python

        import aeppl

        logprob, (sigma2_vv, Y_vv) = aeppl.joint_logprob(sigma2_rv, Y_rv)

    To build the joint log-density graph, `joint_logprob` must generate the
    value variable associated with each random variable. They are returned along
    with the graph in the same order as the random variables were passed to
    `joint_logprob`. Here, the value variables ``sigma2_vv`` and ``Y_vv``
    correspond to the values :math:`s` and :math:`y` taken by :math:`\sigma^2`
    and :math:`Y`, respectively.

    It is also possible to call `joint_logprob` omitting some of the random
    variables in the graph:

    .. code-block:: python

        logprob, (Y_vv,) = aeppl.joint_logprob(Y_rv)

    In this case, ``logprob`` corresponds to the joint log-density
    :math:`\operatorname{P}\left(Y, \sigma^2\right)` where :math:`\sigma^2` is
    a stochastic variable.

    Another important case is when one of the variables is already realized. For
    instance, if `Y_rv` is observed we can include its realized value directly
    in the log-density graphs:

    .. code-block:: python

        y_obs = Y_rv.copy()
        logprob, (sigma2_vv,) = aeppl.joint_logprob(sigma2_rv, realized={Y_rv: y_obs})

    In this case, `joint_logprob` uses the value ``y_obs`` mapped to ``Y_rv`` in the
    conditional log-density graphs it produces, so that ``logprob`` corresponds
    to the density :math:`\operatorname{P}\left(\sigma^2 \mid Y=y\right)` when
    :math:`y` and :math:`Y` correspond to ``y_obs`` and ``Y_rv``, respectively.

    Parameters
    ==========
    random_variables
        A ``list`` of  random variables for which we need to return a
        conditional log-probability graph.
    realized
        A ``dict`` that maps  random variables to their realized value.

    Returns
    =======
    logprob
        A ``TensorVariable`` that represents the joint log-probability of the graph
        implicitly defined by the random variables passed as arguments. ``None`` if
        a log-density cannot be computed.
    value_variables
        A ``list`` of the created valued variables in the same order as the
        order in which their corresponding random variables were passed as
        arguments. Empty if ``random_variables`` is empty.

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
