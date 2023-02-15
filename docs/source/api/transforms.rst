==========
Transforms
==========

------------
Introduction
------------

Random variables in an Aesara model graph can be transformed by AePPL in multiple ways.

One way is by *explicitly* constructing a transform using Aesara `Op`\s within a model graph:

.. code::

    A_rv = at.random.normal(0, 1.0, name="A")
    Z_rv = at.exp(A_rv)
    Z_rv.name = "Z"

    logp, (z_vv,) = joint_logprob(Z_rv)


The resulting log-density should reflect the "change-of-variables" implied by taking the exponential of
a normal random variable:

    >>> print(aesara.pprint(logp))
    ((-0.9189385332046727 + (-0.5 * (log(Z_vv) ** 2))) + (-1.0 * log(Z_vv)))


Transforms can also be applied *implicitly* through the use of `TransformValuesRewrite`:

.. code::

    A_rv = at.random.normal(0, 1.0, name="A")
    B_rv = at.random.normal(A_rv, 1.0, name="B")

    logp, (a_vv, b_vv,) = joint_logprob(
        A_rv,
        B_rv,
        extra_rewrites=TransformValuesRewrite(
            {A_rv: ExpTransform()}
        ),
    )


This approach will apply the designated transforms to all occurrences of their
associated variables and their values:

    >>> print(aesara.pprint(logp))
    sum([((-0.9189385332046727 + (-0.5 * (log(A_vv-trans) ** 2))) + (-1.0 * log(A_vv-trans))),
         (-0.9189385332046727 + (-0.5 * ((B_vv - log(A_vv-trans)) ** 2)))], axis=None)


------------------------
`TransformValuesRewrite`
------------------------

.. autoclass:: aeppl.transforms.TransformValuesRewrite


---------------------
Invertible transforms
---------------------

AePPL currently supports transforms using the following (invertible) Aesara operators. This means that AePPL can compute the log-probability of a random variable that is the result of one of the following transformations applied to another random variable:

.. print-invertible-transforms::

---------
Censoring
---------

AePPL can compute the log-density of `aesara.tensor.clip` applied to a random variable:

.. autofunction:: aeppl.censoring.clip_logprob

--------
Rounding
--------

AePPL can compute the log-density of `aesara.tensor.ceil`, `aesara.tensor.floor`, or `aesara.tensor.round_half_to_even` applied to a random variable:

.. autofunction:: aeppl.censoring.round_logprob
