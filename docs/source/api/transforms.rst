==========
Transforms
==========


---------------------
Invertible transforms
---------------------

AePPL currently supports the following (invertible) Aesara operators. This means that AePPL can compute the log-density of a random variable that is the result of one of the following transformations applied to another random variable:

- `aesara.tensor.add`
- `aesara.tensor.sub`
- `aesara.tensor.mul`
- `aesara.tensor.true_div`
- `aesara.tensor.exponential`
- `aesara.tensor.exp`
- `aesara.tensor.log`

One can also apply the following transforms directly:

.. autoclass:: aeppl.transforms.LocTransform
.. autoclass:: aeppl.transforms.ScaleTransform
.. autoclass:: aeppl.transforms.LogTransform
.. autoclass:: aeppl.transforms.ExpTransform
.. autoclass:: aeppl.transforms.ReciprocalTransform
.. autoclass:: aeppl.transforms.IntervalTransform
.. autoclass:: aeppl.transforms.LogOddsTransform
.. autoclass:: aeppl.transforms.SimplexTransform
.. autoclass:: aeppl.transforms.CircularTransform

These transformations can be chained using:


.. autoclass:: aeppl.transforms.ChainedTransform

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
