========
Mixtures
========

There are two ways to define mixtures in AePPL with Aesara constructs.

By creating an array of random variables and indexing it with a random variable:

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream(0)

   w_rv = srng.normal(-2, 1)
   x_rv = srng.normal(0, 1)
   y_rv = srng.normal(2, 1)
   mix_rv = at.stack([w_rv, x_rv, y_rv])

   p = at.vector('p')
   i_rv = srng.categorical(p)
   Z_rv = mix_rv[i_rv]

Using `aesara.tensor.switch`:

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream(0)

   x_rv = srng.normal(0, 1)
   y_rv = srng.normal(2, 1)

   p = at.scalar('p')
   i_rv = srng.bernoulli(p)
   Z_rv = at.switch(i_rv, x_rv, y_rv)


Rewrites
--------

The following rewrites identify mixtures in Aesara graphs and replace them with :class:`MixtureRV`\s, which are then used to compute the model's log-density.

.. autofunction:: aeppl.mixture.mixture_replace
.. autofunction:: aeppl.mixture.switch_mixture_replace

Log-density
-----------

.. autoclass:: aeppl.mixture.logprob_MixtureRV
