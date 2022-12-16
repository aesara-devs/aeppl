================
Random variables
================

Most random variables are implemented in Aesara, and AePPL only derives their log-densities. In the following we provide examples of the random variables supported by AePPL and how their log-densities can be obtained.

The :py:func:`aeppl.logprob.logprob` function can be called on any random variable instance to create a graph that represents its log-density computed at a given value:

.. code::

   import aesara.tensor as at
   from aeppl.logprob import _logprob

   srng = at.random.RandomStream()

   mu = at.scalar("mu")
   sigma = at.scalar("sigma")
   x_rv = snrg.bernoulli(mu, sigma)

   x_val = at.scalar('x_val')
   x_logprob = logprob(x_rv, x_val)

Bernoulli
---------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.BernoulliRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   p = at.scalar("p")
   x_rv = snrg.bernoulli(p)

Beta
----

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.BetaRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   a = at.scalar("a")
   b = at.scalar("b")
   x_rv = snrg.beta(a, b)


Beta-binomial
--------------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.BetaBinomialRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   n = at.iscalar("n")
   a = at.scalar("a")
   b = at.scalar("b")
   x_rv = snrg.betabinom(n, a, b)


Binomial
--------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.BinomialRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   n = at.iscalar("n")
   p = at.scalar("p")
   x_rv = snrg.binomal(n, p)


Cauchy
------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.CauchyRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   loc = at.scalar("loc")
   scale = at.scalar("scale")
   x_rv = snrg.cauchy(loc, scale)

Categorical
-----------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.CategoricalRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   p = at.vector("p")
   x_rv = snrg.categorical(p)

Chi-squared
-----------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.ChiSquareRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   df = at.scalar("df")
   x_rv = snrg.chisquare(df)

Dirac
-----

The Dirac measure is defined in AePPL.

.. code::

   import aeppl.dists as ad

   loc = at.scalar("loc")
   x_rv = ad.dirac_delta(loc)


Dirichlet
---------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.DirichletRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   alpha = at.vector("alpha")
   x_rv = snrg.dirichlet(alpha)

Discrete Markov Chain
---------------------

.. autofunction:: aeppl.dists.discrete_markov_chain

Exponential
-----------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.ExponentialRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   beta = at.scalar("beta")
   x_rv = snrg.exponential(beta)

Gamma
-----

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.GammaRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   alpha = at.scalar('alpha')
   beta = at.scalar('beta')
   x_rv = srng.gamma(alpha, beta)

Geometric
---------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.GeometricRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   p = at.scalar("p")
   x_rv = snrg.geometric(p)

Gumbel
------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.GumbelRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   mu = at.scalar('mu')
   beta = at.scalar('beta')
   x_rv = srng.gumbel(mu, beta)

Half-Cauchy
-----------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.HalfCauchyRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   x0 = at.scalar('x0')
   gamma = at.scalar('gamma')
   x_rv = srng.halfcauchy(x0, gamma)

Half-Normal
-----------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.HalfNormalRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   mu = at.scalar('mu')
   sigma = at.scalar('sigma')
   x_rv = srng.halfnormal(mu, sigma)

Hypergeometric
--------------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.HyperGeometricRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   ngood = at.scalar("ngood")
   nbad = at.scalar("nbad")
   nsample = at.scalar("nsample")
   x_rv = snrg.hypergeometric(ngood, nbad, nsample)

Inverse-Gamma
-------------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.InvGammaRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   alpha = at.scalar('alpha')
   beta = at.scalar('beta')
   x_rv = srng.invgamma(alpha, beta)

Laplace
-------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.LaplaceRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   mu = at.scalar("mu")
   lmbda = at.scalar("lambda")
   x_rv = snrg.laplace(mu, lmbda)

Logistic
--------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.LogisticRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   mu = at.scalar("mu")
   s = at.scalar("s")
   x_rv = snrg.logistic(mu, s)

Lognormal
---------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.LogNormalRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   mu = at.scalar("mu")
   sigma = at.scalar("sigma")
   x_rv = snrg.lognormal(mu, sigma)

Multinomial
-----------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.MultinomialRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   n = at.iscalar("n")
   p = at.vector("p")
   x_rv = snrg.multinomial(n, p)

Multivariate-Normal
-------------------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.MvNormalRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   mu = at.vector('mu')
   Sigma = at.matrix('sigma')
   x_rv = srng.multivariate_normal(mu, Sigma)


Negative-Binomial
-----------------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.NegBinomialRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   n = at.iscalar("n")
   p = at.scalar("p")
   x_rv = snrg.negative_binomial(n, p)

Normal
------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.NormalRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   mu = at.scalar('mu')
   sigma = at.scalar('sigma')
   x_rv = srng.normal(mu, sigma)

Pareto
------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.ParetoRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   b = at.scalar("b")
   scale = at.scalar("scale")
   x_rv = snrg.pareto(b, scale)

Poisson
-------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.PoissonRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   lmbda = at.scalar("lambda")
   x_rv = snrg.poisson(lmbda)

Student T
---------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.StudentTRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   df = at.scalar('df')
   loc = at.scalar('loc')
   scale = at.scalar('scale')
   x_rv = srng.t(df, loc, scale)

Triangular
----------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.TriangularRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   left = at.scalar('left')
   mode = at.scalar('mode')
   right = at.scalar('right')
   x_rv = srng.triangular(left, mode, right)

Uniform
-------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.UniformRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   low = at.scalar('low')
   high = at.scalar('high')
   x_rv = srng.uniform(low, high)

Von Mises
---------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.VonMisesRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   mu = at.scalar('mu')
   kappa = at.scalar('kappa')
   x_rv = srng.vonmises(mu, kappa)

Wald
----

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.WaldRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   mu = at.scalar('mu')
   lmbda = at.scalar('lambda')
   x_rv = srng.wald(mu, lmbda)


Weibull
-------

Documentation for the Aesara implementation can be found here: :external:py:class:`aesara.tensor.random.basic.WeibullRV`

.. code::

   import aesara.tensor as at

   srng = at.random.RandomStream()

   k = at.scalar('k')
   x_rv = srng.weibull(k)
