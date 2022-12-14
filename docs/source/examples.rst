========
Examples
========

On this page you will find self-contained examples that demonstrate some of the capabilities of AePPL.

Mixture models
==============

You can define mixture models very naturally by specifying their generative process as an Aesara model:

.. code::

    import aeppl
    import aesara
    import aesara.tensor as at
    import numpy as np

    srng = at.random.RandomStream(0)

    loc = np.array([-2, 0, 3.2, 2.5])
    scale = np.array([1.2, 1, 5, 2.8])
    weights = np.array([0.2, 0.3, 0.1, 0.4])

    N_rv = srng.normal(loc, scale, name="N")
    I_rv = srng.categorical(weights, name="I")
    Y_rv = N_rv[I_rv]

    logprob, (y_vv, i_vv) = aeppl.joint_logprob(Y_rv, I_rv)

    logprob_fn = aesara.function([i_vv, y_vv], logprob)
    print(logprob_fn(0, -2))
    # -2.7106980024327276
    print(logprob_fn(0, 3))
    # -11.391253557988284


Stochastic volatility model
===========================

.. code::

    import aeppl
    import aesara
    import aesara.tensor as at

    srng = at.random.RandomStream(0)
    sigma_rv = srng.exponential(1)
    nu_rv = srng.exponential(1)

    length = at.iscalar("length")
    v_rv = at.cumsum(srng.normal(0, sigma_rv**-1, size=(length,)))

    R_rv = srng.t(nu_rv, 0., at.exp(2 * v_rv))

    logprob, (nu_vv, sigma_vv, v_vv, R_vv) = aeppl.joint_logprob(nu_rv, sigma_rv, v_rv, R_rv)

    # Sample from the prior predictive distribution
    sample_fn = aesara.function([length], [nu_rv, sigma_rv, v_rv, R_rv])
    sample = sample_fn(3)
    print(sample)
    # [array(0.80310441),
    #  array(3.29352779),
    #  array([0.28604556, 0.07396521, 0.55627653]),
    #  array([-2.68456399, 11.52348553,  1.6287702 ])]

    # Compute the log-density at this point
    logprob_fn = aesara.function([nu_vv, sigma_vv, v_vv, R_vv], logprob)
    print(logprob_fn(*sample))
    # -16.4683922384827


Rat tumor model
===============

This example shows how to deal with improper priors and specify observed variables in AePPL:

.. code::

    import aeppl
    import aesara.tensor as at

    n_rats = at.iscalar("num_rats")
    y_obs = at.vector("observations")

    a = at.scalar("a")
    b = at.scalar("b")
    hyperprior = at.pow(a + b, -2.5)

    srng = at.random.RandomStream(0)
    theta_rv = srng.beta(a, b, size=(n_trials,))
    Y_rv = srng.binom(theta, n_rats)


    logprob, (theta_vv,) = aeppl.joint_logprob(theta_rv, realized={Y_rv: y_obs})
    total_logprob = prior + logprob


Sparse Regression
=================

.. code::

    import aeppl
    import aesara
    import aesara.tensor as at
    import numpy as np


    srng = at.random.RandomStream(0)

    X = at.matrix("X")

    # Horseshoe `beta_rv`
    tau_rv = srng.halfcauchy(0, 1, name="tau")
    lmbda_rv = srng.halfcauchy(0, 1, size=X.shape[1], name="lambda")
    beta_rv = srng.normal(0, lmbda_rv * tau_rv, size=X.shape[1], name="beta")

    a = at.scalar("a")
    b = at.scalar("b")
    h_rv = srng.gamma(a, b, name="h")

    # Negative-binomial regression
    eta = X @ beta_rv
    p = at.sigmoid(-eta)
    Y_rv = srng.nbinom(h_rv, p, name="Y")

    to_sample_rvs = [tau_rv, lmbda_rv, beta_rv, h_rv, Y_rv]
    logprob, value_variables = aeppl.joint_logprob(*to_sample_rvs)

    # Sample from the prior predictive distribution
    sample_fn = aesara.function([a, b, X], to_sample_rvs)
    sample = sample_fn(1., 1., np.ones((2, 2)))
    print(sample)
    # [array(11.12665139),
    #  array([1.80017179, 0.40136517]),
    #  array([18.87013369, -3.11936304]),
    #  array(1.44847934),
    #  array([ 9149554, 13446053])]

    # Compile the joint log-density function
    logprob_fn = aesara.function([a, b, X] + list(value_variables), logprob)
    print(logprob_fn(1., 1., np.ones((2, 2)), *sample))
    # -50.34214668084496


Discrete HMM
============

AePPL allows one to condition on random variables that are generated inside a loop, which means discrete Hidden Markov Models can be expressed more naturally:

.. code::

    import aeppl
    import aesara
    import aesara.tensor as at

    srng = at.random.RandomStream()

    N_tt = at.iscalar("N")
    M_tt = at.iscalar("M")
    mus_tt = at.matrix("mus_t")

    sigmas_tt = at.ones((N_tt,))
    Gamma_rv = srng.dirichlet(at.ones((M_tt, M_tt)), name="Gamma")

    def scan_fn(mus_t, sigma_t, Gamma_t):
        S_t = srng.categorical(Gamma_t[0], name="S_t")
        Y_t = srng.normal(mus_t[S_t], sigma_t, name="Y_t")
        return Y_t, S_t

    (Y_rv, S_rv), _ = aesara.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv],
        outputs_info=[{}, {}],
        strict=True,
        name="scan_rv",
    )

    logprob, value_variables = aeppl.joint_logprob(Gamma_rv, Y_rv, S_rv)


The PERT distribution
=====================

Aesara supports many basic :doc:`random variables <api/distributions>` out of the box, and it allows one to express even more distributions as transformations of basic ones.

The `PERT distribution <https://en.wikipedia.org/wiki/PERT_distribution>`_, for instance, is a transformation of the Beta distribution, and, with AePPL, we can construct a PERT-distributed random variable by explicitly transforming a Beta:

.. code::

    import aeppl
    import aesara
    import aesara.tensor as at

    srng = at.random.RandomStream(0)

    def pert(srng, a, b, c):
        r"""Construct a random variable that is PERT-distributed."""
        alpha = 1 + 4 * (b - a) / (c - a)
        beta = 1 + 4 * (c - b) / (c - a)

        X_rv = srng.beta(alpha, beta)

        z = a + (b - a) * X_rv

        return z

    A_rv = srng.uniform(10, 20, name="A")
    B_rv = srng.uniform(20, 65, name="B")
    C_rv = srng.uniform(65, 100, name="C")
    Y_rv = pert(srng, A_rv, B_rv, C_rv)

    logprob, (y_vv, a_vv, b_vv, c_vv) = aeppl.joint_logprob(Y_rv, A_rv, B_rv, C_rv)

    # Compile a function that samples from the prior predictive distribution
    sample_fn = aesara.function([], [Y_rv, A_rv, B_rv, C_rv])
    sample = sample_fn()
    print(sample)
    # [array(25.51948424), array(19.42937553), array(50.47385856), array(94.33949018)]

    # Compile the joint log-density function
    logprob_fn = aesara.function([y_vv, a_vv, b_vv, c_vv], logprob)
    print(logprob_fn(*sample))
    # -12.956702290497232
