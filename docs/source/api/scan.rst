====
Scan
====

AePPL can compute the log-density of measures defined as the output of an `aesara.scan` loop:

.. code::

   import aesara
   import aesara.tensor as at
   import numpy as np

   srng = at.random.RandomStream(0)

   p = np.array([0.9, 0.1])
   S_0_rv = srng.categorical(p)

   Gamma_at = at.matrix("Gamma")

   def step_fn(S_tm1, Gamma):
      S_t = srng.categorical(Gamma[S_tm1])
      return S_t

    S_T_rv, _ = aesara.scan(
        fn=step_fn,
        outputs_info=[S_0_rv],
        sequences=[Gamm_at],
        n_steps=10,
    )

    logprob, (s_T_vv, s_0_vv) = aeppl.joint_logprob(S_T_rv, S_0_rv)

Rewrite
-------

.. autofunction:: aeppl.scan.find_measurable_scans

Log-density
-----------

.. autofunction:: aeppl.scan.logprob_ScanRV
