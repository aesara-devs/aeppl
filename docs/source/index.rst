.. image:: logo.png
   :align: center


AePPL is a flexible collection of tools that allows users to define and manipulate probabilistic models implemented in `Aesara <https://github.com/aesara-devs/aesara>`__.

Features
========

AePPL strives to make probabilistic programming as simple as it should be, and as exhaustive as can be.

- **Intuitive:** All you need is an Aesara graph that contains random variables; no need to learn a new syntax
- **Exhaustive:**  If your problem is mathematically well-defined, AePPL is designed to support it.
- **A broad ecosystem:** Benefit from Aesara's current and future compilation backends (e.g. C, JAX, Numba). Use your model with other Aesara, Numba or JAX libraries
- **Flexible:** Define your own distributions by transforming random variables directly; condition on transformed random variables; use loops and conditionals in your model.
- **Extensible:** Aesara provides tools to easily traverse and transform probabilistic models

Example
=======

.. code::

   import aeppl
   import aesara
   import aesara.tensor as at

   srng = at.random.RandomStream()

   S_rv = srng.invgamma(0.5, 0.5)
   Y_rv = srng.normal(0.0, at.sqrt(S_rv))

   # Print a LateX representation of the model
   aeppl.latex_pprint(Y_rv)

   # Compute the joint log-probability for the mixture
   logprob, (s_vv, y_vv) = joint_logprob(S_rv, Y_rv)

   # Compile to C, Numba or JAX
   fn = aesara.function([s_vv, y_vv], logprob)
   numba_fn = aesara.function([s_vv, y_vv], logprob, mode="NUMBA")
   jax_fn = aesara.function([s_vv, y_vv], logprob, mode="JAX")

Install AePPL
=============

AePPL installation can happen in a few different ways. You can install AePPL with `conda` or with `pip`. To get the bleeding edge version you can install `aeppl-nightly.`

.. tab-set::

     .. tab-item:: PyPi

         .. code:: bash

             pip install aeppl


     .. tab-item:: Conda

         .. code:: bash

            conda install -c conda-forge aeppl


     .. tab-item:: Nightly

         .. code:: bash

             pip install aeppl-nightly

Getting Started
===============

- :doc:`Examples </examples>`
- :doc:`Supported random variables </api/distributions>`
- :doc:`Compute my model's log-density </api/logdensity>`

.. toctree::
  :hidden:

  Roadmap <https://github.com/orgs/aesara-devs/projects/2>
  examples
  api/index

.. toctree::
  :hidden:
  :caption: External links

  GitHub <https://github.com/aesara-devs/aeppl>
  Twitter <https://twitter.com/AesaraDevs>
