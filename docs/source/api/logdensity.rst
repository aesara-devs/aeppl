=================
Model log-density
=================

.. currentmodule:: aeppl
.. autosummary::
   :nosignatures:

   joint_logprob
   conditional_logprob

AePPL produces log-density graphs from Aesara graphs containing random variables (i.e. *model graphs*).  The function `aeppl.joint_logprob` is the main entry-point for this functionality.

Not all Aesara model graphs are currently supported, but AePPL takes a very generalizable approach to producing log-density graphs, so support for mathematically well-defined models via basic Aesara operations is expected.  If you find a model graph that isn't supported, feel free to create a `Discussion <https://github.com/aesara-devs/aeppl/discussions>`_ or `issue <https://github.com/aesara-devs/aeppl/issues>`_.

A list of supported random variables can be found in :doc:`/api/distributions` and the list of supported operators in :doc:`/api/transforms` and :doc:`/api/scan`.

In some applications (like Gibbs sampling) the graphs that represent each individual conditional log-density may be needed; AePPL can generate such graphs via `aeppl.conditional_logprob`.

Joint log-density
------------------

.. autofunction:: aeppl.joint_logprob.joint_logprob

Conditional log-densities
-------------------------

.. autofunction:: aeppl.joint_logprob.conditional_logprob
