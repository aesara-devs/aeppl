<div align="center">

# AePPL

[![Pypi][pypi-badge]][pypi]
[![Downloads][downloads-badge]][releases]
[![Contributors][contributors-badge]][contributors]
 </br>
[![Gitter][gitter-badge]][gitter]
[![Discord][discord-badge]][discord]
[![Twitter][twitter-badge]][twitter]

Aeppl provides tools for a[e]PPL written in [Aesara](https://github.com/aesara-devs/aesara).

*Build arbitrarily complex probabilistic models. If it is mathematically defined, AePPL will support it.*

[Features](#features) •
[Get started](#get-started) •
[Install](#install) •
[Get help](#get-help) •
[Contribute](#contribute)

</div>


## Features

- Convert graphs containing Aesara `RandomVariable`s into joint
  log-probability graphs
- Transforms for `RandomVariable`s that map constrained support spaces to
  unconstrained spaces (e.g. the extended real numbers), and a rewrite that
  automatically applies these transformations throughout a graph
- Tools for traversing and transforming graphs containing `RandomVariable`s
- `RandomVariable`-aware pretty printing and LaTeX output


## Get started

Using `aeppl`, one can create a joint log-density graph from a graph
containing Aesara `RandomVariable`s:

``` python
import aesara
from aesara import tensor as at

from aeppl import joint_logprob, pprint

srng = at.random.RandomStream()

# A simple scale mixture model
S_rv = srng.invgamma(0.5, 0.5)
Y_rv = srng.normal(0.0, at.sqrt(S_rv))

# Compute the joint log-probability
logprob, (y, s) = joint_logprob(Y_rv, S_rv)
```

Log-density graphs are standard Aesara graphs, so we can compute
compile them to compute values:

``` python
logprob_fn = aesara.function([y, s], logprob)

logprob_fn(-0.5, 1.0)
# array(-2.46287705)
```

AePPL provides utilities to pretty-print the log-density graphs:

``` python
from aeppl import pprint, latex_pprint


# Print the original graph
print(pprint(Y_rv))
# b ~ invgamma(0.5, 0.5) in R, a ~ N(0.0, sqrt(b)**2) in R
# a

print(latex_pprint(Y_rv))
# \begin{equation}
#   \begin{gathered}
#     b \sim \operatorname{invgamma}\left(0.5, 0.5\right)\,  \in \mathbb{R}
#     \\
#     a \sim \operatorname{N}\left(0.0, {\sqrt{b}}^{2}\right)\,  \in \mathbb{R}
#   \end{gathered}
#   \\
#   a
# \end{equation}

# Simplify the graph so that it's easier to read
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.tensor.rewriting.basic import topo_constant_folding


logprob = rewrite_graph(logprob, custom_rewrite=topo_constant_folding)


print(pprint(logprob))
# s in R, y in R
# (switch(s >= 0.0,
#         ((-0.9189385175704956 +
#           switch(s == 0, -inf, (-1.5 * log(s)))) - (0.5 / s)),
#         -inf) +
#  ((-0.9189385332046727 + (-0.5 * ((y / sqrt(s)) ** 2))) - log(sqrt(s))))
```

Joint log-densities can be computed for some terms that are *derived* from
`RandomVariable`s, as well:

``` python
# Create a switching model from a Bernoulli distributed index
Z_rv = srng.normal([-100, 100], 1.0, name="Z")
I_rv = srng.bernoulli(0.5, name="I")

M_rv = Z_rv[I_rv]
M_rv.name = "M"

# Compute the joint log-probability for the mixture
logprob, (m, z, i) = joint_logprob(M_rv, Z_rv, I_rv)


logprob = rewrite_graph(logprob, custom_rewrite=topo_constant_folding)

print(pprint(logprob))
# i in Z, m in R, a in Z
# (switch((0 <= i and i <= 1), -0.6931472, -inf) +
#  ((-0.9189385332046727 + (-0.5 * (((m - [-100  100][a]) / [1. 1.][a]) ** 2))) -
#   log([1. 1.][a])))
```

Take a look at the [documentation][documentation-examples] for more examples.


## Install

The latest release of `aeppl` can be installed from PyPI using `pip`:

``` bash
pip install aeppl
```

Or via conda-forge:

``` bash
conda install -c conda-forge aeppl
```

The nightly (bleeding edge) version of `aeppl` can be installed using `pip`:

``` bash
pip install aeppl-nightly
```

## Get help

Report bugs by opening an [issue][issues]. If you have a question regarding the usage of AePPL, start a [discussion][discussions] or visit our [Discord server][discord] and [Gitter room][gitter] chats.

## Contribute

AePPL welcomes contributions. To start contributing, take a look at the open [issues][issues].

If you want to implement a new feature, open a [discussion][discussions] or come chat with us on [Discord][discord] or [Gitter][gitter].

[contributors]: https://github.com/aesara-devs/aeppl/graphs/contributors
[contributors-badge]: https://img.shields.io/github/contributors/aesara-devs/aeppl?style=flat-square&logo=github&logoColor=white&color=ECEFF4
[discussions]: https://github.com/aesara-devs/aeppl/discussions
[documentation-examples]: https://aeppl.readthedocs.io/en/latest/examples.html
[downloads-badge]: https://img.shields.io/pypi/dm/aeppl?style=flat-square&logo=pypi&logoColor=white&color=8FBCBB
[discord]: https://discord.gg/h3sjmPYuGJ
[discord-badge]: https://img.shields.io/discord/1072170173785723041?color=81A1C1&logo=discord&logoColor=white&style=flat-square
[gitter]: https://gitter.im/aesara-devs/aeppl
[gitter-badge]: https://img.shields.io/gitter/room/aesara-devs/aeppl?color=81A1C1&logo=matrix&logoColor=white&style=flat-square
[issues]: https://github.com/aesara-devs/aeppl/issues
[releases]: https://github.com/aesara-devs/aeppl/releases
[twitter]: https://twitter.com/AesaraDevs
[twitter-badge]: https://img.shields.io/twitter/follow/AesaraDevs?style=social
[pypi]: https://pypi.org/project/aeppl/
[pypi-badge]: https://img.shields.io/pypi/v/aeppl?color=ECEFF4&logo=python&logoColor=white&style=flat-square
