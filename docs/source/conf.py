import os

import sphinx.addnodes
import sphinx.directives
from docutils import nodes
from sphinx.util.docutils import SphinxDirective

import aeppl

# -- Project information

project = "aeppl"
author = "Aesara Developers"
copyright = f"2021-2023, {author}"

version = aeppl.__version__
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if "." not in rtd_version and rtd_version.lower() != "stable":
        version = "dev"
else:
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", "")
    if branch_name == "main":
        version = "dev"
release = version


# -- General configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_nb",
    "sphinx_copybutton",
]

templates_path = ["_templates"]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "code"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- Options for extensions

nb_execution_mode = "auto"
# execution_excludepatterns = ["*.ipynb"]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]

autosummary_generate = True
always_document_param_types = True

# -- Options for HTML output

html_theme = "furo"
html_logo = "logo.png"

intersphinx_mapping = {
    "aesara": ("https://aesara.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}


class SupportedDistributionsDirective(SphinxDirective):
    def run(self):
        from aesara.tensor.random.op import RandomVariable

        from aeppl.logprob import _logprob

        supported_dists = tuple(
            mtype.__name__
            for mtype, mfunc in _logprob.registry.items()
            if issubclass(mtype, RandomVariable)
            and not mtype.__module__.startswith(r"aeppl.")
            and not mfunc.__name__ == "transformed_logprob"
        )

        res = nodes.bullet_list()
        for dist_name in supported_dists:
            attributes = {}
            reftarget = f"aesara.tensor.random.basic.{dist_name}"
            attributes["reftarget"] = reftarget
            attributes["reftype"] = "class"
            attributes["refdomain"] = "py"

            ref = nodes.paragraph()

            rawsource = rf":external:py:class:`{reftarget}`"
            xref = sphinx.addnodes.pending_xref(rawsource, **attributes)
            xref += nodes.literal(reftarget, reftarget, classes=["xref"])
            # ref += nodes.inline(reftarget, reftarget)

            ref += xref
            item = nodes.list_item("", ref)
            res += item

        return [res]


def setup(app):
    app.add_directive("print-supported-dists", SupportedDistributionsDirective)
