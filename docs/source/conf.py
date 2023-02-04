import os

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
    "sphinx_design",
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

autosummary_generate = True
always_document_param_types = True

# -- Options for HTML output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/aesara-devs/aeppl",
    "use_repository_button": True,
    "use_download_button": False,
}
html_logo = "logo.png"
html_title = ""

intersphinx_mapping = {
    "aesara": ("https://aesara.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}
