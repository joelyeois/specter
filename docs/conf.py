import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "SPECTER"
copyright = "2026, Joel Yeo"
author = "Joel Yeo"
release = "0.1.0"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "autoapi.extension",
    "sphinx_copybutton",
]

# NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_use_param = True

# AutoAPI — reads src/specter and generates API reference
autoapi_dirs = ["../src/specter"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_add_toctree_entry = True
autoapi_ignore = ["*/.ipynb_checkpoints/*"]
suppress_warnings = ["autoapi.python_import_resolution"]

# Cross-references to upstream docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

html_theme = "furo"
html_title = "SPECTER"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
