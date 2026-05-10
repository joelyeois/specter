Installation
============

Requirements
------------

- Python 3.11+
- `uv <https://docs.astral.sh/uv/>`_ (recommended) or pip

From source
-----------

.. code-block:: bash

   git clone <repo-url>
   cd specter
   uv sync
   source .venv/bin/activate

GPU support
-----------

SPECTER uses PyTorch for all array operations. GPU acceleration is available
automatically when a CUDA-capable device is present; all classes fall back to
CPU otherwise.
