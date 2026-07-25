Job management
==============

Runs that record themselves
--------------------------------

A reconstruction can run for days on a shared machine and be restarted by
a scheduler. The jobs system exists so that months later you can still
tell what produced a given map.

Where jobs live
-------------------

There is no default location — you set one explicitly, and failing to do
so raises rather than writing somewhere surprising.

.. code-block:: python

   # any one of these
   jobs.base_directory("/scratch/user/cryo-runs")          # per session
   # export SPECTER_JOBS_DIR=/scratch/user/cryo-runs        # environment
   Job(..., base_dir="/scratch/user/cryo-runs")             # explicit

What a job folder holds
----------------------------

.. code-block:: text

   <base_dir>/<project>/J001/
   ├── job.json           # status, timestamps, version, git commit, full params
   ├── params_A.json      # effective configuration for the A half-set
   ├── metrics_A.json     # per-epoch loss and learning rate
   ├── vol_A.mrc          # final volume
   ├── fsc_A.png
   ├── kmask.pt
   └── epochs/
       ├── 001_A.mrc
       ├── vol_001_A.png  # projection preview
       └── fsc_001_A.png

The ``_A`` and ``_B`` suffixes appear only when you ran half-sets.
``plots.visualize_job_epochs(job_folder)`` turns the ``epochs/`` directory
into a slider you can step through in Jupyter.

Two properties worth relying on
-------------------------------------

**The record includes what you never specified.** A parameter log that
captures only what you explicitly passed is close to useless later,
because the defaults change as the package evolves. ``job.create()``
inspects the target class's signature and applies its defaults before
recording, so ``job.json`` holds the complete effective configuration.
Tensors are stored as shape and dtype rather than data, keeping the record
readable. The output directory is injected automatically, so you never
pass a path. Each run also stamps the installed package version and the
current git commit, so a result can be traced to the code that produced
it.

**Resuming validates rather than assumes.** Pinning a job ID means the
first invocation creates the job and later invocations with the same ID
resume into it. That is what makes the reconstruction script safe to
submit to a scheduler that may restart it.

On resume, incoming parameters are compared key by key against what was
stored, and any mismatch raises with both values shown. This is a real
safety property: it prevents silently combining two half-sets that were
reconstructed under different settings, which would produce a resolution
estimate that means nothing.

Looking at past runs
------------------------

.. code-block:: bash

   specter-jobs list --project empiar-10202
   specter-jobs show empiar-10202 J001
   specter-jobs diff empiar-10202 J001 J002

``diff`` reports only the parameters that differ between two jobs, which
is the fastest way to answer why one run worked and another did not.

Submitting to a scheduler
------------------------------

Pin the job ID in the submission script from the start. The first run
creates ``J001``; a restart resumes into it rather than opening ``J002``
and scattering one experiment across two folders.
