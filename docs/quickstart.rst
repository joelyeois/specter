Quickstart
==========

Simulate a particle stack
--------------------------

The ``generate_particle_stack.py`` demo script is the quickest way to produce
a simulated cryo-EM particle stack from a PDB code:

.. code-block:: bash

   python demo-scripts/generate_particle_stack.py \
       --pdb_code 6bdf \
       --n_particles 20 \
       --num_pixels 256 \
       --pixel_size 1.0 \
       --energy 300 \
       --scattering_model multislice \
       --output_dir ./output

This downloads the structure, builds the scattering potential, applies CTF and
detector effects, and writes a ``.mrcs`` / ``.star`` file pair.

Reconstruct a volume (Ghostbuster)
------------------------------------

Given an experimental particle stack and CryoSPARC ``.cs`` metadata:

.. code-block:: bash

   python demo-scripts/ghostbuster_reconstruct.py \
       --project my-project \
       --mrc_file particles.mrcs \
       --cs_file particles.cs \
       --fsc_ref reference.mrc \
       --fsc_mask mask.mrc \
       --cryosparc_ref cryosparc_vol.mrc \
       --dose_per_angstrom 22.5 \
       --symmetry C1 \
       --return_class 1

See ``demo-notebooks/`` for interactive worked examples.
