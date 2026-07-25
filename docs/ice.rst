Ice
===

The hardest part is the water
---------------------------------

A particle is a small perturbation on a thick slab of frozen water. Get the
water wrong and the simulation looks synthetic no matter how good the rest
of the physics is.

Amorphous ice is not a crystal and not a liquid. It has no long-range
order, but it is far from random: water molecules keep characteristic
distances and angles from their neighbours. Scattering from that structure
is most of the signal in a real micrograph. The naive approach — scatter
molecules at random until you hit the right bulk density — gets the
average density right and the structure entirely wrong.

Three ways to make ice
--------------------------

.. list-table::
   :widths: 20 30 15 35
   :header-rows: 1

   * - Generator
     - Method
     - Cost
     - Structural fidelity
   * - ``RandomIcemaker``
     - Uniform random placement at the correct number density
     - instant
     - None. Molecules may overlap.
   * - ``GradientSKIcemaker``
     - L-BFGS optimisation against a target S(k) and an MLBOP water energy
     - ~22 min / config
     - High — this is the reference.
   * - ``IceBank``
     - Draws rotated, translated crops from a cache of pre-optimised
       configurations
     - single-digit ms
     - High — it *is* the reference, reused.

That third row is the whole trick. Optimised ice is far too slow to
generate per particle, so it is generated once, offline, and then sampled.
In configuration files this appears as ``ice_model = "gd"``, which is the
default nearly everywhere.

How the cache is sampled
----------------------------

The cache holds 20 converged configurations, each a 256 Å cube of
527,178 molecules. Sampling one:

1. **Draw** a cached config.
2. **Rotate + translate**, applied to the molecular coordinates, not the
   voxels.
3. **Crop** to the requested volume.
4. **Tile**, when the requested box exceeds one config, and **relax the
   seams** with a short local MLBOP minimisation.

Two details here are deliberate design decisions rather than
implementation accidents, and both are documented in the source as things
not to "simplify" later.

**Why rotation happens in coordinate space.** The obvious way to get a
fresh-looking ice volume from a cached one is to rotate the voxel grid.
That interpolates, and interpolation smooths — it would systematically
damp exactly the high-frequency structure the optimisation was run to get
right. Instead the rotation is applied to the continuous molecular
coordinates *before* voxelisation. The result is a genuinely different,
equally sharp configuration.

**Why seams are relaxed rather than blended.** A micrograph-sized field is
much larger than a 256 Å cached cube, so several crops must be placed side
by side. Concatenating them leaves boundaries where molecules from
adjacent tiles sit at physically impossible distances — visible as seams,
and measurably unfavourable in energy. Rather than blending in voxel
space, which would smear structure, the tiles are placed in coordinate
space and a short local MLBOP minimisation is run in a margin around each
boundary. The molecules move until the join is physically reasonable.

.. warning::
   **Do not replace with** a plain repeat, hard concatenation, or
   voxel-space blend. The first two leave seams; the third destroys the
   structure that makes the cache worth having.

How the ice is tested
-------------------------

Ice quality is checked against two independent targets, which is what
makes the check meaningful — one is structural, the other energetic, and
they were not optimised for jointly by accident.

**Test 1 — structural.** The configuration's radially-averaged S(k) is
compared against a target computed from molecular dynamics. This is also
the primary optimisation objective, so it measures convergence rather than
independent validity.

**Test 2 — energetic.** Each molecule is scored under a coarse-grained
water potential (Chan et al., *Nat. Commun.* 10, 379, 2019). Lower energy
per atom means a more physically plausible arrangement — a check the S(k)
objective does not directly enforce.

What a cached configuration actually contains
--------------------------------------------------

Read directly out of the shipped ``config_000.pt``:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Field
     - Value
     - Meaning
   * - ``n_molecules``
     - 527,178
     - Coarse-grained water beads in the cell
   * - ``box_L``
     - 256.0 Å
     - Periodic cubic cell edge
   * - ``n_steps``
     - 600
     - L-BFGS iterations, all completed
   * - ``wall_time``
     - 1,315.8 s
     - ~22 minutes on GPU, per configuration
   * - ``E_per_atom``
     - -0.1088
     - MLBOP energy reached
   * - ``mlbop_target``
     - -0.413
     - Energy the optimiser was pulled toward
   * - ``rij_mean``
     - 2.862 Å
     - Mean neighbour separation

Twenty such configurations ship with the repository, about 60 MB in
total. The step count is set to 600 rather than the class default of 400
because S(k) plateaus by roughly step 250 while the energy keeps
improving — and since these are permanent shared assets, the extra time
is paid once.

Why the energy term can be optimised at all
------------------------------------------------

MLBOP began as a diagnostic: you computed it after the fact to check
whether a configuration was plausible. The neighbour search ran through
ASE, which meant leaving torch, moving positions to the host, and losing
the computational graph.

Moving that search to ``vesin_torch`` keeps everything in torch and —
crucially — makes the returned distances differentiable with respect to
position. The diagnostic became usable as a loss. That is what allows both
the seam relaxation and ``GradientSKIcemaker``'s default recipe to
optimise energy directly.

Using it
-----------

``ice_model = "gd"`` selects the cached ``IceBank`` path and is what you
want unless you have a specific reason otherwise (as opposed to
``"random"`` for ``RandomIcemaker``, or ``"none"``). Two things surprise
people:

- ``GradientSKIcemaker`` is not reachable through ``ice_model``. To
  generate fresh optimised ice rather than sampling the cache, construct
  it yourself and pass ``icemaker=``. To extend the cache instead, use
  ``build_ice_cache()``.
- The cache is not installed as package data. Its location is resolved by
  walking up to the repository root. Installing SPECTER outside a
  checkout means copying ``ice-data/ice_cache/`` or pointing
  ``--ice_cache_dir`` at it.
