Pipeline
========

From atoms to counts
------------------------

Eight transformations turn a list of atomic coordinates into something that
looks like it came off a detector::

    Atoms (N,3)  →  Potential (Z,Y,X volts)  →  Pose (quat + shift)  →  Crowd
        →  Scale (× potential)  →  Ice (solvate)  →  Scatter (exit wave)
        →  Aberrate (CTF)  →  Detect (counts)

.. note::
   **Read this first.** The three generators share this physics but not one
   code path. Particle stacks build the specimen inside every forward pass,
   so each particle gets fresh ice. Micrographs assemble crowding and ice
   once and reuse them. Tilt series receive a volume that already contains
   both, and are driven by ``generate_tilt_series()`` rather than
   ``forward()``.

1 — Atoms to potential
--------------------------

``PotentialBuilder`` takes atomic numbers and coordinates and deposits each
atom's electrostatic potential onto a voxel grid. Deposition is **soft** —
each atom is splatted across neighbouring voxels rather than snapped to the
nearest one — which keeps the result a smooth, differentiable function of
the coordinates. That smoothness is what later allows gradients to reach
the atoms.

Kernels are built on a fine supersampled grid near 0.1 Å and pooled down to
your pixel size, so a coarse grid still sees a correctly-shaped atom.

- **Input**: ``coordinates`` (N,3) in Å, centred on the origin;
  ``atomic_numbers`` (N,)
- **Output**: (Z, Y, X) in volts
- **Options**: ``parameterization`` — ``kirkland``, ``lobato``, ``shtyrov``

Render a built potential with ``plot3d(V)`` — the standard way to eyeball a
volume before simulating from it.

2 — Pose
-----------

Each particle gets a quaternion and a 2D translation in Ångström. The
volume is resampled under that transform, then padded — in Z to make room
for ice, and in XY to keep FFT wraparound out of the final field of view.

Poses are either sampled randomly, or read from a CryoSPARC ``.cs`` file
when you want to mirror a real dataset. Under reconstruction these same
quaternions can become learnable.

The two particle classes pad differently: ``ImageGenerator`` pads XY with
zeros, ``ImageGeneratorFromCoordinates`` reflects. Worth knowing if you
compare outputs across them.

3 — Crowding
---------------

Real particles are not alone in the hole. Crowding places additional
copies around the central one by Poisson-disk sampling, each independently
rotated, at a minimum separation you control with ``crowd_min_distance``
(defaulting to the molecule's own diameter). Setting it to ``0`` or
``None`` disables crowding.

With ``water_air_interface`` enabled, positions in Z are rejection-sampled
toward a two-peaked distribution, concentrating particles near the ice
surfaces the way they adsorb in practice.

The copies are duplicates of the same template — there is no compositional
heterogeneity.

4 — Potential scaling
-------------------------

The whole volume is multiplied by ``potential_scale``. Values below 1
weaken the particle relative to everything else, which is a cheap
approximation of a particle sitting in thicker ice. Randomise it per
particle with a ``_min``/``_max`` pair to get a stack with realistically
variable contrast.

Whatever value each particle drew is written into the output STAR file as
``specterPotentialScale``, so you can correlate downstream results against
it.

5 — Ice
----------

An ice volume is generated and blended in through a mask that only fills
voxels below 5% of the specimen's peak potential — so ice fills the space
*around* the particle rather than overwriting it.

Ice has its own frame, because how it is generated is one of the more
interesting parts of the package. See :doc:`ice`.

6 — Scattering
------------------

This is where the electron beam meets the sample. The output is an
**exit wave** — a complex field describing the beam's amplitude and phase
as it leaves the specimen.

Six modes are implemented, trading accuracy for speed:

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Mode
     - Idea
     - Use for
   * - ``multislice``
     - Propagate slice by slice through the volume, accumulating multiple
       scattering.
     - Generation. The default and the most accurate.
   * - ``rytov``
     - Higher-order phase approximation without per-slice propagation.
     - Reconstruction, where the model runs thousands of times.
   * - ``firstborn``
     - Single-scattering approximation.
     - Fast comparisons.
   * - ``kinematic``
     - Variant of first Born.
     - Rarely.
   * - ``projection``
     - Sum the potential along Z, no Fresnel propagation.
     - Thin specimens, quick tests.
   * - ``ctf``
     - Returns a real projected potential rather than a wave.
     - Conventional CTF-only workflows.

The interaction parameter σ is energy-dependent and computed
relativistically. The slice thickness Δz is assumed equal to the pixel
size throughout — there is no independent slice-thickness parameter.

``IterativeScattering``, used for micrographs and tilt series, adds a
seventh mode (``rytov_parallel``) and samples rotated slices on demand
instead of rotating a whole tomogram-sized volume.

7 — Aberration
------------------

Real lenses are imperfect, and those imperfections are what make phase
visible as contrast. The exit wave is Fourier transformed, multiplied by a
transfer function, and brought back.

The phase term is assembled from whichever parameters you supply:

- **Defocus and astigmatism** — ``dfu``, ``dfv``, ``dfang``. Positive
  defocus is underfocus.
- **Spherical aberration** — ``cs``.
- **Phase shift** — for phase-plate work.
- **Beam tilt** and **trefoil** — higher-order terms.

Four envelopes then damp high frequencies, each switched on by supplying
its parameter:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Envelope
     - Enabled by
     - Represents
   * - B-factor
     - ``bfactor``
     - Generic resolution falloff
   * - Spatial coherence
     - ``convergence_angle``
     - Finite source size
   * - Temporal coherence
     - ``cc``
     - Energy spread, lens and voltage instability
   * - Dose
     - ``dose_envelope``
     - Cumulative radiation damage (Grant & Grigorieff 2015)

.. warning::
   **Not implemented.** ``tetrafoil`` is accepted as a parameter name but
   its implementation is an empty stub. Passing any ``tetrafoil*`` key
   raises an error when the transfer function is assembled.

8 — Detector
----------------

Five steps in a fixed order:

1. Form intensity from the wave and scale by dose and pixel area, giving
   expected electron counts.
2. Crop away the FFT padding back to the requested box size.
3. Apply anisotropic magnification, if supplied.
4. Apply the detector MTF in Fourier space.
5. Apply shot noise and coincidence loss.

Only step 4 is a Fourier operation; magnification and the noise model
work in real space.

**Coincidence loss.** A counting detector reads out many times per
exposure and tries to identify individual electron strikes. When two
electrons land close together within one frame, it registers one. SPECTER
simulates this directly rather than correcting for it analytically:
electrons land with sub-pixel positions, are binned into cells of size
``r/√2``, and one survives per occupied cell.

The consequence is a suppression of low spatial frequencies that matches
what real K3 data shows. Set ``coincidence_radius`` to 0 to recover plain
Poisson statistics. Detector MTF curves are available for ``k3_300kv``,
``k3_200kv`` and an idealised ``perfect`` detector.

What comes out
------------------

Inspect a generated stack with ``plot_particle_stack`` — one call renders
the images, their Fourier transforms (where the CTF's Thon rings are
visible), and radial profiles of those transforms against resolution.

.. important::
   **Sign convention.** SPECTER's raw output is physical: dense regions
   scatter electrons away and appear *dark*. Cryo-EM convention stores
   particles normalised, with a bright particle on a dark background.
   Generation therefore ends with a sign flip and standardisation, and
   reconstruction must undo it — which is a common source of confusing
   results. See :doc:`reconstruction`.
