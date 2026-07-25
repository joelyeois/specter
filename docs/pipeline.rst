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
relativistically, following Kirkland Eq. (5.6):

.. math::

   \lambda = \frac{hc}{\sqrt{E(E + 2m_ec^2)}}
   \qquad\qquad
   \sigma = \frac{2\pi}{\lambda E} \cdot \frac{E + m_ec^2}{E + 2m_ec^2}

The slice thickness Δz is assumed equal to the pixel size throughout —
there is no independent slice-thickness parameter. In terms of σ, Δz, a
slice potential :math:`V(z)`, and the Fresnel propagator :math:`F(z)`
(complex exponential of :math:`k^2`, wavelength and Δz) from slice
:math:`z` to the exit plane, the exit wave :math:`\psi` for each mode is:

.. math::

   \text{multislice (per slice, iteratively):}\quad
   \psi \leftarrow \mathcal{F}^{-1}\Big[\mathcal{F}\big[\psi \, e^{i\sigma\Delta z\,V(z)}\big] \cdot F(z)\Big]

.. math::

   \text{rytov:}\quad
   \psi = \exp\!\left(i\sigma\Delta z \sum_z \mathcal{F}^{-1}\big[\hat{F}(z)\cdot\hat{V}(z)\big]\right)

.. math::

   \text{firstborn:}\quad
   \psi = 1 + i\sigma\Delta z \sum_z \mathcal{F}^{-1}\big[\hat{F}(z)\cdot\hat{V}(z)\big]

.. math::

   \text{kinematic:}\quad
   \psi = 1 + \sum_z \mathcal{F}^{-1}\Big[\hat{F}(z)\cdot\mathcal{F}\big[e^{i\sigma\Delta z\,V(z)} - 1\big]\Big]

where :math:`\mathcal{F}` denotes the 2D Fourier transform. ``kinematic``
keeps the per-slice amplitude :math:`e^{i\sigma\Delta z\,V} - 1` exact,
where ``firstborn`` linearises it to :math:`i\sigma\Delta z\,V` — the two
agree when :math:`\sigma\Delta z\,V \ll 1` per slice and diverge for thick
or dense slices.

``IterativeScattering``, used for micrographs and tilt series, adds a
seventh mode (``rytov_parallel``) and samples rotated slices on demand
instead of rotating a whole tomogram-sized volume.

.. note::
   **Not all modes are reachable from the CLI.** The generation demo
   scripts (``generate_particle_stack.py``, ``generate_micrograph.py``,
   ``generate_particle_stack_from_csfile.py``, ``generate_tilt_series.py``)
   only expose ``--scattering_model {multislice, firstborn, projection,
   ctf}``. ``ghostbuster_reconstruct.py`` separately exposes ``{multislice,
   rytov, firstborn, projection}``. ``kinematic`` and ``rytov_parallel``
   are not wired into any script's CLI — use them by constructing
   ``Scattering``/``IterativeScattering`` directly in Python.

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

The transfer function is :math:`\exp(-i\chi(k))`, where :math:`\chi(k)`
sums whichever of these terms are supplied (:math:`k` = spatial frequency
magnitude, :math:`\theta` = its azimuthal angle, :math:`\lambda` =
wavelength):

.. math::

   \chi_{cs} = \frac{\pi}{2}\lambda^3 C_s k^4
   \qquad
   \chi_{defocus} = -\pi\lambda k^2 \Delta f

.. math::

   \Delta f = \tfrac{1}{2}\Big[(\Delta f_u + \Delta f_v) + (\Delta f_v - \Delta f_u)\cos\big(2(\theta + \theta_{ast})\big)\Big]

.. math::

   \chi_{trefoil} = t_1 k^3 \sin(3\theta) + t_2 k^3 \cos(3\theta)

.. math::

   \chi_{tilt} = -2\pi\lambda^2 C_s k^2 \big(\sin(\tau_y)\,k_x + \sin(\tau_x)\,k_y\big)

A supplied phase shift (Volta phase plate, holography model) contributes a
uniform :math:`\chi_{phaseshift} = -\phi_0`, forced to zero at DC to keep
Fourier optics valid.

Four envelopes then damp high frequencies, each switched on by supplying
its parameter. Each multiplies the transfer function's amplitude:

.. math::

   E_{bfactor}(k) = \exp\!\left(-\frac{B k^2}{4}\right)

.. math::

   E_{spatial}(k) = \exp\!\left(-\left(\frac{\pi\alpha_c}{\lambda}\right)^{\!2}\left(C_s\lambda^3 k^3 + \lambda\,\Delta f\,k\right)^{\!2}\right)

.. math::

   E_{temporal}(k) = \exp\!\left(-\tfrac{1}{2}\left(\pi\lambda\,\Delta f_c\,k^2\right)^{\!2}\right)
   \qquad
   \Delta f_c = C_c\sqrt{\left(\frac{\Delta E}{V}\right)^{\!2} + \left(\frac{\Delta V}{V}\right)^{\!2} + \left(2\frac{\Delta I}{I}\right)^{\!2}}

where :math:`\alpha_c` is the beam convergence semi-angle
(``convergence_angle``) and :math:`\Delta f_c` is the chromatic focus
spread, combining ``cc``, ``energy_spread``, ``deltaV_V`` and ``deltaI_I``.
The dose envelope (Grant & Grigorieff 2015) is a fitted, dose-dependent
curve rather than a closed form — see the table below.

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
   particles normalised, with a bright particle on a dark background. The
   particle-stack scripts end with a sign flip and standardisation when
   ``normalize_particles`` is enabled (the default), and reconstruction
   must undo it — which is a common source of confusing results. See
   :doc:`reconstruction`. Micrographs and tilt series are standardised but
   never sign-flipped — they stay in the raw dense-is-dark convention
   regardless of normalisation settings.
