Reconstruction
==============

Ghostbuster
--------------

Start from an empty volume. Simulate what it would look like under the
known imaging conditions, compare against the real images, and adjust the
volume to reduce the difference. Repeat.

Because the forward model is differentiable, "adjust the volume" is
ordinary gradient descent. The volume is a tensor of learnable parameters,
and every voxel receives a gradient from every image that sees it. Nothing
is back-projected.

What you can refine
------------------------

Four quantities can be optimised. Each has a learning-rate argument, and
that argument is also the on/off switch: **leaving it unset means the
quantity is held fixed**. This is the least obvious thing about the API
and the most common source of confusion.

.. list-table::
   :widths: 15 25 20 40
   :header-rows: 1

   * - Argument
     - Controls
     - If unset
     - If given a value
   * - ``lr``
     - The 3D volume
     - frozen
     - refined + scheduled
   * - ``lr_R``
     - Per-particle orientation
     - poses held fixed
     - refined, constant rate
   * - ``lr_T``
     - Per-particle shift
     - shifts held fixed
     - refined, constant rate
   * - ``lr_D``
     - One global defocus offset
     - defocus held fixed
     - refined, constant rate

.. note::
   **On lr_D.** The defocus offset is a single scalar shared by the
   entire stack, not a per-particle value. It corrects a systematic error
   in the defocus estimate, not individual particle defoci.

Running it
-------------

.. code-block:: python

   import specter.jobs as jobs
   from specter.ghostbuster import Ghostbuster

   jobs.base_directory("/scratch/user/cryo-runs")
   with jobs.Job("ghostbuster", "my-project") as job:
       gb = job.create(
           Ghostbuster,
           cs_file="particles.cs",
           mrc_file="stack.mrcs",
           dose_per_angstrom=40.0,
           lr=0.1, symmetry="I1", epochs=5,
       )
       gb.test_run()          # 8x-binned, one epoch — do this first
       model = gb.run(device=0)

``test_run()`` bins the images eight-fold and runs a single epoch. A
configuration error shows up in seconds instead of hours, which is worth
the habit.

The other settings, in plain terms
----------------------------------------

**Scattering model — use rytov here, not multislice.** Reconstruction
evaluates the forward model thousands of times. Multislice, which is the
right choice for generating data, is too slow to iterate with. Rytov
captures phase accumulation without per-slice propagation and is accurate
enough for the purpose.

.. warning::
   **Inconsistent defaults.** The ``Ghostbuster`` driver defaults to
   ``rytov``, but ``Reconstructor`` itself still defaults to
   ``multislice``. Constructing a ``Reconstructor`` directly gives you the
   slow path unless you say otherwise.

**Symmetry — averaging equivalent orientations after each epoch.** Many
complexes have point-group symmetry — an octahedral particle looks
identical from 24 orientations. Declaring it means every image contributes
to all equivalent views, which multiplies effective data. Averaging is
applied to the volume after each epoch. ``symmetry_batchsize`` limits how
many symmetry operations are held in memory at once; reduce it if you run
out during the averaging step.

.. warning::
   **Convention mismatch fails silently.** Input poses must already be in
   RELION's symmetry-axis convention. A mismatch does not raise an error —
   it produces a smeared map, which is easy to misread as a data problem.

**Loss function — four options, chosen by which argument you set.** The
first matching option wins:

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Set this
     - Uses
     - Choose it when
   * - ``use_ncc=True``
     - Normalised cross-correlation
     - Image scale or gain does not match the model's absolute prediction.
   * - ``learn_noise_model=True``
     - Learned per-shell noise weighting
     - You want RELION-style weighting, estimated from the residuals as
       training proceeds.
   * - ``nps_weight=...``
     - Fixed noise-power weighting
     - You already know the noise spectrum and want it held constant.
   * - none of the above
     - Mean squared error
     - The default. Start here.

All four accept per-particle weights through ``scale`` — CryoSPARC's
``alignments3D/alpha``, for instance — so unreliable particles contribute
less. An optional ``sparsity`` term adds an L1 penalty on the volume,
suppressing low-level noise in empty regions.

The NCC loss is scaled by the target variance so it sits in the same
units as MSE, letting you switch loss functions without retuning ``lr``:

.. math::

   \mathrm{NCC} = \frac{\sum_i (p_i - \bar{p})(t_i - \bar{t})}
                       {\lVert p - \bar{p} \rVert \, \lVert t - \bar{t} \rVert}
   \qquad
   \mathcal{L}_{ncc} = \mathrm{Var}(t) \cdot (1 - \mathrm{NCC})

where :math:`p` is the simulated image, :math:`t` the experimental one,
and the sums run over pixels within one image.

**Frequency masking — kmask, and why it is applied every batch.**
``kmask`` restricts the volume to a region of frequency space after each
batch — typically ``ball3d(n, n)``, a sphere reaching to the Nyquist
limit. Frequencies beyond that cannot be supported by the sampling and
would otherwise accumulate noise that looks like structure.

.. important::
   **Intensity convention — the single most common setup error.**
   Ghostbuster works in expected electron counts, not normalised
   particles. Converting back means undoing the sign flip and the
   standardisation applied at generation:

   .. math::

      I_{\text{counts}} = -\sqrt{D_{\text{area}}} \cdot p + D_{\text{area}}
      \qquad
      D_{\text{area}} = \text{dose\_per\_angstrom} \times \text{voxel\_size}^2

   where :math:`p` is the raw normalised particle stack. The
   ``Ghostbuster`` driver does this for you. Driving ``Reconstructor``
   directly means doing it yourself.

   **Tomograms differ.** The tomogram path's forward model is noiseless,
   so observed tilts are instead divided by
   ``dose_per_angstrom × voxel_size²``. Or pass ``TiltSeriesGenerator``'s
   ``clean_images`` straight through.

Tomogram reconstruction
----------------------------

``TomogramReconstructor`` and ``TomogramGhostbuster`` handle tilt series —
a different class, with fewer knobs. Orientations there are known — they
are the stage tilts — so only the volume is refined. There is no ``lr_R``,
``lr_T`` or ``lr_D``.

The design constraint is memory: one tilt per step by default, with
``slice_batch_size`` and ``checkpoint_chunks`` trading compute for
activation memory. Gradient checkpointing matters at high tilt, where the
propagated Z extent grows considerably.

.. note::
   ``checkpoint_chunks`` is not plumbed through ``TomogramGhostbuster``.
   Construct ``TomogramReconstructor`` directly to use it.

Watching it converge
------------------------

``VolumeMonitorCallback`` refreshes orthogonal projections every N steps
in a notebook, so a run that is diverging is obvious early.

Judging the result
----------------------

Two Fourier shell correlation functions exist and they apply different
thresholds. Using the wrong one is a reliable way to misreport a
resolution. For two volumes' Fourier transforms :math:`F_1(\mathbf{k})`,
:math:`F_2(\mathbf{k})`, FSC is the normalised cross-correlation within
each spatial-frequency shell :math:`|\mathbf{k}|`:

.. math::

   \mathrm{FSC}(k) = \frac{\mathrm{Re}\sum_{\mathbf{k}\in\text{shell}} F_1(\mathbf{k})\,F_2^*(\mathbf{k})}
                          {\sqrt{\sum_{\mathbf{k}\in\text{shell}} |F_1(\mathbf{k})|^2 \cdot \sum_{\mathbf{k}\in\text{shell}} |F_2(\mathbf{k})|^2}}

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Function
     - Compares
     - Criterion
   * - ``plot_map_to_model_fsc``
     - A volume against a known reference
     - FSC = 0.5
   * - ``plot_halfmap_fsc``
     - Two independent half-maps
     - FSC = 0.143

Half-map reconstructions
------------------------------

There is no single-call helper — here is the procedure. Run the pipeline
twice against the same job folder with ``return_class="0"`` and then
``"1"``. This produces ``vol_A.mrc`` and ``vol_B.mrc`` side by side, which
``plot_halfmap_fsc`` then compares.

The job system is what makes this trustworthy: resuming into the same job
validates that every other setting matched. ``return_class`` and
``cryosparc_ref`` are excluded from that check precisely because they are
the two things that legitimately differ between half-sets.

Per-atom Q-scores
---------------------

``specter.qscore`` implements the Q-score of Pintilie et al. (2020),
scoring how well the map supports each individual atom rather than
reporting one number per shell. That localises problems FSC averages
away.

.. note::
   **Alignment first.** The model must already be spatially aligned to
   the map. A misaligned model returns Q ≈ 0 regardless of map quality.
