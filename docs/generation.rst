Data generation
===============

Four ways to make data
--------------------------

Each workflow exists as a notebook for exploring, a command-line script for
batches, and the underlying classes for anything custom. Activate the
environment first (``source .venv/bin/activate``), or run scripts via
``uv run <script name>.py``.

Particle stack
------------------

Isolated particles in random orientations with randomised imaging
conditions — the direct analogue of an extracted particle stack, and where
most people start. Parameters are loaded from a TOML config file
(``configs/particle.toml`` by default, see :doc:`configuration`); any flag
below overrides a single field without editing it.

.. code-block:: bash

   python demo-scripts/generate_particle_stack.py \
       --config configs/particle.toml \
       --pdb_code 6bdf \
       --n_particles 1000 \
       --device cuda:0 \
       --output_dir ./output/

Inspect the output with ``plot_particle_stack``: each column is a
different randomly drawn defocus, and the Thon rings in the Fourier-space
row tighten as defocus increases.

**Randomising per particle.** Defocus, dose, coincidence radius and
potential scale each take a ``_min``/``_max`` pair — omitting ``_max`` (or
setting it equal to ``_min``) uses a fixed value for all particles.
Whatever each particle drew is written into the STAR file as
``specterDosePerAngstrom``, ``specterCoincidenceRadius`` and
``specterPotentialScale``, so a downstream result can be correlated
against the exact conditions that produced it.

**Multiple GPUs.** The only workflow that supports it — pass a
comma-separated device list and the run distributes under Lightning DDP:

.. code-block:: bash

   --device 0,1,2,3   # multi-GPU
   --device cuda:0    # single GPU
   --device cpu       # CPU

.. list-table:: Arguments
   :widths: 24 18 58
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--pdb_code``
     - *(required)*
     - PDB accession code or path to local ``.cif``/``.pdb`` file
   * - ``--assembly``
     - ``True``
     - Fetch biological assembly
   * - ``--pdb_savefolder``
     - ``../pdb-data/``
     - Folder to cache downloaded PDB files
   * - ``--n_particles``
     - ``20``
     - Number of particles to simulate
   * - ``--num_pixels``
     - ``256``
     - Box size in pixels
   * - ``--pixel_size``
     - ``1.0``
     - Pixel size in Å
   * - ``--energy``
     - ``300.0``
     - Beam energy in keV
   * - ``--dose_min``
     - ``20.0``
     - Minimum dose in e⁻/Å²; used as fixed dose if ``--dose_max`` is not set
   * - ``--dose_max``
     - ``None``
     - Maximum dose in e⁻/Å²; if set, dose is sampled uniformly per particle
   * - ``--num_frames``
     - ``int(mean dose)``
     - Number of frames
   * - ``--cs``
     - ``2.0``
     - Spherical aberration in mm
   * - ``--alpha``
     - ``0.1``
     - Amplitude contrast ratio
   * - ``--convergence_angle``
     - ``None``
     - Beam convergence semi-angle in mrad; enables the Cs (spatial
       coherence) envelope
   * - ``--cc``
     - ``None``
     - Chromatic aberration coefficient in mm; enables the Cc (temporal
       coherence) envelope
   * - ``--energy_spread``
     - ``0.7``
     - FWHM of beam energy spread in eV, used by the Cc envelope
   * - ``--deltaV_V``
     - ``0.06e-6``
     - Relative high-voltage instability, used by the Cc envelope
   * - ``--deltaI_I``
     - ``0.01e-6``
     - Relative objective-lens current instability, used by the Cc envelope
   * - ``--dose_envelope``
     - ``False``
     - Apply the Grant & Grigorieff (2015) cumulative-dose envelope
   * - ``--bfactor``
     - ``None``
     - Isotropic B-factor envelope in Å²; enables the B-factor resolution-falloff envelope
   * - ``--defocus_min``
     - ``5000``
     - Minimum defocus in Å; used as fixed value if ``--defocus_max`` is not set
   * - ``--defocus_max``
     - ``15000``
     - Maximum defocus in Å; if set, defocus is sampled uniformly per particle
   * - ``--shift``
     - ``2.0``
     - Max in-plane shift in Å (uniform ±shift)
   * - ``--scattering_model``
     - ``multislice``
     - ``multislice`` \| ``firstborn`` \| ``projection`` \| ``ctf``
   * - ``--aberration_model``
     - ``holography``
     - ``holography`` \| ``ctf``
   * - ``--noise_model``
     - ``poisson``
     - ``poisson`` \| ``none``
   * - ``--coincidence_radius_min``
     - ``1.8``
     - Minimum coincidence radius in pixels; used as fixed value if
       ``--coincidence_radius_max`` is not set
   * - ``--coincidence_radius_max``
     - ``None``
     - Maximum coincidence radius in pixels; if set, sampled uniformly per particle
   * - ``--potential_scale_min``
     - ``1.0``
     - Minimum potential scale factor; used as fixed value if
       ``--potential_scale_max`` is not set
   * - ``--potential_scale_max``
     - ``None``
     - Maximum potential scale factor; if set, sampled uniformly per
       particle. Values < 1 approximate thicker ice (weaker particle signal)
   * - ``--ice_model``
     - ``gd``
     - ``gd`` (samples from the pre-generated ``IceBank`` cache) \|
       ``random`` (instant, cheap ``RandomIcemaker`` placement) \| ``none``
   * - ``--ice_cache_dir``
     - ``None``
     - Directory of cached ice configs for ``ice_model='gd'``. Defaults to
       the bundled ``ice-data/ice_cache``
   * - ``--ice_thickness``
     - ``0.0``
     - Ice thickness in Å; ``0`` = minimum (particle box size)
   * - ``--crowd_min_distance``
     - ``pdb.max_diameter``
     - Min distance between crowded molecules in Å; ``0`` disables crowding
   * - ``--crowd_max_distance_z``
     - ``None``
     - Max z-separation between crowded molecules in Å
   * - ``--pad_fft``
     - ``True``
     - Pad volume to avoid FFT edge artefacts
   * - ``--detector_model``
     - ``none``
     - ``none`` \| ``perfect`` \| ``k3_300kv`` \| ``k3_200kv``
   * - ``--normalize_particles``
     - ``True``
     - Normalise to zero mean and unit std
   * - ``--save_exitwaves``
     - ``False``
     - Also save exit wave magnitude and phase as ``.mrcs``
   * - ``--save_clean_exitwaves``
     - ``False``
     - Save clean (no-ice) exit wave magnitude and phase; runs scattering
       twice per batch
   * - ``--device``
     - ``cpu``
     - ``cpu`` \| ``cuda`` \| ``cuda:0`` \| ``0,1,2,3`` (multi-GPU)
   * - ``--batchsize``
     - ``5``
     - Particles per forward pass
   * - ``--output_dir``
     - ``./output/``
     - Output directory
   * - ``--filename``
     - ``particles``
     - Base name for output files (no extension)

.. list-table:: Output files
   :widths: 35 65
   :header-rows: 1

   * - File
     - Description
   * - ``<filename>.mrcs``
     - Particle image stack
   * - ``<filename>.star``
     - RELION-compatible metadata (poses, CTF, pixel size, voltage) with
       per-particle ``specterDosePerAngstrom``, ``specterCoincidenceRadius``,
       and ``specterPotentialScale`` columns
   * - ``<filename>_exitwave_magnitude.mrcs``
     - Exit wave magnitude (``--save_exitwaves True`` only)
   * - ``<filename>_exitwave_phase.mrcs``
     - Exit wave phase (``--save_exitwaves True`` only)
   * - ``<filename>_clean_exitwave_magnitude.mrcs``
     - Clean exit wave magnitude (``--save_clean_exitwaves True`` only)
   * - ``<filename>_clean_exitwave_phase.mrcs``
     - Clean exit wave phase (``--save_clean_exitwaves True`` only)

A twin of a real dataset
-----------------------------

Take the poses and optics from an actual CryoSPARC refinement and simulate
the same experiment from a known structure. You get data with matched
imaging conditions and ground truth attached.

.. code-block:: bash

   python demo-scripts/generate_particle_stack_from_csfile.py \
       --cs_path /path/to/particles.cs \
       --pdb_code 6bdf \
       --n_particles 1000 \
       --dose 53 \
       --device 0,1,2,3 \
       --output_dir ./output/

Energy, pixel size and amplitude contrast are read out of the ``.cs``
file, so you do not restate them. Dose is the exception — it is not stored
there, and the EMDB entry's Experiment tab is the usual source.

**What is read from the file, and why this is useful for method
development.** Extraction returns beam energy, pixel size, amplitude
contrast, orientations, translations, the full CTF parameter set including
astigmatism and trefoil, per-particle scale factors, and anisotropic
magnification. Because the simulated stack shares its geometry with the
real one, differences between processing them are attributable to the data
rather than to the imaging conditions — that makes this the natural setup
for testing whether a method recovers what it should.

.. list-table:: Arguments
   :widths: 24 18 58
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--cs_path``
     - *(required)*
     - Path to CryoSPARC ``.cs`` file
   * - ``--pdb_code``
     - *(required)*
     - PDB accession code or path to local ``.cif``/``.pdb`` file
   * - ``--assembly``
     - ``True``
     - Fetch biological assembly
   * - ``--pdb_savefolder``
     - ``../pdb-data/``
     - Folder to cache downloaded PDB files
   * - ``--dose``
     - *(required)*
     - Fixed electron dose in e⁻/Å² applied to all particles (check the
       EMDB Experiment tab)
   * - ``--n_particles``
     - all in file
     - Number of particles to simulate
   * - ``--num_pixels``
     - ``256``
     - Box size in pixels
   * - ``--num_frames``
     - ``int(dose)``
     - Number of frames
   * - ``--scattering_model``
     - ``multislice``
     - ``multislice`` \| ``firstborn`` \| ``projection`` \| ``ctf``
   * - ``--aberration_model``
     - ``holography``
     - ``holography`` \| ``ctf``
   * - ``--noise_model``
     - ``poisson``
     - ``poisson`` \| ``none``
   * - ``--convergence_angle``
     - ``None``
     - Beam convergence semi-angle in mrad; enables the Cs (spatial
       coherence) envelope
   * - ``--cc``
     - ``None``
     - Chromatic aberration coefficient in mm; enables the Cc (temporal
       coherence) envelope
   * - ``--energy_spread``
     - ``0.7``
     - FWHM of beam energy spread in eV, used by the Cc envelope
   * - ``--deltaV_V``
     - ``0.06e-6``
     - Relative high-voltage instability, used by the Cc envelope
   * - ``--deltaI_I``
     - ``0.01e-6``
     - Relative objective-lens current instability, used by the Cc envelope
   * - ``--dose_envelope``
     - ``False``
     - Apply the Grant & Grigorieff (2015) cumulative-dose envelope
   * - ``--coincidence_radius``
     - ``2.1``
     - Fixed coincidence radius in pixels applied to all particles; ``0``
       for standard Poisson
   * - ``--ice_model``
     - ``gd``
     - ``gd`` (samples from the pre-generated ``IceBank`` cache) \|
       ``random`` (instant, cheap ``RandomIcemaker`` placement) \| ``none``
   * - ``--ice_cache_dir``
     - ``None``
     - Directory of cached ice configs for ``ice_model='gd'``. Defaults to
       the bundled ``ice-data/ice_cache``
   * - ``--ice_thickness``
     - ``0.0``
     - Ice thickness in Å; ``0`` = minimum (particle box size)
   * - ``--crowd_min_distance``
     - ``pdb.max_diameter``
     - Min distance between crowded molecules in Å; ``0`` disables crowding
   * - ``--crowd_max_distance_z``
     - ``None``
     - Max z-separation between crowded molecules in Å
   * - ``--pad_fft``
     - ``True``
     - Pad volume to avoid FFT edge artefacts
   * - ``--detector_model``
     - ``none``
     - ``none`` \| ``perfect`` \| ``k3_300kv`` \| ``k3_200kv``
   * - ``--normalize_particles``
     - ``True``
     - Normalise to zero mean and unit std
   * - ``--save_exitwaves``
     - ``False``
     - Also save exit wave magnitude and phase as ``.mrcs``
   * - ``--save_clean_exitwaves``
     - ``False``
     - Save clean (no-ice) exit wave magnitude and phase; runs scattering
       twice per batch
   * - ``--device``
     - ``cpu``
     - ``cpu`` \| ``cuda`` \| ``cuda:0`` \| ``0,1,2,3`` (multi-GPU)
   * - ``--batchsize``
     - ``5``
     - Particles per forward pass
   * - ``--output_dir``
     - ``./output/``
     - Output directory
   * - ``--filename``
     - ``particles``
     - Base name for output files (no extension)

Output files are the same as the particle-stack script above.

Full micrographs
--------------------

A complete detector-sized field — 4096 × 4096 by default — with many
crowded particles suspended in a slab of ice, rather than pre-extracted
boxes.

.. code-block:: bash

   python demo-scripts/generate_micrograph.py \
       --pdb_code 6bdf \
       --n_micrographs 10 \
       --micrograph_size 4096 \
       --ice_thickness 500 \
       --chunk_size 8 \
       --device cuda:0

The expensive part — assembling the volume with its crowding and ice —
happens once at initialisation. Each subsequent forward pass applies a
freshly drawn defocus, so producing ten micrographs costs far less than
ten independent setups. Call ``regenerate_specimen()`` when you want a
genuinely new arrangement rather than the same field at a new defocus.

.. note::
   **Memory.** Micrograph-sized volumes are large. ``--chunk_size 8``
   processes the specimen in slice chunks and is the first thing to reach
   for when the GPU runs out of memory.

**The water–air interface.** Particles in a real ice film are not
uniformly distributed in depth; they adsorb to the two air–water
interfaces. With ``--water_air_interface True``, positions in Z are
rejection-sampled toward a two-peaked distribution concentrated near the
surfaces rather than spread evenly through the slab.

.. list-table:: Arguments
   :widths: 24 18 58
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--pdb_code``
     - *(required)*
     - PDB accession code or path to local ``.cif``/``.pdb`` file
   * - ``--assembly``
     - ``True``
     - Fetch biological assembly
   * - ``--pdb_savefolder``
     - ``../pdb-data/``
     - Folder to cache downloaded PDB files
   * - ``--n_micrographs``
     - ``1``
     - Number of micrographs to simulate
   * - ``--num_pixels``
     - ``256``
     - Particle box size in pixels (for potential building)
   * - ``--pixel_size``
     - ``1.0``
     - Pixel size in Å
   * - ``--micrograph_size``
     - ``4096``
     - Micrograph size in pixels (square)
   * - ``--energy``
     - ``300.0``
     - Beam energy in keV
   * - ``--dose_min``
     - ``20.0``
     - Minimum dose in e⁻/Å²; used as fixed dose if ``--dose_max`` is not set
   * - ``--dose_max``
     - ``None``
     - Maximum dose in e⁻/Å²; if set, dose is sampled uniformly per micrograph
   * - ``--num_frames``
     - ``int(mean dose)``
     - Number of frames
   * - ``--cs``
     - ``2.0``
     - Spherical aberration in mm
   * - ``--alpha``
     - ``0.1``
     - Amplitude contrast ratio
   * - ``--convergence_angle``
     - ``None``
     - Beam convergence semi-angle in mrad; enables the Cs (spatial
       coherence) envelope
   * - ``--cc``
     - ``None``
     - Chromatic aberration coefficient in mm; enables the Cc (temporal
       coherence) envelope
   * - ``--energy_spread``
     - ``0.7``
     - FWHM of beam energy spread in eV, used by the Cc envelope
   * - ``--deltaV_V``
     - ``0.06e-6``
     - Relative high-voltage instability, used by the Cc envelope
   * - ``--deltaI_I``
     - ``0.01e-6``
     - Relative objective-lens current instability, used by the Cc envelope
   * - ``--dose_envelope``
     - ``False``
     - Apply the Grant & Grigorieff (2015) cumulative-dose envelope
   * - ``--defocus_min`` / ``--defocus_max``
     - ``5000`` / ``15000``
     - Defocus range in Å
   * - ``--scattering_model``
     - ``multislice``
     - ``multislice`` \| ``firstborn`` \| ``projection`` \| ``ctf``
   * - ``--aberration_model``
     - ``holography``
     - ``holography`` \| ``ctf``
   * - ``--noise_model``
     - ``poisson``
     - ``poisson`` \| ``none``
   * - ``--coincidence_radius_min``
     - ``1.8``
     - Minimum coincidence radius in pixels; used as fixed value if
       ``--coincidence_radius_max`` is not set
   * - ``--coincidence_radius_max``
     - ``None``
     - Maximum coincidence radius in pixels; if set, sampled uniformly per micrograph
   * - ``--potential_scale_min``
     - ``1.0``
     - Minimum potential scale factor; used as fixed value if
       ``--potential_scale_max`` is not set
   * - ``--potential_scale_max``
     - ``None``
     - Maximum potential scale factor; if set, sampled uniformly per
       micrograph. Values < 1 approximate thicker ice
   * - ``--ice_model``
     - ``gd``
     - ``gd`` (samples from the pre-generated ``IceBank`` cache) \|
       ``random`` (instant, cheap ``RandomIcemaker`` placement) \| ``none``
   * - ``--ice_cache_dir``
     - ``None``
     - Directory of cached ice configs for ``ice_model='gd'``. Defaults to
       the bundled ``ice-data/ice_cache``
   * - ``--ice_thickness``
     - ``500.0``
     - Ice thickness in Å
   * - ``--crowd_min_distance``
     - ``pdb.max_diameter``
     - Min distance between crowded molecules in Å; ``0`` disables crowding
   * - ``--crowd_max_distance_z``
     - ``None``
     - Max z-separation between crowded molecules in Å
   * - ``--water_air_interface``
     - ``True``
     - Simulate water-air interface
   * - ``--pad_fft``
     - ``False``
     - Pad volume to avoid FFT edge artefacts
   * - ``--chunk_size``
     - ``None``
     - Slice chunk size for specimen generation; set (e.g. ``8``) if GPU
       memory is limited
   * - ``--detector_model``
     - ``none``
     - ``none`` \| ``perfect`` \| ``k3_300kv`` \| ``k3_200kv``
   * - ``--normalize_micrographs``
     - ``False``
     - Normalise each micrograph to zero mean and unit std
   * - ``--save_exitwaves``
     - ``False``
     - Save icy exit wave magnitude and phase as ``.mrcs``
   * - ``--save_clean_exitwaves``
     - ``False``
     - Save iceless (no-ice) exit wave magnitude and phase; runs scattering
       twice per micrograph
   * - ``--device``
     - ``cpu``
     - ``cpu`` \| ``cuda`` \| ``cuda:0``
   * - ``--output_dir``
     - ``./output/``
     - Output directory
   * - ``--filename``
     - ``micrographs``
     - Base name for output files (no extension)

.. list-table:: Output files
   :widths: 35 65
   :header-rows: 1

   * - File
     - Description
   * - ``<filename>.mrcs``
     - Micrograph stack
   * - ``<filename>.star``
     - Per-micrograph metadata (defocus, voltage, pixel size, amplitude
       contrast) with ``specterDosePerAngstrom``, ``specterCoincidenceRadius``,
       and ``specterPotentialScale`` columns
   * - ``<filename>_exitwave_magnitude.mrcs``
     - Icy exit wave magnitude (``--save_exitwaves True`` only)
   * - ``<filename>_exitwave_phase.mrcs``
     - Icy exit wave phase (``--save_exitwaves True`` only)
   * - ``<filename>_clean_exitwave_magnitude.mrcs``
     - Iceless exit wave magnitude (``--save_clean_exitwaves True`` only)
   * - ``<filename>_clean_exitwave_phase.mrcs``
     - Iceless exit wave phase (``--save_clean_exitwaves True`` only)

Cryo-ET tilt series
------------------------

Image a thick specimen repeatedly as it rotates, the way a tomogram is
collected. This workflow has two halves: build a specimen, then image it.

**Building the specimen.** ``CryoETSpecimenGenerator`` uses a two-stage
strategy to make large, crowded, multi-species specimens tractable. Polnet
performs non-overlapping packing of proteins and membranes at *low*
resolution, so it never allocates tomogram-sized bookkeeping arrays at the
target resolution. SPECTER then renders real high-resolution density from
those placements — a genuine atomic scattering potential per protein
species, and an analytic bilayer per membrane.

**Proteins are placed as chains, not scattered independently, and
exported as picking ground truth.** Placement follows self-avoiding
worm-like chains, so instances carry polymer membership rather than being
independent draws. Polymer indices are unique within a species.
``export_picks()`` writes per-species annotations in the CryoET Data
Portal and copick point-pick schema, with coordinates in physical
Ångström — directly usable as ground truth for training or benchmarking a
particle picker.

.. note::
   Scope is deliberately limited to cytosolic proteins and membranes.
   Membrane-embedded proteins and solvent filling are not handled.

**Imaging it:**

.. code-block:: bash

   python demo-scripts/generate_tilt_series.py \
       --mrc_path /path/to/tomo.mrc \
       --voxel_size 3.0 \
       --min_tilt_angle -45 --max_tilt_angle 45 --n_tilts 61 \
       --dose_per_tilt 3.0 \
       --defocus 22000 \
       --ice_model gd \
       --device cuda:0

Dose per tilt is typically only a few electrons per Å² — a single tilt
carries very little signal on its own; this is expected and matches real
data.

.. list-table:: Arguments
   :widths: 24 18 58
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--mrc_path``
     - *(required)*
     - Path to input MRC volume (Z, Y, X)
   * - ``--voxel_size``
     - ``3.0``
     - Voxel size in Å
   * - ``--micrograph_size``
     - volume XY size
     - Output image size in pixels (square)
   * - ``--energy``
     - ``300.0``
     - Beam energy in keV
   * - ``--dose_per_tilt``
     - ``3.0``
     - Dose per tilt angle in e⁻/Å²
   * - ``--num_frames``
     - ``10``
     - Number of movie frames per tilt
   * - ``--cs``
     - ``2.0``
     - Spherical aberration in mm
   * - ``--alpha``
     - ``0.1``
     - Amplitude contrast ratio
   * - ``--convergence_angle``
     - ``None``
     - Beam convergence semi-angle in mrad; enables the Cs (spatial
       coherence) envelope
   * - ``--cc``
     - ``None``
     - Chromatic aberration coefficient in mm; enables the Cc (temporal
       coherence) envelope
   * - ``--energy_spread``
     - ``0.7``
     - FWHM of beam energy spread in eV, used by the Cc envelope
   * - ``--deltaV_V``
     - ``0.06e-6``
     - Relative high-voltage instability, used by the Cc envelope
   * - ``--deltaI_I``
     - ``0.01e-6``
     - Relative objective-lens current instability, used by the Cc envelope
   * - ``--dose_envelope``
     - ``False``
     - Apply the Grant & Grigorieff (2015) cumulative-dose envelope
   * - ``--defocus``
     - ``22000.0``
     - Defocus in Å (positive = underfocus)
   * - ``--min_tilt_angle``
     - ``-45.0``
     - Minimum tilt angle in degrees
   * - ``--max_tilt_angle``
     - ``45.0``
     - Maximum tilt angle in degrees
   * - ``--n_tilts``
     - ``61``
     - Number of tilt angles (evenly spaced)
   * - ``--tilt_axis``
     - ``y``
     - ``x`` \| ``y``
   * - ``--scattering_model``
     - ``multislice``
     - ``multislice`` \| ``firstborn`` \| ``projection`` \| ``ctf``
   * - ``--noise_model``
     - ``poisson``
     - ``poisson`` \| ``none``
   * - ``--coincidence_radius``
     - ``1.5``
     - Coincidence radius in Å for direct-detector modelling
   * - ``--add_ice``
     - ``True``
     - Generate and blend amorphous ice
   * - ``--ice_model``
     - ``gd``
     - ``gd`` (samples from the pre-generated ``IceBank`` cache) \|
       ``random`` (instant, cheap ``RandomIcemaker`` placement)
   * - ``--ice_cache_dir``
     - ``None``
     - Directory of cached ice configs for ``ice_model='gd'``. Defaults to
       the bundled ``ice-data/ice_cache``
   * - ``--tomo_to_ice_ratio``
     - ``0.75``
     - Scale factor for tomogram intensity relative to ice
   * - ``--normalize``
     - ``False``
     - Normalise each tilt image to zero mean and unit std
   * - ``--save_exitwaves``
     - ``False``
     - Save exit wave magnitude and phase as ``.mrcs``
   * - ``--device``
     - ``cuda``
     - ``cpu`` \| ``cuda`` \| ``cuda:0``
   * - ``--output_dir``
     - ``./output/``
     - Output directory
   * - ``--filename``
     - ``tilt_series``
     - Base name for output files (no extension)

.. list-table:: Output files
   :widths: 35 65
   :header-rows: 1

   * - File
     - Description
   * - ``<filename>.mrcs``
     - Tilt series stack ``(n_tilts, H, W)``
   * - ``<filename>_exitwave_magnitude.mrcs``
     - Icy exit wave magnitude per tilt (``--save_exitwaves True``, ``--add_ice True``)
   * - ``<filename>_exitwave_phase.mrcs``
     - Icy exit wave phase per tilt (``--save_exitwaves True``, ``--add_ice True``)
   * - ``<filename>_clean_exitwave_magnitude.mrcs``
     - Iceless exit wave magnitude per tilt (``--save_exitwaves True``, ``--add_ice False``)
   * - ``<filename>_clean_exitwave_phase.mrcs``
     - Iceless exit wave phase per tilt (``--save_exitwaves True``, ``--add_ice False``)

**Matching a real tilt series.** ``specter.aretomo3`` reads AreTomo3
alignment files. ``tilt_to_quaternions`` converts the ``TILT`` and ``ROT``
columns of an ``.aln`` file into quaternions for
``TiltSeriesGenerator(quaternions=...)``. This matters because the
simpler ``angles=`` interface assumes the tilt axis is exactly horizontal
or vertical — on real data it never is. Readers for global shifts and
``.xf`` files are alongside it.

**Per-tilt defocus correction.** As the specimen tilts, the beam
traverses a longer path through it, and the Z extent the wave propagates
through grows. The generator recomputes that extent per tilt and shifts
the defocus so the CTF stays referenced to the specimen midplane rather
than drifting with tilt angle.
