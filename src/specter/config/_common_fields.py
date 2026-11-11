"""
Shared `setting()` factories for fields several commands declare alike.

Every command that writes output declares ``output_dir``, ``project``,
``job_id`` and ``seed``, and what those fields mean is the same everywhere
(see `specter.pipelines._common.resolve_output_dir`). Declaring them once
here keeps their ``--help`` text from drifting apart between commands, while
each factory still takes the per-command detail -- what the command writes,
its job-type folder, whether it runs under multi-GPU dispatch -- as an
argument. The simulating commands' physics fields follow: those that need a
per-command note take it as an argument, the rest are declared verbatim.
A field whose default or meaning genuinely differs between commands stays in
its own config.
"""

from __future__ import annotations

from typing import Any

from .._defaults import CC_MM, DELTA_I_I, DELTA_V_V, ENERGY_SPREAD_EV
from ._field import setting
from ._paths import default_pdb_cache_dir


def output_dir_setting(writes: str, job_type: str) -> Any:
    """
    The ``output_dir`` field of a command whose tracking is opt-in.

    Parameters
    ----------
    writes : str
        What the command does in that directory, completing "Directory to
        ...", e.g. ``"save .mrcs and .star files"``.
    job_type : str
        The command's job-type folder, which is also its untracked default
        (`default_output_dir`), e.g. ``"particles"``.

    Returns
    -------
    dataclasses.Field
    """
    return setting(
        None,
        help=(
            f"Directory to {writes} when untracked. Setting --project or "
            "--job_id instead makes this the root of the numbered job tree, so "
            "tracking organises output within the folder you chose rather than "
            f"moving it elsewhere. Unset defaults to {job_type}/ untracked, and "
            "to the project root found by walking up from cwd for an existing "
            ".specter marker when tracked."
        ),
    )


def project_setting(job_type: str, note: str = "") -> Any:
    """
    The ``project`` field of a command whose tracking is opt-in.

    Parameters
    ----------
    job_type : str
        The command's job-type folder, e.g. ``"particles"``.
    note : str, optional
        A command-specific sentence appended to the help.

    Returns
    -------
    dataclasses.Field
    """
    help_text = (
        "Optional: number and track this run through specter.jobs. Not "
        "required for tracking -- job_id alone also triggers it. The run lands "
        f"in <output_dir>/[<project>/]{job_type}/J00N/ with a job.json recording "
        "every parameter, the git commit and the run's status."
    )
    return setting(None, help=f"{help_text} {note}" if note else help_text)


def job_id_setting(note: str = "") -> Any:
    """
    The ``job_id`` field shared by every tracked command.

    Parameters
    ----------
    note : str, optional
        A command-specific sentence appended to the help.

    Returns
    -------
    dataclasses.Field
    """
    help_text = (
        "Pin the job directory (e.g. J001) rather than auto-assigning the next "
        "one: resumes into it if it exists, creates it otherwise."
    )
    return setting(None, help=f"{help_text} {note}" if note else help_text)


def seed_setting(draws: str, unset: str = "Auto-generated and logged if unset.") -> Any:
    """
    The ``seed`` field of a simulating command.

    Parameters
    ----------
    draws : str
        What the seed controls, completing "RNG seed for ...".
    unset : str, optional
        What happens when no seed is given.

    Returns
    -------
    dataclasses.Field
    """
    return setting(None, help=f"RNG seed for {draws}. {unset}")


# --- Absorption and exposure physics ---------------------------------------
#
# Shared by the particle, micrograph and tilt-series configs, which model the
# same physics through the same settings groups (`Propagation`, `Optics`,
# `Envelopes`, `Ice`), so the fields read the same in every command's help.


def absorption_model_setting() -> Any:
    """The ``absorption_model`` field."""
    return setting(
        "alpha",
        help=(
            "Where the imaginary potential comes from. 'alpha' scales the real "
            "potential by the amplitude-contrast ratio, tying absorption to every "
            "atomic cusp. 'inelastic_mfp' derives it per material from a measured "
            "inelastic mean free path instead, and ignores alpha (including a "
            ".cs/.star file's, which CTF estimation takes as an input and never "
            "fits)."
        ),
    )


def inelastic_mfp_solvent_setting() -> Any:
    """The ``inelastic_mfp_solvent`` field, in Angstrom."""
    return setting(
        None,
        help=(
            "Inelastic mean free path of the ice, in Angstrom, for "
            "absorption_model='inelastic_mfp'. Unset takes the measured value for "
            "the run's voltage (3950 at 300 kV, 2030 at 120 kV), and elsewhere an "
            "estimate from those two with a warning (200 kV: 3040 +/- 7%; 100 kV: "
            "1730 +/- 20%)."
        ),
        check="positive",
    )


def inelastic_mfp_specimen_setting() -> Any:
    """The ``inelastic_mfp_specimen`` field, in Angstrom."""
    return setting(
        None,
        help=(
            "Inelastic mean free path of the specimen, in Angstrom, for "
            "absorption_model='inelastic_mfp'. Unset gives the specimen the ice's "
            "value, so it absorbs like the water it displaces and carries no "
            "absorption contrast. 2460 is the derived value for protein at 300 kV."
        ),
        check="positive",
    )


def objective_aperture_setting() -> Any:
    """The ``objective_aperture`` field, in milliradians."""
    return setting(
        None,
        help=(
            "Objective aperture semi-angle in milliradians (a 70 um aperture on a "
            "Krios is ~12 mrad), for absorption_model='inelastic_mfp'. Electrons "
            "scattered elastically beyond it leave the image; the loss is charged "
            "per material from the scattering cross section, since no practical "
            "grid carries it (2.8% of the beam through 400 Angstrom of ice at "
            "12 mrad, 300 kV). Unset: no aperture."
        ),
        check="positive",
    )


def dose_envelope_target_setting(owner: str, note: str = "") -> Any:
    """
    The ``dose_envelope_target`` field.

    Parameters
    ----------
    owner : str
        What is damaged, e.g. ``"particle"`` or ``"specimen"``.
    note : str, optional
        A command-specific sentence appended to the help.

    Returns
    -------
    dataclasses.Field
    """
    help_text = (
        "Where the dose envelope acts. 'transfer_function' filters the whole "
        f"image, solvent included. 'specimen' damages the {owner}'s own "
        "potential before the ice is added, with occupancy read from the "
        f"undamaged {owner}, so the water keeps its 3.7 A ring (raw movies "
        "show it does not fade with dose); required for ice_motion_variance."
    )
    return setting(
        "transfer_function", help=f"{help_text} {note}" if note else help_text
    )


def ice_motion_variance_setting(note: str = "") -> Any:
    """
    The ``ice_motion_variance`` field, in A^2 per e-/A^2.

    Parameters
    ----------
    note : str, optional
        A command-specific sentence appended to the help.

    Returns
    -------
    dataclasses.Field
    """
    help_text = (
        "Beam-induced displacement of the water, per axis, in A^2 per e-/A^2 "
        "(McMullan et al. 2015's sigma0^2, 0.38 for their 300 kV exposure, "
        "as measured from the 3.7 A ring). The ice fluctuation is filtered to "
        "what survives the summed exposure, frame weights included, using a "
        "decorrelation measured on relaxed ice trajectories; its mean is kept. "
        "With dose_envelope on, needs dose_envelope_target='specimen'. Unset: "
        "frozen ice."
    )
    return setting(
        None, help=f"{help_text} {note}" if note else help_text, check="non_negative"
    )


# --- Fields the simulating commands share verbatim -------------------------
#
# Lens, camera, scattering and structure-input fields that
# `ParticleStackConfig`, `MicrographConfig` and `TiltSeriesConfig` (and,
# for the structure inputs, `TomogramConfig`) declare with the same default
# and help.


def pdb_source_setting() -> Any:
    """The ``pdb_source`` field."""
    return setting(
        help=(
            "Path to a local .cif/.pdb file, or a 4-character PDB "
            "accession code to fetch and cache. A local file is read where it "
            "lies and never copied into the cache; an existing file wins over a "
            "same-named accession code."
        )
    )


def assembly_setting() -> Any:
    """The ``assembly`` field."""
    return setting(True, help="Fetch the biological assembly.")


def pixel_size_setting() -> Any:
    """The ``pixel_size`` field."""
    return setting(1.0, help="Pixel size in Angstrom.", check="positive")


def voltage_setting() -> Any:
    """The ``voltage`` field."""
    return setting(
        300.0, help="Electron beam accelerating voltage in kV.", check="positive"
    )


def cs_setting() -> Any:
    """The ``cs`` field."""
    return setting(
        2.0, help="Spherical aberration in mm (1-3 mm typical).", check="non_negative"
    )


def alpha_setting() -> Any:
    """The ``alpha`` field."""
    return setting(0.1, help="Amplitude contrast ratio.", range=(0.0, 1.0))


def scattering_model_setting() -> Any:
    """The ``scattering_model`` field."""
    return setting("multislice", help="Scattering model.")


def detector_model_setting() -> Any:
    """The ``detector_model`` field."""
    return setting("none", help="Detector model.")


def save_exitwaves_setting() -> Any:
    """The ``save_exitwaves`` field."""
    return setting(
        False, help="Save exit wave magnitude and phase as separate .mrcs files."
    )


def save_clean_exitwaves_setting() -> Any:
    """The ``save_clean_exitwaves`` field."""
    return setting(
        False,
        help="Save clean (particle-only, no ice) exit wave magnitude and phase.",
    )


def pdb_cache_dir_setting() -> Any:
    """The ``pdb_cache_dir`` field."""
    return setting(
        factory=default_pdb_cache_dir,
        help=(
            "Where downloaded PDB/mmCIF structures are cached. An "
            "input cache shared by every run, not an output location -- job tracking "
            "does not redirect it."
        ),
    )


def convergence_angle_setting() -> Any:
    """The ``convergence_angle`` field."""
    return setting(
        None,
        help=(
            "Beam convergence semi-angle in mrad, for the Cs "
            "(spatial coherence) envelope."
        ),
        check="non_negative",
    )


def cc_setting() -> Any:
    """The ``cc`` field."""
    return setting(
        CC_MM,
        help=(
            "Chromatic aberration coefficient in mm, for the temporal coherence "
            "loss from the beam's energy spread. Default 2.7, the Titan Krios "
            "objective; 0 disables it. On by default since every real image "
            "carries it, filtered or not."
        ),
        check="non_negative",
    )


def energy_spread_setting() -> Any:
    """The ``energy_spread`` field."""
    return setting(
        ENERGY_SPREAD_EV,
        help="FWHM of the beam energy spread in eV, used by the Cc envelope.",
        check="non_negative",
    )


def deltaV_V_setting() -> Any:
    """The ``deltaV_V`` field."""
    return setting(
        DELTA_V_V,
        help="Relative high-voltage instability, used by the Cc envelope.",
        check="non_negative",
    )


def deltaI_I_setting() -> Any:
    """The ``deltaI_I`` field."""
    return setting(
        DELTA_I_I,
        help="Relative objective-lens current instability, used by the Cc envelope.",
        check="non_negative",
    )


def temporal_coherence_setting() -> Any:
    """The ``temporal_coherence`` field."""
    return setting(
        "integrated",
        help=(
            "How the Cc focal spread is applied when cc is set: 'integrated' sums the image incoherently over the defocus spread (exact), 'envelope' multiplies the transfer function by the Cc envelope (exact only for the part linear in the scattered wave; 0.4% low in mean intensity in 2000 A of ice at 300 kV, 5% at 120 kV)."
        ),
    )


def dose_envelope_setting() -> Any:
    """The ``dose_envelope`` field."""
    return setting(
        False, help="Apply the Grant & Grigorieff (2015) cumulative-dose envelope."
    )


def bfactor_setting() -> Any:
    """The ``bfactor`` field."""
    return setting(
        None, help="Isotropic B-factor envelope in Angstrom^2.", check="non_negative"
    )


def noise_model_setting() -> Any:
    """The ``noise_model`` field."""
    return setting("poisson", help="Noise model. Use 'none' for no noise.")


def ice_cache_dir_setting() -> Any:
    """The ``ice_cache_dir`` field."""
    return setting(
        None,
        help=(
            "Directory of cached ice configs for ice_model='gd'. "
            "Defaults to the bundled ice_data/ice_cache."
        ),
    )


def crowd_max_distance_z_setting() -> Any:
    """The ``crowd_max_distance_z`` field."""
    return setting(
        None,
        help="Maximum z-distance between crowded particles in Angstrom.",
        check="non_negative",
    )


def scattering_factors_setting() -> Any:
    """The ``scattering_factors`` field."""
    return setting(
        "shtyrov",
        help=(
            "Atomic scattering-factor parameterization used to "
            "build the structure's scattering potential."
        ),
    )


def monomer_library_path_setting() -> Any:
    """The ``monomer_library_path`` field."""
    return setting(
        None,
        help=(
            "Path to a Monomer Library "
            "(https://github.com/MonomerLibrary/monomers), which completes a "
            "structure's bond topology and hydrogens so Shtyrov species resolve. "
            "Unset falls back to $CLIBD_MON. Without one, around 44% of a "
            "hydrogen-free protein falls back to per-element Peng factors."
        ),
    )


def klim_setting() -> Any:
    """The ``klim`` field."""
    return setting(
        None,
        help=(
            "Bandlimit for Kirkland's FFT anti-aliasing, as a fraction of "
            "Nyquist. Kirkland recommends 0.66 (2/3), which prevents aliasing but "
            "discards real spatial frequency content above it. Unset (the default) "
            "keeps the full Nyquist range and accepts the aliasing."
        ),
        check="non_negative",
    )


def oversample_setting() -> Any:
    """The ``oversample`` field."""
    return setting(
        2,
        help=(
            "Grid refinement for the multislice recursion, the Fourier-space counterpart of pad_fft: slices are interpolated onto an xy grid this many times finer (same frame, higher Nyquist), so the transmission-times-wave product no longer wraps at Nyquist; the image is formed on that grid as well and its intensity sampled to the pixels. 1 reproduces the volume-grid recursion (0.2-1.1% of the scattered power in error, 1.9% at Nyquist); 2 is at the float32 noise floor for ~3x the multislice time. Also applied to Rytov's and projection's final exponential, which aliases identically, at negligible cost; first Born and ctf are linear and unaffected."
        ),
        check="positive",
    )


def bulk_scattering_factors_setting() -> Any:
    """The ``bulk_scattering_factors`` field."""
    return setting(
        "kirkland",
        help=(
            "Atomic scattering-factor parameterization for the ice -- everything rendered that is not a biomolecule. Deliberately separate from scattering_factors: Shtyrov is fitted for biomolecules, and these materials are outside that domain."
        ),
    )
