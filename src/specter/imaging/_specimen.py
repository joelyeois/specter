"""
`Structure` (an atomic model) and `Specimen` (what the beam passes through).

A `Specimen` is a recipe: a template potential plus what to build around it,
recorded by :meth:`Specimen.crowd`, :meth:`Specimen.solvate` and
:meth:`Specimen.damage`. :meth:`Specimen.scatter` builds each image's own
specimen from it, in the order physics fixes, and propagates the beam through
it. The building is :mod:`specter.imaging._stages`; this module only orders it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace
from typing import Any, cast

import roma
import torch

from .. import rotations
from ..aberrations._functions import defocus_midplane_shift
from ..arrays import compute_nz, pad_volume
from ..cpu_threads import limited_cpu_threads
from ..crowding import CrowdWithDuplicates
from ..ice import IceBank, RandomIcemaker, resolve_icemaker
from ..options import ConvBackend, PotentialMethod, ScatteringFactors
from ..pdb import DEFAULT_PDB_CACHE_DIR, PDB
from ..potential import (
    PotentialBuilder,
    absorption_potential,
    aperture_lowpass,
    apply_dose_damage,
    molecular_mass_from_atoms,
    template_occupancy_reference,
)
from ..rotations import VolumeRotator, translate_coordinates
from ..scattering import surface_slice_sum
from ..settings import Camera, Crowding, Ice, Optics, Propagation
from . import _stages as stages
from ._data import Exposure, Poses, settings_group
from ._engines import scattering_engine
from ._waves import ExitWave

__all__ = ["Specimen", "Structure"]


def _seeds(seed: int | None, ids: torch.Tensor, stage: int) -> list[int] | None:
    """One derived seed per image for `stage`, or None for the global RNG."""
    if seed is None:
        return None
    from ..random_seed import derive_seed

    return [derive_seed(seed, int(i), stage) for i in ids.tolist()]


@dataclass(frozen=True)
class Structure:
    """
    An atomic model, to be rendered into a potential.

    Attributes
    ----------
    pdb : PDB
        The parsed model: coordinates, elements and, for Shtyrov factors,
        bonded-species types.
    scattering_factors : str
        The parameterization the model was typed for and renders with.
    """

    pdb: PDB
    scattering_factors: ScatteringFactors = "shtyrov"

    @classmethod
    def from_pdb(
        cls,
        source: str,
        *,
        assembly: bool = True,
        scattering_factors: ScatteringFactors = "shtyrov",
        pdb_cache_dir: str | None = None,
        readd_hydrogens: bool | str = "auto",
        monomer_library_path: str | None = None,
    ) -> Structure:
        """
        Read a model from an accession code or a local file.

        Shtyrov factors [1]_ are fitted per bonded species, so the model is
        typed from its bond topology only when they are asked for; the other
        parameterizations are per element.

        References
        ----------
        .. [1] Shtyrov, A., Wilson, H., Slowik, D., Yamashita, K., Li, J., Wojdyr, M.,
           et al. (2026). Measurement of atomic scattering factors by cryoelectron
           microscopy. *Proceedings of the National Academy of Sciences*, 123(19),
           e2528758123. https://doi.org/10.1073/pnas.2528758123
        """
        pdb = PDB(
            source,
            assembly=assembly,
            pdb_cache_dir=pdb_cache_dir or DEFAULT_PDB_CACHE_DIR,
            compute_atom_species=scattering_factors == "shtyrov",
            readd_hydrogens=readd_hydrogens,
            monomer_library_path=monomer_library_path,
        )
        return cls(pdb, scattering_factors)

    @property
    def molecular_mass(self) -> float:
        """In daltons, hydrogens included."""
        return molecular_mass_from_atoms(self.pdb.atomic_numbers)

    def render(
        self,
        voxel_size: float,
        box: int,
        *,
        method: PotentialMethod = "analytic",
        conv_backend: ConvBackend = "fftconvolve",
        atom_species: list[str] | None = None,
        shtyrov_params_path: str | None = None,
        rcut: float | None = None,
        use_deposited_bfactors: bool = False,
        periodic: bool = False,
        device: torch.device | str = "cpu",
    ) -> Specimen:
        """
        The model's potential in a ``box``-voxel cube, as a `Specimen`.

        Rendered on `device` and returned there; the analytic renderer is a
        loop of small ops per atom, an order of magnitude faster on a GPU.
        """
        builder = PotentialBuilder(
            box,
            voxel_size,
            self.pdb.atomic_numbers,
            parameterization=self.scattering_factors,
            conv_backend=conv_backend,
            atom_species=atom_species or self.pdb.atom_species,
            shtyrov_params_path=shtyrov_params_path,
            rcut=rcut,
            b_factors=self.pdb.b_factors if use_deposited_bfactors else None,
            periodic=periodic,
        ).to(device)
        with torch.no_grad():
            V = builder(self.pdb.coordinates.to(device), method=method).clone()
        return Specimen(V, voxel_size, molecular_mass=self.molecular_mass)


@dataclass(frozen=True)
class Atoms:
    """
    Atoms rendered afresh at every pose, for a specimen whose coordinates are
    themselves the variable (refined, or differentiated through).

    Attributes
    ----------
    coordinates : torch.Tensor
        ``(n, 3)`` in Angstrom; gradients reach them through the render.
    builder : PotentialBuilder
        Renders them into the specimen's box.
    """

    coordinates: torch.Tensor
    builder: PotentialBuilder


@dataclass(frozen=True)
class Specimen:
    """
    What the beam passes through: a template, and how to build around it.

    A recipe rather than a volume: :meth:`crowd`, :meth:`solvate` and
    :meth:`damage` record what to do, and :meth:`scatter` does it for each
    image, drawing that image's own neighbours and ice. :meth:`plan` lists
    the steps and :meth:`materialize` returns the volumes exactly as they
    would be propagated.

    Attributes
    ----------
    potential : torch.Tensor
        The template's potential, ``(Z, Y, X)``, in volts.
    voxel_size : float
        In Angstrom.
    molecular_mass : float, optional
        In daltons; sets how much ice the template displaces. Without it the
        template is read against protein's mean inner potential.
    crowding : Crowding, optional
        Set by :meth:`crowd`.
    crowd_on_cpu : bool
        Assemble the neighbours on the host, trading speed for device memory.
    ice : Ice, optional
        Set by :meth:`solvate`; its ``thickness`` sets the box depth.
    icemaker : IceBank or RandomIcemaker, optional
        The ice library, resolved once by :meth:`solvate`.
    damaged : bool
        Set by :meth:`damage`.
    absorbs : bool
        Whether the mean-free-path absorption and the objective aperture act;
        cleared by :meth:`dry`.
    """

    potential: torch.Tensor
    voxel_size: float
    molecular_mass: float | None = None
    crowding: Crowding | None = None
    crowd_on_cpu: bool = False
    ice: Ice | None = None
    icemaker: IceBank | RandomIcemaker | None = field(default=None, repr=False)
    damaged: bool = False
    absorbs: bool = True
    atoms: Atoms | None = field(default=None, repr=False)

    @classmethod
    def from_atoms(
        cls,
        coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor,
        voxel_size: float,
        box: int,
        *,
        conv_backend: ConvBackend = "fftconvolve",
        periodic: bool = False,
    ) -> Specimen:
        """
        A specimen rendered from its atoms at every pose.

        The atoms are rotated rather than the potential, so the image is a
        differentiable function of `coordinates`. Bare coordinates carry no
        bonds, so every atom takes its element's scattering factors. The
        unrotated render is the template that crowding copies and that sets
        how much ice the specimen displaces.
        """
        builder = PotentialBuilder(
            box,
            voxel_size,
            atomic_numbers,
            conv_backend=conv_backend,
            periodic=periodic,
        ).to(coordinates.device)
        with torch.no_grad():
            template = builder(coordinates.detach())
        return cls(
            template,
            voxel_size,
            molecular_mass=molecular_mass_from_atoms(atomic_numbers),
            atoms=Atoms(coordinates, builder),
        )

    @property
    def nxy(self) -> int:
        """Lateral size of the box, in voxels."""
        return self.potential.shape[-1]

    @property
    def nz(self) -> int:
        """Depth of the box, grown to hold the ice."""
        thickness = None if self.ice is None else self.ice.thickness
        return compute_nz(self.potential.shape[0], thickness, self.voxel_size)

    def pad_nxy(self, pad_fft: bool) -> int:
        """Lateral size of the propagation box, with an ``nxy // 2`` FFT margin."""
        return self.nxy + (self.nxy // 2) * 2 if pad_fft else self.nxy

    def crowd(
        self,
        crowding: Crowding | None = None,
        *,
        on_cpu: bool = False,
        **settings: Any,
    ) -> Specimen:
        """
        Pack copies of the template around it, drawn afresh for every image.

        Takes a `Crowding`, or its fields as keywords. ``max_distance_z``
        defaults to the template's own depth, not the ice's: thicker ice is a
        contrast setting, not a denser specimen.
        """
        crowding = settings_group(Crowding, crowding, settings)
        if crowding.min_distance is None:
            raise ValueError("crowd() needs min_distance")
        return replace(self, crowding=crowding, crowd_on_cpu=on_cpu)

    def solvate(
        self,
        ice: Ice | None = None,
        *,
        icemaker: IceBank | RandomIcemaker | None = None,
        **settings: Any,
    ) -> Specimen:
        """
        Embed it in amorphous ice, drawn afresh for every image.

        Takes an `Ice`, or its fields as keywords, and loads the ice library
        here, once. A prebuilt `icemaker` is used as given.
        """
        specimen = replace(self, ice=settings_group(Ice, ice, settings))
        assert specimen.ice is not None
        maker = resolve_icemaker(
            specimen.ice.model,
            self.voxel_size,
            self.nxy,
            specimen.nz,
            ice_cache_dir=specimen.ice.cache_dir,
            icemaker=icemaker,
            parameterization=specimen.ice.parameterization,
            progressbars=False,
        )
        if maker is None:
            raise ValueError("solvate() needs an ice model ('gd' or 'random')")
        if maker.device != self.potential.device:
            maker = maker.to(self.potential.device)
        return replace(specimen, icemaker=maker)

    def damage(self) -> Specimen:
        """
        Apply each exposure's dose envelope to the specimen, before the ice.

        References
        ----------
        .. [1] Grant, T., & Grigorieff, N. (2015). Measuring the optimal exposure for
           single particle cryo-EM using a 2.6 Å reconstruction of rotavirus VP6.
           *eLife*, 4, e06980. https://doi.org/10.7554/eLife.06980
        """
        return replace(self, damaged=True)

    def dry(self) -> Specimen:
        """
        The same specimen with no ice, no damage and no absorption, in the
        same box.

        Scattered with the same poses and seed it draws the same neighbours,
        so it is the reference those effects are measured against. Only the
        amplitude contrast (``alpha``) remains, as it does in every image.
        """
        return replace(self, icemaker=None, damaged=False, absorbs=False)

    def to(self, device: torch.device | str) -> Specimen:
        """The same specimen, on `device`."""
        icemaker = None if self.icemaker is None else self.icemaker.to(device)
        return replace(self, potential=self.potential.to(device), icemaker=icemaker)

    def plan(
        self,
        voltage: float = 300.0,
        *,
        propagation: Propagation | None = None,
        **settings: Any,
    ) -> list[str]:
        """What :meth:`scatter` will do to each image, in order."""
        p = settings_group(Propagation, propagation, settings)
        nz, nxy, pad = self.nz, self.nxy, self.pad_nxy(p.pads(False))
        if p.rotate_mode == "surface":
            steps = [
                f"sample the {nxy}^3 template's transform on the posed "
                f"{'plane' if p.scattering_model in ('projection', 'ctf') else 'Ewald surface'}"
            ]
        else:
            steps = [
                f"rotate the {nxy}^3 template to the image's pose ({p.rotate_mode})"
            ]
        steps.append(f"pad to {nz} x {pad} x {pad} voxels")
        if self.crowding is not None:
            steps.append(
                f"crowd: copies {self.crowding.min_distance:.1f} A apart "
                f"({self.crowding.method} placement)"
            )
        if p.absorption_model == "inelastic_mfp":
            steps.append("absorb: mean-free-path field read off the dry specimen")
        if self.damaged:
            steps.append("damage: dose envelope on the specimen, occupancy read first")
        if self.icemaker is not None:
            assert self.ice is not None
            steps.append(f"solvate: {self.ice.model} ice, {nz * self.voxel_size:.0f} A")
        steps.append(
            f"scatter: {p.scattering_model} at {voltage:g} kV, "
            f"oversample {p.oversample}, alpha {p.alpha}"
        )
        return steps

    def materialize(
        self,
        poses: Poses,
        *,
        voltage: float,
        exposure: Exposure | None = None,
        seed: int | None = None,
        objective_aperture: float | None = None,
        propagation: Propagation | None = None,
        **settings: Any,
    ) -> torch.Tensor:
        """
        The volumes :meth:`scatter` would propagate, ``(n, Z, Y, X)``.

        Takes :meth:`scatter`'s arguments and draws the same specimens.
        """
        p = settings_group(Propagation, propagation, settings)
        return self._volume(poses, voltage, exposure, seed, objective_aperture, p)[0]

    def scatter(
        self,
        poses: Poses,
        *,
        voltage: float,
        exposure: Exposure | None = None,
        seed: int | None = None,
        objective_aperture: float | None = None,
        propagation: Propagation | None = None,
        **settings: Any,
    ) -> ExitWave:
        """
        Build each image's specimen and propagate the beam through it.

        Parameters
        ----------
        poses : Poses
            One pose per image.
        voltage : float
            Accelerating voltage in kV.
        exposure : Exposure, optional
            Required by :meth:`damage` and by ``Ice(motion_variance=...)``.
        seed : int, optional
            Run seed; each image's neighbours and ice are drawn from
            ``(seed, poses.ids[i], stage)``. Default None, the global RNG.
        objective_aperture : float, optional
            Aperture semi-angle in mrad, charged as absorption; requires
            ``absorption_model="inelastic_mfp"``.
        propagation : Propagation, optional
            How the exit wave is computed, or its fields as keywords
            (``scattering_model``, ``alpha``, ``absorption_model``,
            ``oversample``, ``pad_fft``, ...).

        Returns
        -------
        ExitWave
            On the propagation grid, ``oversample`` times finer than the
            pixels, or on the pixel grid for the models that never leave it.
        """
        p = settings_group(Propagation, propagation, settings)
        device = self.potential.device
        uniform = 0.0
        if p.absorption_model == "inelastic_mfp" and p.inelastic_mfp_specimen is None:
            if self.icemaker is not None:
                rates = stages.resolve_removal_mfps(
                    p, Optics(objective_aperture=objective_aperture), voltage
                )
                assert rates is not None
                uniform = absorption_potential(rates.removal("solvent"), voltage)
        pad_nxy = self.pad_nxy(p.pads(False))
        scattering = scattering_engine(
            pad_nxy,
            self.voxel_size,
            voltage,
            p.scattering_model,
            p.klim,
            p.oversample,
            p.ews_curvature_sign,
            self.nz,
            p.alpha,
            uniform,
            str(device),
        )
        if p.rotate_mode == "surface":
            self._check_surface(objective_aperture)
            spectrum = surface_slice_sum(
                self.potential,
                poses.quaternions,
                poses.shifts,
                pad_nxy=pad_nxy,
                pixel_size=self.voxel_size,
                wavelength=scattering.wavelength,
                ews_curvature_sign=p.ews_curvature_sign,
                flat=p.scattering_model in ("projection", "ctf"),
            )
            wave = scattering.from_slice_sum(spectrum, self.nz)
            del spectrum
        else:
            V, v_ab = self._volume(
                poses, voltage, exposure, seed, objective_aperture, p
            )
            wave = scattering(V, absorption_potential=v_ab)
            del V, v_ab
        linear = p.scattering_model == "ctf"
        has_depth = p.scattering_model not in ("projection", "ctf")
        return ExitWave(
            array=wave,
            pixel_size=self.voxel_size,
            voltage=voltage,
            nxy=self.nxy,
            pad_nxy=pad_nxy,
            oversample=1 if linear else p.oversample,
            defocus_shift=(
                defocus_midplane_shift(self.nz, self.voxel_size) if has_depth else 0.0
            ),
            ids=poses.ids,
            linear=linear,
            alpha=p.alpha,
            damaged=self.damaged,
            solvent_filtered=self._filters_solvent,
        )

    def _check_surface(self, objective_aperture: float | None) -> None:
        """Refuse what ``rotate_mode="surface"`` cannot do: it never forms the posed volume."""
        dressed = {
            "crowding": self.crowding is not None,
            "ice": self.icemaker is not None,
            "damage": self.damaged,
            "per-image atom rendering": self.atoms is not None,
            "an objective aperture": objective_aperture is not None,
        }
        present = [name for name, on in dressed.items() if on]
        if present:
            raise ValueError(
                "rotate_mode='surface' samples the bare template's transform and "
                f"cannot apply {', '.join(present)}, which act on the posed volume; "
                "use rotate_mode='real'"
            )

    def _volume(
        self,
        poses: Poses,
        voltage: float,
        exposure: Exposure | None,
        seed: int | None,
        objective_aperture: float | None,
        p: Propagation,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Each image's specimen, ``(n, Z, Y, X)``, and its absorption field.

        The order is the one physics fixes: the absorption field and the
        occupancy are read off the dry specimen, the dose envelope acts
        before the ice is added so the solvent keeps its structure, and the
        aperture filters what will be propagated.
        """
        if objective_aperture is not None and p.absorption_model != "inelastic_mfp":
            raise ValueError(
                "objective_aperture requires absorption_model='inelastic_mfp': "
                "under 'alpha' the fitted amplitude contrast already stands in "
                "for aperture loss"
            )
        if (self.damaged or self._filters_solvent) and exposure is None:
            raise ValueError("damage() and Ice(motion_variance=...) need an exposure")
        if p.rotate_mode == "surface":
            raise ValueError(
                "rotate_mode='surface' never forms the posed volume; materialize "
                "it with rotate_mode='real'"
            )
        template, device = self.potential, self.potential.device
        nxy, nz, n = self.nxy, self.nz, len(poses)
        thickness = None if self.ice is None else self.ice.thickness
        nz_t, ny_t, nx_t = template.shape
        rotator = VolumeRotator(
            nz_t, ny_t, nx_t, origin="relion", mode=p.rotate_mode
        ).to(device)
        crowd = self._build_crowd(p.pads(False))
        if not self.absorbs:
            objective_aperture = None
        rates = (
            stages.resolve_removal_mfps(
                p, Optics(objective_aperture=objective_aperture), voltage
            )
            if self.absorbs
            else None
        )
        doses = None if exposure is None else exposure.doses(n)
        exposure_filter = self._solvent_exposure_filter(exposure, doses)
        reference: list[torch.Tensor] = []  # solved on first use

        def full() -> torch.Tensor:
            # A voxel reading this many volts of specimen is full.
            if not reference:
                volts = template_occupancy_reference(
                    template, self.voxel_size, self.molecular_mass
                )
                reference.append(torch.full((1, 1, 1, 1), volts, device=device))
            return reference[0]

        # With the volume on the device, the CPU work left is the crowd's and
        # the ice's small-op loops, each of which pays torch's per-op sync
        # cost in proportion to the thread pool; capping once here spares
        # their own scoped caps the ~12 ms pool resize each. Nothing else
        # here wants the cap, and resizing costs ~1.5 ms each way per call,
        # a tenth of a small reconstruction step.
        small_ops = crowd is not None or self.icemaker is not None
        cap = (
            limited_cpu_threads()
            if template.is_cuda and small_ops
            else contextlib.nullcontext()
        )
        with cap:
            if self.atoms is None:
                R = roma.unitquat_to_rotmat(poses.quaternions.to(device))
                T = rotations.translations_angstrom_to_torch(
                    poses.shifts.to(device), nxy, self.voxel_size
                )
                V = rotator(template, rotations.build_affine_matrix(R, T))
            else:
                V = self._render_atoms(poses, device)
            V = pad_volume(V, nxy, nz, thickness, p.pads(False), xy_pad_mode="constant")

            if crowd is not None:
                stages.crowd_into(
                    V, crowd, _seeds(seed, poses.ids, stages.SEED_STAGE_CROWD)
                )

            v_ab = None
            if rates is not None and rates.inelastic_specimen is not None:
                with torch.no_grad():
                    v_ab = stages.mfp_absorption_field(
                        V,
                        self.voxel_size,
                        rates,
                        has_solvent=self.icemaker is not None,
                        full_potential=full(),
                    )

            free = None
            if self.damaged:
                assert exposure is not None and doses is not None
                pre = exposure.pre_exposure
                with torch.no_grad():
                    if self.icemaker is not None:
                        free = stages.free_fraction(V, self.voxel_size, full())
                    V = apply_dose_damage(
                        V,
                        self.voxel_size,
                        doses.to(device),
                        pre_exposure=0.0 if pre is None else pre.to(device),
                        weighted=exposure.weighted,
                        voltage=voltage,
                        n_frames=exposure.n_frames,
                        frame_weights=exposure.weights,
                        frame_weights_max_frequency=exposure.weights_max_frequency,
                    )

            if self.icemaker is not None:
                assert self.ice is not None
                with torch.no_grad():
                    V = stages.solvate(
                        V,
                        self.icemaker,
                        pixel_size=self.voxel_size,
                        nxy=nxy,
                        pad=nxy // 2 if p.pads(False) else 0,
                        full=full(),
                        relax_steps=self.ice.relax_steps,
                        free=free,
                        seeds=_seeds(seed, poses.ids, stages.SEED_STAGE_ICE),
                        exposure_filter=exposure_filter,
                    )
                del free

            if objective_aperture is not None:
                with torch.no_grad():
                    V = aperture_lowpass(
                        V, self.voxel_size, objective_aperture, voltage
                    )
        return V, v_ab

    def _render_atoms(self, poses: Poses, device: torch.device) -> torch.Tensor:
        """
        The atoms at each pose, rendered, ``(n, Z, Y, X)``.

        The inverse rotation applied to the atoms, then the shift subtracted
        in the lab frame: the same origin-offset semantics the volume path
        gets from :func:`~specter.rotations.build_affine_matrix`.
        """
        atoms = cast(Atoms, self.atoms)
        R = roma.unitquat_to_rotmat(poses.quaternions.to(device)).transpose(-2, -1)
        coordinates = atoms.coordinates.to(device)
        rotated = torch.einsum(
            "bij,bkj->bki", R, coordinates.unsqueeze(0).expand(len(R), -1, 3)
        )
        moved = translate_coordinates(rotated, poses.shifts.to(device), inverse=True)
        V = atoms.builder.to(device)(moved)
        return V if V.ndim == 4 else V.unsqueeze(0)

    @property
    def _filters_solvent(self) -> bool:
        return (
            self.icemaker is not None
            and self.ice is not None
            and self.ice.motion_variance is not None
        )

    def _build_crowd(self, pad_fft: bool) -> CrowdWithDuplicates | None:
        c = self.crowding
        if c is None:
            return None
        return CrowdWithDuplicates(
            self.potential,
            self.voxel_size,
            cast(float, c.min_distance),
            nxy_out=self.pad_nxy(pad_fft),
            nz_out=self.nz,
            max_distance_z=(
                c.max_distance_z
                if c.max_distance_z is not None
                else self.potential.shape[0] * self.voxel_size
            ),
            max_distance_xy=c.max_distance_xy,
            method=c.method,
            n_points=c.n_points if c.n_points is not None else torch.inf,
            seed=c.seed,
            move_to_cpu=self.crowd_on_cpu,
            progressbars=False,
            chunk_size=c.chunk_size,
            water_air_interface=c.water_air_interface,
            sigma_frac=c.sigma_frac,
            peak_amplitude=c.peak_amplitude,
            baseline=c.baseline,
        ).to(self.potential.device)

    def _solvent_exposure_filter(
        self, exposure: Exposure | None, doses: torch.Tensor | None
    ) -> Any:
        if not self._filters_solvent:
            return None
        assert self.ice is not None and exposure is not None and doses is not None
        if not torch.allclose(doses, doses[0].expand_as(doses)):
            raise ValueError(
                "Ice(motion_variance=...) needs one dose for every image: the "
                "exposure filter acts on a whole batch's ice canvas at once"
            )
        return stages.solvent_exposure_filter(
            self.ice,
            Camera(n_frames=exposure.n_frames),
            self.voxel_size,
            float(doses[0]),
            exposure.weights,
            exposure.weights_max_frequency,
        )
