"""
`ImageGenerator` and `ImageGeneratorFromCoordinates`: single-particle image
simulation from a pre-built potential or from atomic coordinates.
"""

from __future__ import annotations

from typing import Any

import roma
import torch
import torch.nn as nn

from specter import logger

from .. import rotations
from ..ice import IceBank, RandomIcemaker, resolve_icemaker
from ..potential import PotentialBuilder
from ..rotations import VolumeRotator, translate_coordinates
from ..settings import Camera, Crowding, Envelopes, Ice, Optics, Propagation
from ._base import compute_nz, pad_volume
from ._micrograph import MicrographGenerator
from ._particle_base import ParticleGeneratorBase
from ._tiltseries import TiltSeriesGenerator
from specter.options import ConvBackend


__all__ = [
    "ImageGenerator",
    "ImageGeneratorFromCoordinates",
    "MicrographGenerator",
    "ParticleGeneratorBase",
    "TiltSeriesGenerator",
    "compute_nz",
    "pad_volume",
]


class ImageGeneratorFromCoordinates(ParticleGeneratorBase):
    """
    Generates images from atomic coordinates.

    Particles are defined by atomic coordinates, rotated, and then voxelized into potential volumes.

    Parameters
    ----------
    coordinates : torch.Tensor
        Atomic coordinates. Shape (n_atoms, 3).
    atomic_numbers : torch.Tensor
        Atomic numbers for each atom. Shape (n_atoms,).
    nxy : int
        Image size in pixels.
    pixel_size : float
        Pixel size in Å.
    quaternions : torch.Tensor
        Rotation quaternions for each batch item. Shape (B, 4).
    translations : torch.Tensor
        Translations (x, y) in Å for each batch item. Shape (B, 2).
    ctf_params : dict or None
        Per-image CTF parameters. Required unless ``optics`` is ``None``.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Total electron dose (fluence) per image in e⁻/Å². Scalar, or a 1-D
        tensor of length n giving a separate dose for each image.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    propagation : Propagation, optional
        How the exit wave is computed. Default ``Propagation()``.
    optics : Optics, optional
        The aberration stage; ``None`` skips it. Default ``Optics()``.
    envelopes : Envelopes, optional
        Coherence and radiation-damage envelopes. Default ``Envelopes()``.
    camera : Camera, optional
        The detector chain. Default ``Camera()``.
    ice : Ice, optional
        The amorphous ice the particle is embedded in: model, thickness,
        library, seam relaxation and scattering factors. Default ``Ice()``,
        no ice.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker instance to reuse across generators. When
        supplied, ``ice.model`` and ``ice.cache_dir`` are ignored;
        ``ice.thickness`` still sizes the column.
    crowding : Crowding, optional
        How duplicates of the particle are packed around it. Default
        ``Crowding()``, none. ``max_distance_z`` defaults to the template's
        own depth, which deliberately does **not** follow ``ice.thickness``:
        raising that on a particle stack is reaching for lower contrast and
        more solvent background, not for a denser specimen, so the two stay
        separate knobs. At 2000 Å of ice the neighbours occupy the middle
        of the column and the rest is water. `MicrographSpecimenGenerator`
        scales with ``nz`` instead, and should: a micrograph is a specimen,
        a particle stack is a controlled image-formation experiment.
    conv_backend : str, optional
        Backend for convolution in potential building. Default 'fftconvolve'.
    periodic_potential : bool, optional
        If True, use periodic boundary conditions when voxelizing coordinates
        into the potential. Required when coordinates come from a periodic ice
        generator (e.g. GradientSKIcemaker) to avoid density deficiency at
        the box boundary. Default False.
    bfactor : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.
    """

    def __init__(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor,
        nxy: int,
        pixel_size: float,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        ctf_params: dict[str, Any] | None,
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        anisomag: torch.Tensor | None = None,
        propagation: Propagation = Propagation(),
        optics: Optics | None = Optics(),
        envelopes: Envelopes = Envelopes(),
        camera: Camera = Camera(),
        ice: Ice = Ice(),
        icemaker: IceBank | RandomIcemaker | None = None,
        crowding: Crowding = Crowding(),
        conv_backend: ConvBackend = "fftconvolve",
        verbose: bool = True,
        coincidence_radius: float | torch.Tensor = 0.0,
        mean_squared_displacement_per_dose: float = 0.0,
        periodic_potential: bool = False,
        bfactor: float | torch.Tensor | None = None,
    ):
        self.pad_fft = propagation.pad_fft
        self.ice = ice
        self.ice_thickness = ice.thickness
        self.ice_relax_steps = ice.relax_steps
        self.nxy = nxy

        self.pad_nxy = nxy + (nxy // 2) * 2 if self.pad_fft else nxy
        self.nz = compute_nz(self.nxy, ice.thickness, pixel_size)

        super().__init__(
            pixel_size=pixel_size,
            voltage=voltage,
            dose_per_angstrom=dose_per_angstrom,
            nxy=self.nxy,
            nz=self.nz,
            pad_nxy=self.pad_nxy,
            propagation=propagation,
            optics=optics,
            envelopes=envelopes,
            camera=camera,
            anisomag=anisomag,
            ctf_params=ctf_params,
            progressbars=False,
            verbose=verbose,
            coincidence_radius=coincidence_radius,
            bfactor=bfactor,
        )
        self.ice_model = ice.model
        # The TEMPLATE's depth, not `self.nz * pixel_size`: the neighbour slab
        # must not follow `ice.thickness`. See the constructor docstring. The
        # two agree until the ice is deeper than the box, which is what makes
        # this a no-op for every run that was not growing its crowding.
        self.crowding = crowding
        self.crowd_max_distance_z = (
            crowding.max_distance_z
            if crowding.max_distance_z is not None
            else self.nxy * pixel_size
        )
        self.crowd_min_distance = crowding.min_distance

        self.coordinates = nn.Parameter(coordinates)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
        self.atomic_numbers = atomic_numbers
        self.mean_squared_displacement_per_dose = mean_squared_displacement_per_dose
        if mean_squared_displacement_per_dose != 0 and self.verbose:
            logger.info(
                f"Perturbing coordinates by: {mean_squared_displacement_per_dose}."
            )

        self.potentialbuilder = PotentialBuilder(
            self.nxy,
            self.pixel_size,
            self.atomic_numbers,
            conv_backend=conv_backend,
            periodic=periodic_potential,
        )

        self.V = self.potentialbuilder(self.coordinates.detach())

        self._init_optics()
        self.scattering = self._build_scattering()

        self._apply_defocus_shift(
            shift_required=self.scattering_model not in ["projection", "ctf"]
        )

        if self.crowd_min_distance is not None:
            self.crowd = self._build_crowd(self.V, crowding, progressbars=True)

        self.ice_parameterization = ice.parameterization
        self.icemaker: IceBank | RandomIcemaker | None = resolve_icemaker(
            self.ice_model,
            pixel_size,
            self.nxy,
            self.nz,
            ice_cache_dir=ice.cache_dir,
            icemaker=icemaker,
            parameterization=self.ice_parameterization,
        )
        if icemaker is not None:
            self.ice_model = icemaker.method

    def rotate(self, Q: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Rotate and translate atomic coordinates.

        Parameters
        ----------
        Q : torch.Tensor
            Rotation quaternions.
        T : torch.Tensor
            Translations (x, y) in Å.

        Returns
        -------
        r_coordinates : torch.Tensor
            Rotated and translated coordinates, in Å.

        Notes
        -----
        The translation is applied directly in Å, after the rotation, so it acts
        in the lab frame and is subtracted -- the same origin-offset semantics
        :class:`ImageGenerator`'s volume path gets from
        :func:`~specter.rotations.build_affine_matrix`. It is deliberately *not*
        routed through
        :func:`~specter.rotations.translations_angstrom_to_torch`, which
        normalizes to ``grid_sample``'s [-1, 1] convention and only makes sense
        for an affine grid, not for atom positions.
        """
        R = roma.unitquat_to_rotmat(Q).transpose(-2, -1)  # inverse rotation
        N = self.coordinates.shape[0]
        if R.ndim == 2:
            rotated = self.coordinates @ R.T
        else:
            B = R.shape[0]
            vectors_exp = self.coordinates.unsqueeze(0).expand(B, N, 3)
            rotated = torch.einsum("bij,bkj->bki", R, vectors_exp)
        r_coordinates = translate_coordinates(rotated, T, inverse=True)
        return r_coordinates

    def forward(self, idx: int | torch.Tensor) -> torch.Tensor:
        """
        Generate images for the given batch indices.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices.

        Returns
        -------
        images : torch.Tensor
            Simulated images.
        """
        if self.mean_squared_displacement_per_dose != 0.0:
            msd = (
                self.mean_squared_displacement_per_dose * self.dose_per_angstrom.mean()
            )
            self.sigma_angstrom = (msd / 3) ** 0.5
            self.coordinates.add_(
                torch.randn_like(self.coordinates) * self.sigma_angstrom
            )

        coordinates = self.rotate(self.quaternions[idx], self.translations[idx])

        V = self.potentialbuilder(coordinates)
        if V.ndim == 3:
            V = V.unsqueeze(0)

        # "constant" -- see ImageGenerator.__getitem__ for why a single-particle
        # box zero-pads its protein channel while MicrographGenerator reflects.
        V = pad_volume(
            V,
            self.nxy,
            self.nz,
            self.ice_thickness,
            self.pad_fft,
            xy_pad_mode="constant",
        )
        return self.process_volume(V, idx)


class ImageGenerator(ParticleGeneratorBase):
    """
    Generates images from a pre-computed scattering potential volume.

    The volume is rotated and translated for each simulation instance.

    Parameters
    ----------
    scattering_potential : torch.Tensor
        Scattering potential volume. Shape (Z, Y, X).
    pixel_size : float
        Pixel size in Å.
    quaternions : torch.Tensor
        Rotation quaternions. Shape (B, 4).
    translations : torch.Tensor
        Translations (x, y) in Å. Shape (B, 2).
    ctf_params : dict or None
        Per-image CTF parameters. Required unless ``optics`` is ``None``.
    voltage : float
        Electron beam accelerating voltage in kV.
    dose_per_angstrom : float or torch.Tensor
        Total electron dose (fluence) per image in e⁻/Å². Scalar, or a 1-D
        tensor of length n giving a separate dose for each image.
    anisomag : torch.Tensor, optional
        Anisotropic magnification matrices.
    propagation : Propagation, optional
        How the exit wave is computed. Default ``Propagation()``.
    optics : Optics, optional
        The aberration stage; ``None`` skips it. Default ``Optics()``.
    envelopes : Envelopes, optional
        Coherence and radiation-damage envelopes. Default ``Envelopes()``.
    camera : Camera, optional
        The detector chain. Default ``Camera()``.
    ice : Ice, optional
        The amorphous ice the particle is embedded in: model, thickness,
        library, seam relaxation and scattering factors. Default ``Ice()``,
        no ice.
    icemaker : IceBank or RandomIcemaker, optional
        A pre-built icemaker instance to reuse across generators. When
        supplied, ``ice.model`` and ``ice.cache_dir`` are ignored;
        ``ice.thickness`` still sizes the column.
    crowding : Crowding, optional
        How duplicates of the particle are packed around it. Default
        ``Crowding()``, none. ``max_distance_z`` defaults to the template's
        own depth, which deliberately does **not** follow ``ice.thickness``:
        raising that on a particle stack is reaching for lower contrast and
        more solvent background, not for a denser specimen, so the two stay
        separate knobs. At 2000 Å of ice the neighbours occupy the middle
        of the column and the rest is water. `MicrographSpecimenGenerator`
        scales with ``nz`` instead, and should: a micrograph is a specimen,
        a particle stack is a controlled image-formation experiment.
    crowd_move_to_cpu : bool, optional
        Assemble the crowd on the host, trading speed for device memory.
        Default False.
    progressbars : bool, optional
        Whether to show progress bars. Default True.
    bfactor : float or torch.Tensor or None, optional
        Isotropic B-factor envelope in Å² applied in the microscope transfer
        function. None or 0.0 means no envelope. Default None.
    """

    def __init__(
        self,
        scattering_potential: torch.Tensor,
        pixel_size: float,
        quaternions: torch.Tensor,
        translations: torch.Tensor,
        ctf_params: dict[str, Any] | None,
        voltage: float,
        dose_per_angstrom: float | torch.Tensor,
        anisomag: torch.Tensor | None = None,
        propagation: Propagation = Propagation(),
        optics: Optics | None = Optics(),
        envelopes: Envelopes = Envelopes(),
        camera: Camera = Camera(),
        ice: Ice = Ice(),
        icemaker: IceBank | RandomIcemaker | None = None,
        crowding: Crowding = Crowding(),
        crowd_move_to_cpu: bool = False,
        progressbars: bool = True,
        verbose: bool = True,
        coincidence_radius: float | torch.Tensor = 0.0,
        potential_scale: float | torch.Tensor = 1.0,
        bfactor: float | torch.Tensor | None = None,
    ):
        nxy = scattering_potential.shape[-1]
        self.pad_fft = propagation.pad_fft
        self.ice = ice
        self.ice_thickness = ice.thickness
        self.ice_relax_steps = ice.relax_steps
        self.pad_nxy = nxy + (nxy // 2) * 2 if self.pad_fft else nxy

        volume_nz = scattering_potential.shape[0]
        self.nz = compute_nz(volume_nz, ice.thickness, pixel_size)

        super().__init__(
            pixel_size=pixel_size,
            voltage=voltage,
            dose_per_angstrom=dose_per_angstrom,
            nxy=nxy,
            nz=self.nz,
            pad_nxy=self.pad_nxy,
            propagation=propagation,
            optics=optics,
            envelopes=envelopes,
            camera=camera,
            anisomag=anisomag,
            ctf_params=ctf_params,
            progressbars=progressbars,
            verbose=verbose,
            coincidence_radius=coincidence_radius,
            potential_scale=potential_scale,
            bfactor=bfactor,
        )

        self.ice_parameterization = ice.parameterization
        self.ice_model = ice.model
        # The TEMPLATE's depth, not `self.nz * pixel_size`: the neighbour slab
        # must not follow `ice.thickness`. See the constructor docstring. The
        # two agree until the ice is deeper than the box, which is what makes
        # this a no-op for every run that was not growing its crowding.
        self.crowding = crowding
        self.crowd_max_distance_z = (
            crowding.max_distance_z
            if crowding.max_distance_z is not None
            else volume_nz * pixel_size
        )
        self.crowd_min_distance = crowding.min_distance

        self.register_buffer("V", scattering_potential)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("translations", translations)
        if self.verbose:
            logger.info("Initializing ImageGenerator modules")

        self._init_optics()
        self.scattering = self._build_scattering()

        if self.crowd_min_distance is not None:
            self.crowd = self._build_crowd(
                self.V,
                crowding,
                progressbars=self.progressbars,
                move_to_cpu=crowd_move_to_cpu,
            )

        self.icemaker: IceBank | RandomIcemaker | None = resolve_icemaker(
            self.ice_model,
            pixel_size,
            self.nxy,
            self.nz,
            ice_cache_dir=ice.cache_dir,
            icemaker=icemaker,
            parameterization=self.ice_parameterization,
            progressbars=self.progressbars,
        )
        if icemaker is not None:
            self.ice_model = icemaker.method

        self._apply_defocus_shift(
            shift_required=self.scattering_model not in ["projection", "ctf"]
        )

        nz, ny, nx = self.V.shape
        self.rotator = VolumeRotator(
            nz, ny, nx, origin="relion", mode=self.propagation.rotate_mode
        )

    def rotate(self, Q: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Rotate volume using affine transformation.

        Parameters
        ----------
        Q : torch.Tensor
            Rotation quaternions.
        T : torch.Tensor
            Translations.

        Returns
        -------
        V : torch.Tensor
            Rotated volume.
        """
        if len(Q.shape) < 2:
            Q = Q.unsqueeze(0)
        if len(T.shape) < 2:
            T = T.unsqueeze(0)
        R = roma.unitquat_to_rotmat(Q)
        T = rotations.translations_angstrom_to_torch(T, self.nxy, self.pixel_size)
        theta = rotations.build_affine_matrix(R, T)
        return self.rotator(self.V, theta)

    def forward(self, idx: int | torch.Tensor) -> torch.Tensor:
        """
        Generate images for the given batch indices.

        Parameters
        ----------
        idx : int or torch.Tensor
            Batch indices.

        Returns
        -------
        images : torch.Tensor
            Simulated images.
        """
        if self.verbose:
            logger.info("Rotating volume")
        V = self.rotate(self.quaternions[idx], self.translations[idx])

        # This pads the PROTEIN channel only. Ice fills the margin by its own
        # route (`solvate` reflect-pads it separately) and crowding is built
        # straight at the padded size, so the margin is never vacuum overall
        # and the choice here is only about what lies beyond the particle.
        #
        # "constant", i.e. no protein beyond the box. "reflect" would mirror
        # the target particle into the margin, putting deterministic copies of
        # it just outside the box, mirror-symmetric about each edge and
        # correlated with the particle being imaged. Real neighbours sit at
        # random positions and orientations, which is what `crowd` models.
        # MicrographGenerator reflects instead, and should: its volume is a
        # whole specimen field, so mirroring continues a statistically similar
        # one. A single-particle box is not that problem.
        V = pad_volume(
            V,
            self.nxy,
            self.nz,
            self.ice_thickness,
            self.pad_fft,
            xy_pad_mode="constant",
        )
        return self.process_volume(V, idx)
