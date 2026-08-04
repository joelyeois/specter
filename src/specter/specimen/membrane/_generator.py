"""
Public membrane specimen generator.

Ties together the four membrane submodules built this session into one
class: organic shape (``_field``), calibrated bilayer potential
(``_profile``), anti-aliased rasterization onto an output grid
(``_raster``), and normal-aligned transmembrane protein insertion
(``_placement``).

Everything shares one centered-origin physical coordinate frame: physical
``(0, 0, 0)`` is the volume's own center, matching the convention already
used throughout ``_field``/``_raster``. This generator
does not yet support placing a membrane instance off-center within a larger
tomogram -- that compositing step belongs to a higher-level assembler, the
same way the other specimen generators in this package are used standalone
before being composited.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from ...arrays import clip_insert_bounds
from ...pdb import PDB
from ...potential import PotentialBuilder
from ...rotations import build_affine_matrix, rotate_volume
from ..packing import estimate_protein_box_size
from ._field import MembraneField, generate_membrane_field
from ._placement import (
    align_principal_axis_to_z,
    align_transmembrane_depth,
    orientation_for_normal,
    sample_surface_sites,
)
from ._profile import (
    BilayerProfile,
    build_reference_lipid_patch,
    compute_bilayer_profile,
)
from ._raster import rasterize_membrane_density


@dataclass
class TransmembraneSpec:
    """
    One transmembrane protein species to attempt placing in the membrane.

    Parameters
    ----------
    pdb_source : str
        PDB ID or local PDB/mmCIF path. Used as the species id and, when
        ``template`` is not supplied, as the input to ``PotentialBuilder``.
    frequency : int, optional
        Relative placement weight among species. Default 1.
    tm_span_mask : torch.Tensor, optional
        Boolean mask, same atom order as the loaded structure's
        coordinates, selecting the transmembrane span. Default None, which
        centers the full structure's z-extent instead (see
        :func:`~specter.specimen.membrane._placement.align_transmembrane_depth`).
        Ignored when ``template`` is supplied -- depth alignment must
        already be baked into a prebuilt template.
    parameterization : str, optional
        PotentialBuilder parameterization for PDB-backed specs. Default
        "shtyrov".
    template : torch.Tensor, optional
        Prebuilt density template with shape ``(Z, Y, X)``, already
        depth-aligned (transmembrane span centered on the template's own
        z-center). If supplied, no PDB is loaded for this species.
    """

    pdb_source: str
    frequency: int = 1
    tm_span_mask: torch.Tensor | None = None
    parameterization: str = "shtyrov"
    template: torch.Tensor | None = None


@dataclass
class TransmembranePlacement:
    """One accepted transmembrane placement."""

    species_id: str
    center_xyz: torch.Tensor
    rotation_matrix: torch.Tensor


class MembraneGenerator:
    """
    Generate an organic membrane specimen volume with transmembrane inserts.

    Parameters
    ----------
    target_shape_zyx : tuple of int
        Output volume shape, ``(Z, Y, X)``.
    v_size : float
        Output voxel size, Angstrom. Also used to render transmembrane
        protein templates, so their scale matches the membrane's.
    n_sources : int, optional
        Number of blended metaball sources for the organic shape. Default 6.
    radius_range_a : tuple of float, optional
        Source radius range, Angstrom. Default ``(150.0, 400.0)``.
    spread_a : float, optional
        Source center spread, Angstrom. Default is
        :func:`~specter.specimen.membrane._field.generate_membrane_field`'s
        own default (a quarter of the working grid's smallest extent).
    noise_amplitude_a : float, optional
        Undulation noise amplitude, Angstrom. Default 15.0.
    correlation_length_a : float, optional
        Undulation noise correlation length, Angstrom. Default 40.0.
    curvature_iterations : int, optional
        Curvature-capping relaxation steps. Default 30.
    n_lipids_per_leaflet : int, optional
        Reference lipid patch size for the calibrated bilayer profile (see
        :func:`~specter.specimen.membrane._profile.build_reference_lipid_patch`).
        Default 200.
    parameterization : str, optional
        PotentialBuilder parameterization for the lipid reference patch.
        Default "shtyrov".
    transmembrane_specs : list of TransmembraneSpec, optional
        Transmembrane protein species to attempt placing. Default None (no
        transmembrane proteins).
    pdb_cache_dir : str, optional
        Passed to `PDB` for PDB-backed transmembrane specs. Default
        "../pdb-data/".
    max_field_voxels : int, optional
        Safety cap on the working field grid's total voxel count. The
        field's own spacing is derived to resolve the bilayer's physical
        extent (see `generate`'s inline comment) independent of
        `target_shape_zyx` -- fine at the small scales this was built and
        tested at, but that spacing applied UNIFORMLY across a real
        production-scale volume (hundreds of voxels per axis) produces a
        working grid orders of magnitude larger than `target_shape_zyx`
        itself: confirmed a (200, 600, 600)-voxel, 10 A/voxel target
        implying a 500x1500x1500 (~1.1 billion voxel) field, ballooning to
        50+ GB of resident memory across the several such arrays field
        generation allocates, well past what's practical. If the naive
        spacing would exceed this budget, it's coarsened (increased) just
        enough to fit, with a warning -- a real, disclosed accuracy
        tradeoff (a coarser-than-ideal field resolves the bilayer's own
        sub-structure less crisply), not a silent one; this is a stopgap,
        not a fix for the underlying issue (the field is one uniform grid
        over the WHOLE domain regardless of how much of it is actually
        near a membrane surface -- a real architectural limitation, worth
        a proper adaptive/local-patch redesign later, not attempted here).
        Default 200_000_000 (~800 MB at float32 per array).
    device : str or torch.device, optional
        Device for generation. Default "cpu".
    seed : int, optional
        Random seed. Default None.
    """

    def __init__(
        self,
        target_shape_zyx: tuple[int, int, int],
        v_size: float,
        n_sources: int = 6,
        radius_range_a: tuple[float, float] = (150.0, 400.0),
        spread_a: float | None = None,
        noise_amplitude_a: float = 15.0,
        correlation_length_a: float = 40.0,
        curvature_iterations: int = 30,
        n_lipids_per_leaflet: int = 200,
        parameterization: str = "shtyrov",
        transmembrane_specs: list[TransmembraneSpec] | None = None,
        pdb_cache_dir: str = "../pdb-data/",
        max_field_voxels: int = 200_000_000,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ):
        tz, ty, tx = target_shape_zyx
        self.target_shape_zyx: tuple[int, int, int] = (int(tz), int(ty), int(tx))
        self.v_size = float(v_size)
        self.n_sources = n_sources
        self.radius_range_a = radius_range_a
        self.spread_a = spread_a
        self.noise_amplitude_a = noise_amplitude_a
        self.correlation_length_a = correlation_length_a
        self.curvature_iterations = curvature_iterations
        self.n_lipids_per_leaflet = n_lipids_per_leaflet
        self.parameterization = parameterization
        self.transmembrane_specs = transmembrane_specs or []
        self.pdb_cache_dir = pdb_cache_dir
        self.max_field_voxels = max_field_voxels
        self.device = torch.device(device)
        self.seed = seed

        self.field: MembraneField | None = None
        self.profile: BilayerProfile | None = None
        self.volume: torch.Tensor | None = None
        self.placements: list[TransmembranePlacement] = []
        self._origin_xyz: torch.Tensor | None = None

    def generate(self) -> torch.Tensor:
        """
        Build the calibrated bilayer profile and organic field, and
        rasterize into ``self.volume``.

        Returns
        -------
        torch.Tensor
            Density volume, shape ``target_shape_zyx``.
        """
        atomic_numbers, coordinates = build_reference_lipid_patch(
            n_lipids_per_leaflet=self.n_lipids_per_leaflet,
            seed=self.seed,
            device=self.device,
        )
        self.profile = compute_bilayer_profile(
            atomic_numbers, coordinates, parameterization=self.parameterization
        )

        # The field's working spacing must stay fine enough to resolve the
        # bilayer's own extent (curvature capping, leaflet offset) --
        # derived from the profile's actual rendered table rather than a
        # hardcoded thickness, so this stays correct if the profile's
        # template is retuned later.
        half_extent_a = float(self.profile.distance_a.abs().max())
        field_spacing_a = min(self.v_size, half_extent_a / 8)

        nz, ny, nx = self.target_shape_zyx
        extent_a = torch.tensor([nx, ny, nz], dtype=torch.float32) * self.v_size
        self._origin_xyz = -0.5 * extent_a

        field_shape_zyx = (
            int(torch.ceil(extent_a[2] / field_spacing_a)),
            int(torch.ceil(extent_a[1] / field_spacing_a)),
            int(torch.ceil(extent_a[0] / field_spacing_a)),
        )
        n_field_voxels = field_shape_zyx[0] * field_shape_zyx[1] * field_shape_zyx[2]
        if n_field_voxels > self.max_field_voxels:
            # coarsen just enough to fit max_field_voxels (voxel count
            # scales as 1/spacing^3) -- see max_field_voxels' own
            # docstring for why this is needed and what it trades away.
            field_spacing_a *= (n_field_voxels / self.max_field_voxels) ** (1.0 / 3.0)
            field_shape_zyx = (
                int(torch.ceil(extent_a[2] / field_spacing_a)),
                int(torch.ceil(extent_a[1] / field_spacing_a)),
                int(torch.ceil(extent_a[0] / field_spacing_a)),
            )
            warnings.warn(
                f"MembraneGenerator: the working field grid at the resolution "
                f"needed to fully resolve the bilayer ({n_field_voxels:,} voxels) "
                f"exceeds max_field_voxels ({self.max_field_voxels:,}) -- "
                f"coarsened field_spacing_a to {field_spacing_a:.2f} A "
                f"({field_shape_zyx[2]}x{field_shape_zyx[1]}x{field_shape_zyx[0]} "
                "voxels). The bilayer's own sub-structure will be resolved less "
                "crisply than at full resolution; raise max_field_voxels if you "
                "have the memory to spare.",
                stacklevel=2,
            )
        self.field = generate_membrane_field(
            shape_zyx=field_shape_zyx,
            spacing_a=field_spacing_a,
            n_sources=self.n_sources,
            radius_range_a=self.radius_range_a,
            spread_a=self.spread_a,
            noise_amplitude_a=self.noise_amplitude_a,
            correlation_length_a=self.correlation_length_a,
            curvature_iterations=self.curvature_iterations,
            device=self.device,
            seed=self.seed,
        )

        self.volume = rasterize_membrane_density(
            self.field,
            self.profile,
            target_shape_zyx=self.target_shape_zyx,
            target_spacing_a=self.v_size,
            target_origin_xyz=self._origin_xyz,
        )
        return self.volume

    def place_transmembrane(
        self, min_spacing_a: float = 40.0, max_attempts: int | None = None
    ) -> list[TransmembranePlacement]:
        """
        Sample surface sites and insert transmembrane protein templates.

        Must be called after :meth:`generate`. Species are chosen per site
        by weighted random draw (weights = ``frequency``), matching
        ``TetrisPackingSpecimenGenerator``'s species-selection convention.

        Parameters
        ----------
        min_spacing_a : float, optional
            Minimum center-to-center spacing between placed sites,
            Angstrom. Default 40.0.
        max_attempts : int, optional
            Passed to
            :func:`~specter.specimen.membrane._placement.sample_surface_sites`.
            Default is that function's own default.

        Returns
        -------
        list of TransmembranePlacement
        """
        if self.field is None or self.volume is None or self._origin_xyz is None:
            raise RuntimeError("call generate() before place_transmembrane()")
        if not self.transmembrane_specs:
            return []

        n_sites = sum(spec.frequency for spec in self.transmembrane_specs)
        sites_xyz, normals_xyz = sample_surface_sites(
            self.field,
            n_sites=n_sites,
            min_spacing_a=min_spacing_a,
            max_attempts=max_attempts,
            seed=self.seed,
        )

        templates = {
            spec.pdb_source: self._build_template(spec)
            for spec in self.transmembrane_specs
        }
        weights = torch.tensor(
            [spec.frequency for spec in self.transmembrane_specs], dtype=torch.float32
        )
        chooser = torch.Generator(device="cpu")
        if self.seed is not None:
            chooser.manual_seed(self.seed)

        placements: list[TransmembranePlacement] = []
        for i in range(sites_xyz.shape[0]):
            chosen_idx = int(torch.multinomial(weights, 1, generator=chooser).item())
            spec = self.transmembrane_specs[chosen_idx]
            template = templates[spec.pdb_source]

            site_seed = None if self.seed is None else self.seed + i
            rotation = orientation_for_normal(normals_xyz[i], seed=site_seed)
            theta = build_affine_matrix(rotation.to(self.device))
            rotated = rotate_volume(template, theta, padding_mode="zeros")[0]

            center_zyx = self._physical_to_voxel_index(sites_xyz[i])
            self._insert_add(rotated, center_zyx)

            placements.append(
                TransmembranePlacement(
                    species_id=spec.pdb_source,
                    center_xyz=sites_xyz[i].detach().cpu(),
                    rotation_matrix=rotation.detach().cpu(),
                )
            )

        self.placements = placements
        return placements

    def _build_template(self, spec: TransmembraneSpec) -> torch.Tensor:
        if spec.template is not None:
            return spec.template.detach().to(self.device, dtype=torch.float32)

        pdb = PDB(spec.pdb_source, savefolder=self.pdb_cache_dir, verbose=False)
        coordinates = align_principal_axis_to_z(pdb.coordinates)
        coordinates = align_transmembrane_depth(coordinates, spec.tm_span_mask)
        n = estimate_protein_box_size(pdb.max_diameter, self.v_size)
        builder = PotentialBuilder(
            n_xyz=n,
            dx=self.v_size,
            atomic_numbers=pdb.atomic_numbers,
            progressbars=False,
            parameterization=spec.parameterization,
        )
        # "analytic" (PotentialBuilder's own documented default), not "3d" --
        # confirmed empirically the two methods integrate to the same total
        # potential (ratio ~1.001 across a 10x span of voxel sizes, on a
        # real structure) but "3d"'s convolution-based approach spreads that
        # same total over a visibly lower, smoother peak (25-45% lower
        # max value than "analytic" for identical atoms). The bilayer
        # profile is rendered with "analytic"; using "3d" here compared two
        # differently-sharpened renderings of the same underlying physics,
        # not a deliberate contrast difference between membrane and
        # protein.
        return builder.forward(coordinates, method="analytic").to(self.device)

    def _physical_to_voxel_index(self, site_xyz: torch.Tensor) -> torch.Tensor:
        assert self._origin_xyz is not None
        origin = self._origin_xyz.cpu()
        idx_xyz = (site_xyz.detach().cpu() - origin) / self.v_size
        idx_zyx = torch.stack([idx_xyz[2], idx_xyz[1], idx_xyz[0]])
        return idx_zyx.round().long()

    def _insert_add(
        self, local_density: torch.Tensor, center_zyx: torch.Tensor
    ) -> None:
        assert self.volume is not None
        bounds = clip_insert_bounds(
            center_zyx.tolist(), local_density.shape, self.volume.shape
        )
        if bounds is None:
            return
        dst, src = bounds
        self.volume[dst] += local_density[src].to(self.volume.device)


__all__ = [
    "MembraneGenerator",
    "TransmembraneSpec",
    "TransmembranePlacement",
]
