"""
Unit tests for specter.specimen._grid (BeadGenerator), the gold fiducial
bead generator descended from CTS's ``gen_beads.m``. Placement-level
coverage (via MembraneTomogramGenerator's bead_specs) lives in
test_tomogram_generator.py; these focus on the physics this module adds:
real mean-inner-potential calibration surviving the atomic fill, the
lattice texture, boundary geometry, and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import specter
from specter.specimen._grid import BeadGenerator

# Literature mean inner potential for bulk gold, V -- see the module
# docstring of specter.specimen._grid for the derivation route (bulk mass
# density -> number density -> per-atom potential integral).
_MIP_LO, _MIP_HI = 20.0, 40.0

_V_SIZE = 10.0
_RADIUS = 60.0


def test_mean_inner_potential_is_physical_and_resolution_independent() -> None:
    """Gold's MIP lands in the literature range and does not depend on
    voxel size -- the property that motivated replacing CTS's raw
    atom-count intensity."""
    mips = [BeadGenerator(voxel_size=v).mean_inner_potential for v in (6.0, 10.0, 20.0)]
    for mip in mips:
        assert _MIP_LO < mip < _MIP_HI, f"gold MIP out of range: {mip} V"
    assert max(mips) - min(mips) < 0.05 * np.mean(mips)


def test_atomic_fill_preserves_mean_density() -> None:
    """The lattice texture is variation about the right value, not a
    rescaling: the mean density over the bead's interior must recover the
    bulk MIP."""
    gen = BeadGenerator(voxel_size=_V_SIZE)
    bead = gen.generate(radius=_RADIUS)

    # Interior only -- rim voxels are partly outside the boundary and so
    # legitimately average below bulk.
    interior = bead.density[_erode(bead.mask)]
    assert interior.numel() > 50
    assert torch.isclose(
        interior.mean(), torch.tensor(gen.mean_inner_potential), rtol=0.1
    )


def test_fill_carries_lattice_texture_not_uniform_density() -> None:
    """A bead is built from discrete atoms, so interior voxels vary. The
    variation must be far below the Poisson level a random gas would give
    (1/sqrt(atoms per voxel)) -- that gap is the whole reason the
    Poisson-gas fill was dropped."""
    gen = BeadGenerator(voxel_size=_V_SIZE)
    bead = gen.generate(radius=_RADIUS)
    interior = bead.density[_erode(bead.mask)]

    assert interior.std() > 0, "fill is perfectly uniform -- atoms were lost"
    cv = (interior.std() / interior.mean()).item()
    poisson_cv = 1.0 / np.sqrt(gen.number_density * _V_SIZE**3)
    assert cv < 0.5 * poisson_cv, f"texture is Poisson-like: {cv} vs {poisson_cv}"


def test_beads_are_independent_realisations() -> None:
    """Two beads of the same radius must not share texture, or every
    fiducial in a tomogram would be a copy of the same one."""
    gen = BeadGenerator(voxel_size=_V_SIZE)
    a = gen.generate(radius=_RADIUS).density
    b = gen.generate(radius=_RADIUS).density
    assert not torch.allclose(a, b)


def test_reproducible_under_global_seed() -> None:
    """Same seed, same bead -- to floating-point accumulation order, not
    bitwise. `soft_voxelize_coordinates` scatters atoms in parallel, so
    with >1 thread its summation order varies run to run (a property of
    specter's shared splat path, which ice and protein rendering use too,
    not of the bead code). Measured at ~5e-7 relative on 64 threads;
    single-threaded it is bitwise identical."""
    specter.seed(1234)
    a = BeadGenerator(voxel_size=_V_SIZE).generate(radius=_RADIUS).density
    specter.seed(1234)
    b = BeadGenerator(voxel_size=_V_SIZE).generate(radius=_RADIUS).density
    assert torch.allclose(a, b, rtol=0, atol=1e-5 * a.max())


def _voxel_distance(n: int) -> torch.Tensor:
    """Distance of each voxel centre from the bead centre, in voxels. The
    centre sits exactly on index ``n // 2`` -- the origin convention
    `soft_voxelize_coordinates` and `arrays.clip_insert_bounds` share, so
    a rendered bead lands where placement expects it."""
    idx = torch.arange(n, dtype=torch.float32) - n // 2
    zz, yy, xx = torch.meshgrid(idx, idx, idx, indexing="ij")
    return torch.sqrt(zz**2 + yy**2 + xx**2)


def test_mask_is_centred_and_unclipped() -> None:
    """Geometry is separate from the fill: the mask is centred with
    padding on every face (CTS's was offset by 1.5 voxels and clipped on
    three), and volume-matched to the nominal sphere."""
    gen = BeadGenerator(voxel_size=_V_SIZE)
    specter.seed(0)
    bead = gen.generate(radius=_RADIUS)
    mask = bead.mask
    n = mask.shape[0]

    assert n % 2 == 1, "odd n puts the centre exactly on a voxel"

    # Padded on all six faces.
    for axis in range(3):
        assert not mask.movedim(axis, 0)[0].any()
        assert not mask.movedim(axis, 0)[-1].any()

    # Centred: the mask's centre of mass sits on the array centre. (Not
    # reflection-symmetric -- the boundary is deliberately irregular.)
    idx = torch.arange(n, dtype=torch.float32) - n // 2
    for axis in range(3):
        profile = mask.float().sum(dim=[d for d in range(3) if d != axis])
        centroid = (profile * idx).sum() / profile.sum()
        assert abs(centroid.item()) < 0.35 * (_RADIUS / _V_SIZE)

    # Volume-matched to a sphere of this radius, to within voxelisation.
    expected = (4 / 3) * np.pi * (_RADIUS / _V_SIZE) ** 3
    assert abs(mask.sum().item() - expected) < 0.15 * expected


def test_density_stays_within_a_kernel_width_of_the_boundary() -> None:
    """Atoms are pruned in continuous coordinates and then splatted through
    the atomic potential kernel, so density reaches slightly past the
    voxel-centre mask -- but only by the kernel's own half-width plus half
    a voxel diagonal, never further."""
    from specter.specimen._grid import _BeadShape

    gen = BeadGenerator(voxel_size=_V_SIZE)
    specter.seed(0)
    bead = gen.generate(radius=_RADIUS)

    # Reconstruct the same shape realisation to get its true extent.
    specter.seed(0)
    gen._random_orientation(None)
    shape = _BeadShape(_RADIUS, gen.roughness_range[0], None)

    dist = _voxel_distance(bead.density.shape[0])
    lit = bead.density > 1e-6 * bead.density.max()
    reach = shape.half_extent / _V_SIZE + np.sqrt(3) / 2 + gen._kernel.shape[0] / 2

    assert dist[lit].max() <= reach
    assert (lit & ~bead.mask).any(), "density stops exactly at the mask"


# --------------------------------------------------------------------
# Shape and orientation
# --------------------------------------------------------------------


@pytest.mark.parametrize("roughness", [0.0, 0.06, 0.12, 0.25])
def test_volume_matched_at_every_roughness(roughness) -> None:
    """The whole point of volume-matching: total integrated potential is
    gold's MIP times the nominal sphere volume, however lumpy a particular
    bead came out. Otherwise `roughness` would silently rescale every
    fiducial's signal."""
    radius = 50.0
    specter.seed(0)
    gen = BeadGenerator(voxel_size=_V_SIZE, roughness=roughness)
    bead = gen.generate(radius=radius)

    nominal = (4 / 3) * np.pi * radius**3 * gen.mean_inner_potential
    integral = bead.density.sum().item() * _V_SIZE**3
    assert abs(integral / nominal - 1.0) < 0.03


def test_mask_follows_the_irregular_boundary() -> None:
    """The segmentation mask must track the lumpy boundary, not quietly
    stay spherical. Measured as the symmetric difference against the
    equal-volume sphere: for a genuinely irregular bead that is a large
    fraction of the bead, for a round one it is only voxelisation noise."""

    def mismatch_vs_equal_volume_sphere(roughness: float) -> float:
        specter.seed(0)
        bead = BeadGenerator(voxel_size=2.0, roughness=roughness).generate(radius=50.0)
        mask = bead.mask
        # Radius of the sphere holding the same number of voxels.
        r_eq = (3.0 * mask.sum().item() / (4.0 * np.pi)) ** (1.0 / 3.0)
        sphere = _voxel_distance(mask.shape[0]) <= r_eq
        return (mask ^ sphere).sum().item() / mask.sum().item()

    round_bead = mismatch_vs_equal_volume_sphere(0.0)
    lumpy_bead = mismatch_vs_equal_volume_sphere(0.12)

    assert round_bead < 0.05, f"roughness=0 is not round: {round_bead}"
    assert lumpy_bead > 4 * round_bead, (
        f"boundary barely differs from a sphere: {lumpy_bead} vs {round_bead}"
    )


def test_each_bead_gets_a_fresh_crystal_orientation() -> None:
    """Consecutive fiducials must not share a lattice direction, or every
    bead in a tomogram shows its fringes running the same way. The
    orientation must also be uniform over SO(3), not clustered."""
    gen = BeadGenerator(voxel_size=_V_SIZE)
    specter.seed(0)
    axes = torch.stack([gen._random_orientation(None)[:, 0] for _ in range(300)])

    # Consecutive beads point somewhere genuinely different.
    cos = (axes[:-1] * axes[1:]).sum(-1).clamp(-1, 1)
    assert torch.rad2deg(torch.acos(cos)).mean() > 60.0

    # Uniform on SO(3) => the rotated axis is uniform on the sphere, so
    # each component has mean 0 and variance 1/3.
    assert axes.mean(0).abs().max() < 0.12
    assert (axes.var(0) - 1 / 3).abs().max() < 0.08


def test_explicit_generator_controls_the_whole_bead() -> None:
    """`generate(generator=...)` must drive the orientation as well as the
    jitter and shape. `roma.random_rotmat` takes no generator, so using it
    would leave the crystal orientation reading the global RNG -- silently
    unreproducible under an explicit generator."""
    gen = BeadGenerator(voxel_size=_V_SIZE)

    specter.seed(111)
    a = gen.generate(radius=40.0, generator=torch.Generator().manual_seed(7)).density
    specter.seed(999)  # deliberately different global state
    b = gen.generate(radius=40.0, generator=torch.Generator().manual_seed(7)).density

    assert torch.allclose(a, b, rtol=0, atol=1e-5 * a.max())


def test_lumpiness_is_isotropic() -> None:
    """A bead's irregularity must have no preferred axis. Building the
    harmonic field from bare Re/Im parts instead of the orthonormal real
    combination (sqrt(2)*(-1)^m, via the membrane backend's
    `_sample_sh_coefficients`/`_real_spherical_harmonic`) over-weights the
    m=0 modes, which peak at the poles: that gave ~50% more RMS
    modulation at theta=0 than at the equator."""
    from specter.specimen._grid import _BeadShape

    probes = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # pole
            [np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)],  # mid latitude
            [1.0, 0.0, 0.0],  # equator
        ]
    )
    samples = []
    for trial in range(120):
        gen = torch.Generator().manual_seed(trial)
        shape = _BeadShape(50.0, roughness=0.06, generator=gen)
        samples.append(shape._field(probes))

    rms = torch.stack(samples).std(dim=0)
    assert (rms > 0).all()
    assert rms.max() / rms.min() < 1.25, f"anisotropic lumpiness: {rms.tolist()}"


def test_rejects_bad_roughness() -> None:
    with pytest.raises(ValueError, match="roughness must be >= 0"):
        BeadGenerator(voxel_size=_V_SIZE, roughness=-0.1)
    with pytest.raises(ValueError, match=r"roughness range must be \[low, high\]"):
        BeadGenerator(voxel_size=_V_SIZE, roughness=[0.2, 0.05])


def test_roughness_range_varies_lumpiness_between_beads() -> None:
    """A [low, high] roughness must give a population a MIX of near-round
    and misshapen particles. With a single number every bead is a
    different lumpy shape but the same DEGREE of lumpy -- measured here as
    each mask's mismatch against its own equal-volume sphere."""

    def mismatches(roughness) -> torch.Tensor:
        specter.seed(0)
        gen = BeadGenerator(voxel_size=4.0, roughness=roughness)
        out = []
        for _ in range(8):
            mask = gen.generate(radius=50.0).mask
            r_eq = (3.0 * mask.sum().item() / (4.0 * np.pi)) ** (1.0 / 3.0)
            sphere = _voxel_distance(mask.shape[0]) <= r_eq
            out.append((mask ^ sphere).sum().item() / mask.sum().item())
        return torch.tensor(out)

    fixed = mismatches(0.15)
    ranged = mismatches([0.0, 0.3])
    assert ranged.std() > 2 * fixed.std(), (
        f"range gives no extra variety: {ranged.tolist()} vs {fixed.tolist()}"
    )


def _erode(mask: torch.Tensor) -> torch.Tensor:
    """Strip the one-voxel rim shell, where partial occupancy legitimately
    lowers the mean."""
    eroded = mask.clone()
    for axis in range(3):
        for shift in (1, -1):
            eroded &= torch.roll(mask, shifts=shift, dims=axis)
    return eroded
