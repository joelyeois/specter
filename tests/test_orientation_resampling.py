"""Drawing orientations from an existing dataset's viewing directions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import roma
import starfile
import torch
from cryosparc.dataset import Dataset

from specter import rotations
from conftest import blob
from specter.config import MicrographConfig, ParticleStackConfig
from specter.crowding import CrowdWithDuplicates
from specter.imaging import Poses
from specter.io import read_orientations
from specter.pipelines import run_micrograph, run_particle_stack

_PDB_FIXTURE = str(Path(__file__).parent / "test_data" / "1mbo.cif")

# Two well-separated poses, neither the identity, as rotation vectors.
_ROTVECS = np.array([[0.4, -1.1, 0.7], [-1.3, 0.2, 2.0]], dtype=np.float32)


def _viewing_directions(quaternions: torch.Tensor) -> torch.Tensor:
    """The particle-frame axis along the beam: the rotation's last column."""
    return roma.unitquat_to_rotmat(quaternions)[..., 2]


def _write_pose_only_csfile(path: Path, rotvecs: np.ndarray) -> None:
    Dataset(
        {
            "uid": np.arange(len(rotvecs), dtype=np.uint64),
            "alignments3D/pose": rotvecs,
        }
    ).save(str(path))


def _blobs(n: int = 48) -> torch.Tensor:
    """A compact, smooth object with no symmetry."""
    g = torch.arange(n) - n / 2
    z, y, x = torch.meshgrid(g, g, g, indexing="ij")
    V = torch.zeros(n, n, n)
    for centre, sigma, amplitude in [
        ((6, 0, 0), 3.0, 1.0),
        ((0, 8, 2), 2.5, 0.7),
        ((-5, -4, 7), 2.0, 1.3),
        ((3, -9, -6), 3.0, 0.5),
    ]:
        r2 = (z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2
        V += amplitude * torch.exp(-r2 / (2 * sigma**2))
    return V


def _radial_profile(image: torch.Tensor) -> torch.Tensor:
    n = image.shape[-1]
    g = torch.arange(n) - n / 2
    y, x = torch.meshgrid(g, g, indexing="ij")
    r = torch.sqrt(y**2 + x**2).round().long()
    return torch.bincount(r.flatten(), weights=image.flatten(), minlength=n)[: n // 2]


def test_each_draw_keeps_a_source_viewing_direction() -> None:
    source = roma.rotvec_to_unitquat(torch.as_tensor(_ROTVECS))
    drawn = rotations.resample_viewing_directions(
        source, 200, generator=torch.Generator().manual_seed(0)
    )
    assert drawn.shape == (200, 4)
    torch.testing.assert_close(drawn.norm(dim=-1), torch.ones(200), atol=1e-6, rtol=0)

    # cosine between every draw's direction and each source direction
    cosines = _viewing_directions(drawn) @ _viewing_directions(source).T
    best, which = cosines.max(dim=-1)
    torch.testing.assert_close(best, torch.ones(200), atol=1e-5, rtol=0)
    # both source poses are used, and the draws are not copies of them
    assert set(which.tolist()) == {0, 1}
    assert not any(torch.allclose(q, s, atol=1e-3) for q in drawn for s in source)


def test_the_in_plane_angle_is_uniform() -> None:
    source = roma.rotvec_to_unitquat(torch.as_tensor(_ROTVECS[:1]))
    drawn = rotations.resample_viewing_directions(
        source, 4000, generator=torch.Generator().manual_seed(1)
    )
    # source^-1 * draw is the rotation about the beam each draw added
    spin = roma.quat_product(roma.quat_conjugation(source.expand(4000, 4)), drawn)
    torch.testing.assert_close(spin[:, :2], torch.zeros(4000, 2), atol=1e-5, rtol=0)
    angle = 2 * torch.atan2(spin[:, 2], spin[:, 3])
    assert abs(torch.cos(angle).mean()) < 0.05
    assert abs(torch.sin(angle).mean()) < 0.05
    assert abs(torch.cos(2 * angle).mean()) < 0.05


def test_a_resampled_pose_images_the_same_view_turned_in_plane() -> None:
    """
    The draw must turn the projection in its own plane, under the rotation
    convention the simulator images with, and not show a different view.
    """
    V = _blobs()
    n = V.shape[-1]

    def project(q: torch.Tensor) -> torch.Tensor:
        R = roma.unitquat_to_rotmat(q.reshape(1, 4))
        turned = rotations.rotate_volume(V, rotations.build_affine_matrix(R))
        return turned.reshape(n, n, n).sum(0)

    source = roma.rotvec_to_unitquat(torch.as_tensor(_ROTVECS[:1]))
    reference = _radial_profile(project(source))
    scale = reference.abs().max()

    drawn = rotations.resample_viewing_directions(
        source, 3, generator=torch.Generator().manual_seed(2)
    )
    for q in drawn:
        change = (_radial_profile(project(q)) - reference).abs().max() / scale
        assert change < 0.04  # interpolation only

    # Control: the same spin applied in the lab frame shows a different view.
    half = torch.tensor(0.6)
    spin = torch.tensor([[0.0, 0.0, torch.sin(half), torch.cos(half)]])
    other = roma.quat_product(spin, source)
    change = (_radial_profile(project(other)) - reference).abs().max() / scale
    assert change > 0.08


def test_resampling_rejects_a_malformed_source() -> None:
    with pytest.raises(ValueError, match=r"shape \(m, 4\)"):
        rotations.resample_viewing_directions(torch.zeros(0, 4), 3)
    with pytest.raises(ValueError, match=r"shape \(m, 4\)"):
        rotations.resample_viewing_directions(torch.zeros(5, 3), 3)


def test_poses_resample_is_seeded_and_leaves_the_global_stream() -> None:
    source = roma.rotvec_to_unitquat(torch.as_tensor(_ROTVECS))
    torch.manual_seed(7)
    expected_next = torch.rand(3)

    torch.manual_seed(7)
    a = Poses.resample(source, 16, shift=2.0, seed=11)
    torch.testing.assert_close(torch.rand(3), expected_next)
    b = Poses.resample(
        Poses(source, torch.zeros(2, 2), torch.arange(2)), 16, 2.0, seed=11
    )
    c = Poses.resample(source, 16, shift=2.0, seed=12)

    torch.testing.assert_close(a.quaternions, b.quaternions)
    torch.testing.assert_close(a.shifts, b.shifts)
    assert not torch.allclose(a.quaternions, c.quaternions)
    assert a.shifts.abs().max() <= 2.0
    assert a.ids.tolist() == list(range(16))


def test_read_orientations_reads_both_formats(tmp_path: Path) -> None:
    cs_path = tmp_path / "poses.cs"
    _write_pose_only_csfile(cs_path, _ROTVECS)
    from_cs = read_orientations(cs_path)
    torch.testing.assert_close(
        from_cs, roma.rotvec_to_unitquat(torch.as_tensor(_ROTVECS))
    )

    euler = np.array([[30.0, 70.0, -40.0], [-120.0, 15.0, 85.0]])
    star_path = tmp_path / "poses.star"
    starfile.write(
        pd.DataFrame(
            {
                "rlnAngleRot": euler[:, 0],
                "rlnAngleTilt": euler[:, 1],
                "rlnAnglePsi": euler[:, 2],
            }
        ),
        star_path,
        overwrite=True,
    )
    from_star = read_orientations(star_path)
    expected = roma.euler_to_unitquat(
        "ZYZ", torch.as_tensor(euler, dtype=torch.float32), degrees=True
    )
    torch.testing.assert_close(from_star, expected)


def test_read_orientations_reports_what_is_missing(tmp_path: Path) -> None:
    no_pose = tmp_path / "no_pose.cs"
    Dataset({"uid": np.arange(3, dtype=np.uint64)}).save(str(no_pose))
    with pytest.raises(KeyError, match="alignments3D/pose"):
        read_orientations(no_pose)

    no_psi = tmp_path / "no_psi.star"
    starfile.write(
        pd.DataFrame({"rlnAngleRot": [0.0, 1.0], "rlnAngleTilt": [0.0, 1.0]}),
        no_psi,
        overwrite=True,
    )
    with pytest.raises(KeyError, match="rlnAnglePsi"):
        read_orientations(no_psi)

    with pytest.raises(ValueError, match=r"\.cs or \.star file, a"):
        read_orientations(tmp_path / "poses.json")


def test_run_particle_stack_draws_from_orientation_distribution_path(
    tmp_path: Path,
) -> None:
    """Every written pose views the particle along the source file's direction."""
    source_path = tmp_path / "source.cs"
    _write_pose_only_csfile(source_path, _ROTVECS[:1])

    config = ParticleStackConfig(
        pdb_source=_PDB_FIXTURE,
        assembly=False,
        n_pixels=48,
        n_particles=6,
        scattering_model="projection",
        ice_model="none",
        detector_model="none",
        device="cpu",
        batchsize=6,
        seed=3,
        output_dir=str(tmp_path),
        filename="biased",
        orientation_distribution_path=str(source_path),
    )
    run_particle_stack(config)

    df = starfile.read(tmp_path / "biased.star")
    euler = torch.as_tensor(
        df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(),
        dtype=torch.float32,
    )
    written = roma.euler_to_unitquat("ZYZ", euler, degrees=True)
    source = roma.rotvec_to_unitquat(torch.as_tensor(_ROTVECS[:1]))
    cosines = _viewing_directions(written) @ _viewing_directions(source)[0]
    torch.testing.assert_close(cosines, torch.ones(6), atol=1e-4, rtol=0)
    # the in-plane angle differs from particle to particle
    assert not torch.allclose(written[0], written[1], atol=1e-3)


def test_run_particle_stack_rejects_orientation_distribution_path_with_a_dataset(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.cs"
    _write_pose_only_csfile(source_path, _ROTVECS)
    config = ParticleStackConfig(
        pdb_source=_PDB_FIXTURE,
        cs_path=str(source_path),
        orientation_distribution_path=str(source_path),
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="orientation_distribution_path can't be set"):
        run_particle_stack(config)


@pytest.mark.parametrize(
    "geometry",
    [
        dict(dx=2.0, min_distance=30.0, nxy_out=192, nz_out=64),
        dict(
            dx=4.0,
            min_distance=30.0,
            nxy_out=96,
            nz_out=48,
            packing_backend="shape",
            atom_coordinates=blob(300),
            packing_max_retries=300,
            packing_stall_patience=2000,
            packing_seed=5,
        ),
    ],
    ids=["poisson_disk", "shape_backend"],
)
def test_crowd_duplicates_take_the_source_viewing_directions(geometry: dict) -> None:
    source = roma.rotvec_to_unitquat(torch.as_tensor(_ROTVECS))
    torch.manual_seed(0)
    crowd = CrowdWithDuplicates(
        torch.rand(16, 16, 16), orientations=source, progressbars=False, **geometry
    )
    crowd.generate_coordinates()
    crowd.generate_affine_matrices()
    assert crowd.N >= 4

    R = crowd.theta[:, :3, :3]
    best = (R[..., 2] @ _viewing_directions(source).T).max(dim=-1).values
    torch.testing.assert_close(best, torch.ones(crowd.N), atol=1e-5, rtol=0)
    # the in-plane angle is free, so the duplicates are not all alike
    assert (R - R[0]).abs().amax(dim=(1, 2)).max() > 0.1


class _Captured(Exception):
    pass


def test_run_micrograph_hands_orientation_distribution_path_to_the_crowd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from specter.imaging import Field

    seen: dict = {}

    def draw(self: Field, **kwargs: object) -> None:
        seen["field"] = self
        raise _Captured

    monkeypatch.setattr(Field, "draw", draw)
    source_path = tmp_path / "source.cs"
    _write_pose_only_csfile(source_path, _ROTVECS)
    config = MicrographConfig(
        pdb_source=_PDB_FIXTURE,
        assembly=False,
        n_pixels=24,
        micrograph_size=48,
        pixel_size=3.0,
        ice_model="none",
        device="cpu",
        output_dir=str(tmp_path),
        orientation_distribution_path=str(source_path),
    )
    with pytest.raises(_Captured):
        run_micrograph(config)
    torch.testing.assert_close(
        seen["field"].crowding.orientations,
        roma.rotvec_to_unitquat(torch.as_tensor(_ROTVECS)),
    )


def _direction_angles(quaternions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = _viewing_directions(quaternions)
    return torch.atan2(d[:, 1], d[:, 0]), torch.asin(d[:, 2].clamp(-1, 1))


def test_histogram_draws_land_in_the_populated_bins() -> None:
    """Two bins hold 1:3 of the counts; the draws split the same way."""
    histogram = torch.zeros(6, 12)  # 30 degree bins
    histogram[4, 9] = 1.0  # elevation 30..60, azimuth 90..120
    histogram[1, 2] = 3.0  # elevation -60..-30, azimuth -120..-90
    drawn = rotations.sample_viewing_direction_histogram(
        histogram, 4000, generator=torch.Generator().manual_seed(4)
    )
    azimuth, elevation = (torch.rad2deg(a) for a in _direction_angles(drawn))

    first = (elevation > 30) & (elevation < 60) & (azimuth > 90) & (azimuth < 120)
    second = (elevation > -60) & (elevation < -30) & (azimuth > -120) & (azimuth < -90)
    assert (first | second).all()
    assert abs(first.float().mean() - 0.25) < 0.03


def test_a_solid_angle_weighted_histogram_gives_uniform_rotations() -> None:
    n_elevation, n_azimuth = 18, 36
    edges = torch.sin(
        -torch.pi / 2 + torch.arange(n_elevation + 1) * torch.pi / n_elevation
    )
    histogram = (edges[1:] - edges[:-1])[:, None].expand(n_elevation, n_azimuth)
    generator = torch.Generator().manual_seed(5)
    drawn = roma.unitquat_to_rotmat(
        rotations.sample_viewing_direction_histogram(
            histogram, 50_000, generator=generator
        )
    )
    uniform = rotations.random_rotation_matrix(50_000, generator=generator)

    assert drawn.mean(dim=0).abs().max() < 0.01
    # the rotation-angle distribution matches the Haar measure's
    levels = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])

    def trace(R: torch.Tensor) -> torch.Tensor:
        return R.diagonal(dim1=-2, dim2=-1).sum(-1)

    torch.testing.assert_close(
        torch.quantile(trace(drawn), levels),
        torch.quantile(trace(uniform), levels),
        atol=0.03,
        rtol=0,
    )


def test_histogram_sampling_rejects_bad_counts() -> None:
    with pytest.raises(ValueError, match="n_elevation, n_azimuth"):
        rotations.sample_viewing_direction_histogram(torch.ones(5), 3)
    with pytest.raises(ValueError, match="non-negative"):
        rotations.sample_viewing_direction_histogram(torch.zeros(4, 8), 3)
    with pytest.raises(ValueError, match="non-negative"):
        rotations.sample_viewing_direction_histogram(-torch.ones(4, 8), 3)


@pytest.mark.parametrize("suffix", [".npy", ".csv", ".txt"])
def test_read_orientations_reads_a_histogram_table(tmp_path: Path, suffix: str) -> None:
    """A binned distribution needs no particle file."""
    histogram = np.zeros((6, 12))
    histogram[4, 9] = 1.0  # elevation 30..60, azimuth 90..120
    path = tmp_path / f"histogram{suffix}"
    if suffix == ".npy":
        np.save(path, histogram)
    else:
        np.savetxt(path, histogram, delimiter="," if suffix == ".csv" else " ")

    drawn = read_orientations(path)
    azimuth, elevation = (torch.rad2deg(a) for a in _direction_angles(drawn))
    assert (elevation > 30).all() and (elevation < 60).all()
    assert (azimuth > 90).all() and (azimuth < 120).all()
    # the set depends on the file alone
    torch.testing.assert_close(read_orientations(path), drawn)


def test_run_particle_stack_draws_from_a_histogram(tmp_path: Path) -> None:
    histogram = np.zeros((6, 12))
    histogram[5, :] = 1.0  # elevation 60..90: views near the particle's +z
    path = tmp_path / "histogram.csv"
    np.savetxt(path, histogram, delimiter=",")
    config = ParticleStackConfig(
        pdb_source=_PDB_FIXTURE,
        assembly=False,
        n_pixels=48,
        n_particles=6,
        scattering_model="projection",
        ice_model="none",
        detector_model="none",
        device="cpu",
        batchsize=6,
        seed=3,
        output_dir=str(tmp_path),
        filename="binned",
        orientation_distribution_path=str(path),
    )
    run_particle_stack(config)

    df = starfile.read(tmp_path / "binned.star")
    euler = torch.as_tensor(
        df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(),
        dtype=torch.float32,
    )
    directions = _viewing_directions(roma.euler_to_unitquat("ZYZ", euler, degrees=True))
    assert (directions[:, 2] > np.sin(np.deg2rad(60)) - 1e-4).all()


def _write_bild(path: Path, directions: np.ndarray, counts: np.ndarray) -> None:
    """Arrows as CryoSPARC draws them: radial, from a sphere about the centre."""
    centre, radius, scale = np.array([72.0, 72.0, 72.0]), 72.0, 0.8
    lines = []
    for d, c in zip(directions, counts):
        d = d / np.linalg.norm(d)
        base, tip = centre + radius * d, centre + (radius + scale * c) * d
        lines += [
            ".color 0.2 0 0.8",
            ".cylinder " + " ".join(f"{v:.5f}" for v in [*base, *tip, 2.3]),
        ]
    path.write_text("\n".join(lines) + "\n")


def test_read_orientations_reads_a_bild_file(tmp_path: Path) -> None:
    """Draws follow the arrows' directions, weighted by the arrows' lengths."""
    directions = np.array([[0.0, 0.0, -1.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 0.2]])
    counts = np.array([1.0, 3.0, 4.0])
    path = tmp_path / "J1_viewing_direction_distribution_iteration_005.bild"
    _write_bild(path, directions, counts)

    drawn = read_orientations(path)
    unit = torch.as_tensor(
        directions / np.linalg.norm(directions, axis=1, keepdims=True)
    )
    cosines = _viewing_directions(drawn).double() @ unit.T
    best, which = cosines.max(dim=-1)
    # spread over each arrow's bin, which is far narrower than the arrows' spacing
    assert (best > np.cos(np.deg2rad(50))).all()
    fractions = torch.bincount(which, minlength=3).double() / len(which)
    torch.testing.assert_close(
        fractions, torch.as_tensor(counts / counts.sum()), atol=0.01, rtol=0
    )
    torch.testing.assert_close(read_orientations(path), drawn)


def test_a_single_arrow_bild_file_gives_its_direction(tmp_path: Path) -> None:
    path = tmp_path / "one.bild"
    _write_bild(path, np.array([[0.3, -0.4, 0.5]]), np.array([7.0]))
    expected = torch.tensor([0.3, -0.4, 0.5]) / torch.tensor([0.3, -0.4, 0.5]).norm()
    torch.testing.assert_close(
        _viewing_directions(read_orientations(path)),
        expected.expand(65536, 3),
        atol=1e-5,
        rtol=0,
    )


def test_a_bild_file_without_arrows_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.bild"
    path.write_text(".color 1 0 0\n.sphere 0 0 0 1\n")
    with pytest.raises(ValueError, match="no .cylinder"):
        read_orientations(path)
