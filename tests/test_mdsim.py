"""
Tests for MDSimDump's LAMMPS dump parsing.
"""

import pytest
import torch

from specter.ice._mdsim import MDSimDump

_DUMP_WITH_TYPE = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 6.0 7.0 8.0
"""

_DUMP_WITHOUT_TYPE = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id x y z c_1
1 6.0 7.0 8.0 -0.403421
"""


@pytest.mark.parametrize("dump_text", [_DUMP_WITH_TYPE, _DUMP_WITHOUT_TYPE])
def test_trim_frame_locates_xyz_regardless_of_column_layout(tmp_path, dump_text):
    # Box is 0-10 in each axis (center 5,5,5); the atom sits at (6,7,8), so a
    # correct parse must recover (1,2,3) after centering — whether or not the
    # dump has a "type" column between id and x (e.g. "id type x y z" vs
    # "id x y z c_1", both real LAMMPS dump conventions).
    filepath = tmp_path / "test.dump"
    filepath.write_text(dump_text)

    mdsim = MDSimDump(str(filepath), n=4, dx=0.5, trim_size=20.0)
    with pytest.warns(UserWarning, match="pre-steady-state"):
        coords = mdsim.get_coordinates(frames=0)

    assert len(coords) == 1
    torch.testing.assert_close(coords[0], torch.tensor([[1.0, 2.0, 3.0]]))


def test_coord_column_start_raises_on_missing_xyz(tmp_path):
    bad_dump = _DUMP_WITH_TYPE.replace("id type x y z", "id type xu yu zu")
    filepath = tmp_path / "bad.dump"
    filepath.write_text(bad_dump)

    with pytest.raises(ValueError, match="Could not find an 'x' column"):
        MDSimDump(str(filepath))


def test_compute_sk_3d_dc_is_sqrt_n_atoms(tmp_path):
    # S(k) = |FFT3(vox)|^2 / N by definition, so sqrt(S(0)) must equal
    # sqrt(N) regardless of how the N atoms are arranged in space (the DC
    # term of a voxel grid's FFT is just the grid's total mass, which
    # soft-voxelization conserves exactly as long as atoms aren't clipped by
    # the trim/grid boundary). A cluster of 9 atoms placed well inside the
    # grid should therefore give a DC value of exactly sqrt(9) = 3 — this is
    # a regression test for the per-frame sqrt(N) normalization that
    # compute_sk_3d/compute_sk_radial must apply before averaging over
    # frames (see interpolate_target_kernel, which rescales this profile by
    # sqrt(n_ice_molecules) of a *different*, target atom count).
    n_atoms = 9
    offsets = [
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
        (0, 0, 0),
    ]
    lines = [
        "ITEM: TIMESTEP",
        "0",
        "ITEM: NUMBER OF ATOMS",
        str(n_atoms),
        "ITEM: BOX BOUNDS pp pp pp",
        "0.0 20.0",
        "0.0 20.0",
        "0.0 20.0",
        "ITEM: ATOMS id type x y z",
    ]
    for i, (dx_, dy_, dz_) in enumerate(offsets):
        lines.append(f"{i + 1} 1 {10 + dx_} {10 + dy_} {10 + dz_}")
    filepath = tmp_path / "cluster.dump"
    filepath.write_text("\n".join(lines) + "\n")

    mdsim = MDSimDump(str(filepath), n=16, dx=1.0, trim_size=16.0)
    with pytest.warns(UserWarning, match="pre-steady-state"):
        sk3d = mdsim.compute_sk_3d(frames=0)

    dc = sk3d[8, 8, 8]  # center voxel post-fftshift for n=16
    torch.testing.assert_close(dc, torch.tensor(float(n_atoms) ** 0.5))
