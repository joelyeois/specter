"""
polnet-facing half of the cryoET specimen pipeline.

polnet is used purely for placement: it runs its non-overlapping,
occupancy-driven packing entirely at low resolution (cheap, fast -- avoids
ever allocating full-tomogram-sized bookkeeping arrays at the target
resolution) and hands back per-instance protein positions/rotations plus a
membrane surface mesh. It never renders the final density and its TEM/IMOD
code path is never invoked (no tem_file_path is ever supplied to SynthTomo).

Axis convention
----------------
VTK points (mesh vertices, motif x/y/z columns) are plain physical (x, y, z)
vectors -- there is no "storage order" ambiguity for a point. The ambiguity
only shows up once physical coordinates are used to index a numpy/torch
ARRAY: polnet's own arrays are (X, Y, Z) (axis0 = physical X -- see
polnet.lio's swapaxes(0, 2) on every load/save), while specter's arrays
(PotentialBuilder output, rotate_volume, crowding) are (Z, Y, X) (axis0 =
physical Z). This module keeps every position/rotation as a plain physical
vector for as long as possible and only reverses to (z, y, x) at the one
point each caller actually indexes a specter-native array (documented at
each call site in cryoet.py).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support

from polnet.utils import lio
from polnet.sample.sample import SyntheticSample

from ..pdb import PDB
from ..potential import PotentialBuilder


@dataclass
class ProteinInstance:
    """One placed copy of a protein species, in physical (Angstrom) units."""

    code: str  # species id (MMER_ID / PDB stem)
    position_xyz: np.ndarray  # (3,) physical x, y, z
    quat_wxyz: np.ndarray  # (4,) unit quaternion, scalar-first
    polymer: int  # SAWLC chain index -- only unique WITHIN the same `code`
    # (each protein species gets its own NetSAWLC/PnSAWLCNet, so e.g.
    # code="1bxn", polymer=0 and code="5mac", polymer=0 are unrelated
    # chains; group by (code, polymer) to identify a chain uniquely).


@dataclass
class MembraneInstance:
    """One fitted membrane, in physical (Angstrom) units.

    thick/layer_s are the midpoint of the configured range for the batch
    this membrane came from (not separately recoverable per-instance from
    polnet's public API when a batch places more than one membrane).
    """

    mb_type: str  # "sphere" | "ellipsoid" | "toroid"
    center_xyz: np.ndarray  # (3,)
    thick: float
    layer_s: float
    radius: float | None = None  # sphere
    semi_axes: np.ndarray | None = None  # (3,) a, b, c -- ellipsoid
    rot_matrix: np.ndarray | None = None  # (3,3) local-frame -> physical xyz
    rad_a: float | None = None  # toroid major radius
    rad_b: float | None = None  # toroid minor (tube) radius
    cf: float = (
        1.0  # density contrast factor, from MB_DEN_CF_RG (see fit_membrane_instances)
    )


# specter.potential.compute_supersampling_parameters's own default atomic
# kernel width (see potential.py: width_atom=5.0) -- each atom's own
# potential is evaluated over a +/-2.5 A box around it. The box built for a
# whole molecule needs at least this much margin beyond the outermost atom
# (at max_diameter/2 from center) or that atom's own kernel gets truncated
# by fftconvolve's mode="same" cropping (a real linear convolution, not
# circular -- see fft.py -- so under-margining truncates, it does not wrap
# around, but it does genuinely clip the kernel tail).
ATOM_KERNEL_HALF_WIDTH_A = 2.5


def estimate_protein_box_size(max_diameter: float, v_size: float) -> int:
    """Grid size (voxels, per axis) for a molecule with the given max
    diameter (Angstrom, from PDB.max_diameter -- true convex-hull max
    pairwise distance) at voxel size v_size.

    Margin is a fixed PHYSICAL distance (2x ATOM_KERNEL_HALF_WIDTH_A, i.e.
    the outermost atom's own kernel width, doubled for safety margin),
    converted to voxels -- not a fixed voxel count. A fixed voxel count
    would under-margin at fine resolutions (truncating atoms' own
    potential kernels) and over-pad at coarse ones (wasted compute); this
    scales correctly at any v_size. No separate minimum grid size is
    applied on top -- there was one (64 voxels) in an earlier version, but
    it wasn't derived from any real requirement (PotentialBuilder's own
    internal minimum is for the atomic kernel's supersampling grid, a
    different and much smaller number) and could force the low-res
    placeholder's box bigger than the low-res VOI itself for a small
    target volume, with no benefit once the margin above was fixed to be
    physically correct on its own.
    """
    margin_A = 2 * ATOM_KERNEL_HALF_WIDTH_A
    n = int(np.ceil((max_diameter + 2 * margin_A) / v_size))
    n += n % 2
    return n


def build_low_res_svol(
    pdb: PDB,
    mmer_id: str,
    v_size: float,
    scratch_dir: Path,
    parameterization: str = "shtyrov",
) -> str:
    """Build a small placeholder potential map for one PDB species and save
    it as an MRC for polnet's cytosolic-protein placement to reference as
    MMER_SVOL (polnet only needs this for its own collision geometry; the
    real high-resolution potential is built separately, later, at the
    target resolution).

    `pdb` is an already-constructed specter.pdb.PDB (built once from a PDB
    code and reused for both this low-res placeholder and the real high-res
    potential -- avoids re-fetching/re-parsing the same structure twice).

    Returns
    -------
    filename : the MRC filename, relative to scratch_dir (suitable as
        MMER_SVOL when data_path=scratch_dir).
    """
    n = estimate_protein_box_size(pdb.max_diameter, v_size)

    builder = PotentialBuilder(
        n_xyz=n,
        dx=v_size,
        atomic_numbers=pdb.atomic_numbers,
        progressbars=False,
        parameterization=parameterization,
    )
    vol = builder.forward(pdb.coordinates)
    vol_np = vol.detach().cpu().numpy().astype(np.float32)

    # vol_np is specter-native (Z, Y, X), which is ALSO raw-mrcfile's native
    # storage order -- so no axis swap here (no_saxes=False). polnet's own
    # PnGen loader applies its own swapaxes(0, 2) when it reads this file
    # back, converting it into polnet's (X, Y, Z) convention for its
    # internal placement math -- consistent on both ends.
    filename = f"{mmer_id}_lowres.mrc"
    lio.write_mrc(vol_np, str(scratch_dir / filename), v_size=v_size, no_saxes=False)
    return filename


def build_protein_params(spec: dict, mmer_svol_filename: str) -> dict:
    """Build a .pns-equivalent params dict for polnet's add_set_cproteins
    from a user-supplied protein spec.

    `spec` is a single, self-contained dict (must include "PDB_CODE"; other
    keys use polnet's own PMER_*/MMER_* naming and defaults if omitted) --
    matches membrane_specs' convention of one complete dict per species,
    rather than a separate code list + override dict.
    """
    return {
        "MMER_ID": spec["PDB_CODE"],
        "MMER_SVOL": mmer_svol_filename,
        "MMER_ISO": spec.get("MMER_ISO", 0.1),
        "PMER_L": spec.get("PMER_L", 1.2),
        "PMER_L_MAX": spec.get("PMER_L_MAX", 3000.0),
        "PMER_OCC": spec.get("PMER_OCC", 0.5),
        "PMER_OVER_TOL": spec.get("PMER_OVER_TOL", 0.01),
    }


def build_low_res_sample_with_membranes(
    shape: tuple[int, int, int],
    v_size: float,
    offset: tuple[int, int, int],
    membrane_params: list[dict],
    max_mbtries: int = 10,
) -> SyntheticSample:
    """Construct the low-res SyntheticSample and run polnet's membrane
    placement into it (populates sample's VOI/occupancy grid and
    sample.mbs_vtp). Depends only on shape/v_size/membrane_params -- not on
    any protein SVOL, so this is safe to run concurrently (e.g. on a
    background thread) with add_proteins_to_sample's SVOL-building
    prerequisite (build_low_res_svol) as long as add_proteins_to_sample
    itself is only called after this function returns (protein packing
    reads the membrane-updated occupancy grid to avoid overlapping placed
    membranes).
    """
    sample = SyntheticSample(shape=shape, v_size=v_size, offset=offset)
    for params in membrane_params:
        sample.add_set_membranes(params=params, max_mbtries=max_mbtries)
    return sample


def add_proteins_to_sample(
    sample: SyntheticSample,
    protein_params: list[dict],
    scratch_dir: Path,
    mmer_tries: int = 20,
    pmer_tries: int = 100,
    surf_dec: float = 0.9,
) -> None:
    """Pack cytosolic proteins into an already-built sample (mutates it in
    place). Must run after any membrane placement on this sample -- each
    species' packing reads/updates the shared occupancy grid, so this loop
    itself stays strictly sequential across protein_params.
    """
    for params in protein_params:
        sample.add_set_cproteins(
            params=params,
            data_path=scratch_dir,
            surf_dec=surf_dec,
            mmer_tries=mmer_tries,
            pmer_tries=pmer_tries,
        )


def build_low_res_sample(
    shape: tuple[int, int, int],
    v_size: float,
    offset: tuple[int, int, int],
    membrane_params: list[dict],
    protein_params: list[dict],
    scratch_dir: Path,
    max_mbtries: int = 10,
    mmer_tries: int = 20,
    pmer_tries: int = 100,
    surf_dec: float = 0.9,
) -> SyntheticSample:
    """Run polnet's placement entirely at low resolution and return the
    populated SyntheticSample. No TEM config is ever supplied -- IMOD is
    never touched.

    Purely sequential convenience wrapper around
    build_low_res_sample_with_membranes + add_proteins_to_sample; callers
    that want membrane placement to run concurrently with protein SVOL
    building (see CryoETSpecimenGenerator._run_low_res_placement) should
    call those two functions directly instead.
    """
    sample = build_low_res_sample_with_membranes(
        shape=shape,
        v_size=v_size,
        offset=offset,
        membrane_params=membrane_params,
        max_mbtries=max_mbtries,
    )
    add_proteins_to_sample(
        sample,
        protein_params,
        scratch_dir,
        mmer_tries=mmer_tries,
        pmer_tries=pmer_tries,
        surf_dec=surf_dec,
    )
    return sample


def get_protein_instances(sample: SyntheticSample) -> list[ProteinInstance]:
    """Extract per-instance protein placements from sample.motifs.

    Positions are returned as plain physical (x, y, z) vectors -- callers
    reverse to (z, y, x) only when indexing a specter-native array.
    """
    motifs = sample.motifs
    rows = motifs[motifs["type"] == "cprotein"]
    return [
        ProteinInstance(
            code=str(row["code"]),
            position_xyz=np.array([row["x"], row["y"], row["z"]], dtype=float),
            quat_wxyz=np.array(
                [row["q1"], row["q2"], row["q3"], row["q4"]], dtype=float
            ),
            polymer=int(row["polymer"]),
        )
        for row in rows
    ]


def _split_mesh_components(vtp: vtk.vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
    """Split a merged polydata into per-component point clusters.

    polnet merges each membrane's marching-cubes surface into one polydata
    via vtkAppendPolyData, which does not weld coincident points -- without
    cleaning first, vtkPolyDataConnectivityFilter sees each membrane's
    surface fragmented into many spurious "components" at every duplicated
    vertex, instead of one component per actual membrane.

    Returns (points, region_ids): points is (N, 3) physical xyz, region_ids
    is (N,) giving each point's connected-component index.
    """
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(vtp)
    cleaner.Update()

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputConnection(cleaner.GetOutputPort())
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    labeled = conn.GetOutput()
    points = numpy_support.vtk_to_numpy(labeled.GetPoints().GetData())
    region_ids = numpy_support.vtk_to_numpy(labeled.GetPointData().GetArray("RegionId"))
    return points, region_ids


def _fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    radius = float(np.linalg.norm(points - center, axis=1).mean())
    return center, radius


def _fit_ellipsoid(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (center, semi_axes, rot_matrix). rot_matrix's columns are the
    principal axis directions in physical xyz, i.e. it maps a point in the
    ellipsoid's own axis-aligned local frame to physical xyz.
    """
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # order by descending eigenvalue for a deterministic a >= b >= c convention
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    proj = centered @ eigvecs
    semi_axes = np.abs(proj).max(axis=0)
    # ensure a proper rotation (det +1), flip one axis if needed
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1
    return center, semi_axes, eigvecs


def _fit_torus(points: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Returns (center, rad_a, rad_b, rot_matrix). rot_matrix's 3rd column is
    the torus's symmetry (hole) axis in physical xyz.
    """
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # The symmetry axis has the SMALLEST point spread (points lie in a thin
    # band around the equatorial plane relative to the hole axis).
    order = np.argsort(eigvals)  # ascending: axis (smallest) first
    eigvecs = eigvecs[:, order]
    axis_dir = eigvecs[:, 0]
    # in-plane basis
    u = eigvecs[:, 1]
    v = eigvecs[:, 2]
    z_local = centered @ axis_dir
    x_local = centered @ u
    y_local = centered @ v
    rho = np.sqrt(x_local**2 + y_local**2)  # distance from hole axis, in-plane

    # Algebraic circle fit of (rho, z_local) to recover (rad_a, rad_b):
    # points satisfy (rho - rad_a)^2 + z_local^2 = rad_b^2, i.e. a circle in
    # the (rho, z) half-plane centered at (rad_a, 0).
    A = np.column_stack([rho, z_local, np.ones_like(rho)])
    b = rho**2 + z_local**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    rad_a = float(sol[0] / 2.0)
    rad_b = float(np.sqrt(sol[2] + rad_a**2))

    rot_matrix = np.column_stack([u, v, axis_dir])
    return center, rad_a, rad_b, rot_matrix


def fit_membrane_instances(
    sample: SyntheticSample,
    mb_type: str,
    thick_rg: tuple[float, float],
    layer_s_rg: tuple[float, float],
    den_cf_rg: tuple[float, float] = (1.0, 1.0),
) -> list[MembraneInstance]:
    """Recover each individual membrane's analytic parameters by fitting
    against sample.mbs_vtp (the only place polnet's public API exposes
    membrane geometry -- per-instance center/radius/rotation are not
    retained on any object once generate_set() returns).

    thick/layer_s use the midpoint of the configured range: not separately
    recoverable per-instance when a batch places more than one membrane.

    den_cf_rg (MB_DEN_CF_RG) is likewise drawn ONCE per call and shared by
    every instance returned from it -- this matches polnet's own semantics:
    MbGen.generate_set() draws a single cf = self.rnd_cf() per *batch* of
    same-type membranes and applies it uniformly to that batch's whole
    aggregated density, not per individual membrane.

    Known limitation: membranes configured to physically touch/overlap
    (via over_tolerance) may fuse into one connected component and fit
    incorrectly as a single blended shape.
    """
    vtp = sample.mbs_vtp
    if vtp is None or vtp.GetNumberOfPoints() == 0:
        return []

    points, region_ids = _split_mesh_components(vtp)
    thick = 0.5 * (thick_rg[0] + thick_rg[1])
    layer_s = 0.5 * (layer_s_rg[0] + layer_s_rg[1])
    cf = random.uniform(den_cf_rg[0], den_cf_rg[1])

    instances = []
    for region in np.unique(region_ids):
        sub = points[region_ids == region]
        if mb_type == "sphere":
            center, radius = _fit_sphere(sub)
            instances.append(
                MembraneInstance(
                    mb_type="sphere",
                    center_xyz=center,
                    thick=thick,
                    layer_s=layer_s,
                    radius=radius,
                    cf=cf,
                )
            )
        elif mb_type == "ellipsoid":
            center, semi_axes, rot = _fit_ellipsoid(sub)
            instances.append(
                MembraneInstance(
                    mb_type="ellipsoid",
                    center_xyz=center,
                    thick=thick,
                    layer_s=layer_s,
                    semi_axes=semi_axes,
                    rot_matrix=rot,
                    cf=cf,
                )
            )
        elif mb_type == "toroid":
            center, rad_a, rad_b, rot = _fit_torus(sub)
            instances.append(
                MembraneInstance(
                    mb_type="toroid",
                    center_xyz=center,
                    thick=thick,
                    layer_s=layer_s,
                    rad_a=rad_a,
                    rad_b=rad_b,
                    rot_matrix=rot,
                    cf=cf,
                )
            )
        else:
            raise ValueError(
                f"Unsupported membrane type for geometric fitting: {mb_type!r}"
            )
    return instances
