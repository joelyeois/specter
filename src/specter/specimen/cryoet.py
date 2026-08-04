"""
CryoETSpecimenGenerator: builds a large, high-resolution cryoET specimen
(proteins from PDB files + membranes) by running polnet's placement/packing
logic entirely at low resolution, then rendering the real density with
specter's own potential builder and rotation/insertion machinery.

Scope for this pass (see the approved plan): cytosolic proteins + membranes
only. Membrane-embedded proteins, ice/solvent filling, and TEM/IMOD
simulation are explicitly out of scope.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import roma
import torch
from polnet.logging_conf import setup_logger as _setup_polnet_logger

from ..arrays import clip_insert_bounds
from ..potential import PotentialBuilder
from ..pdb import PDB
from ..rotations import build_affine_matrix, rotate_volume
from . import polnet_bridge as pb
from .filament import FilamentSpec, place_filaments
from .membrane._placement import align_principal_axis_to_z

# Fine-tuning multiplier applied on top of the physically-derived membrane
# potential baseline (see _estimate_organic_potential_reference): the
# membrane density is calibrated to the actual protein templates' own
# scattering-potential magnitude (proteins and lipid bilayers are both
# ordinary, densely-packed organic matter), then modulated per-spec by
# MB_DEN_CF_RG (matching polnet's own contrast-factor semantics). This
# constant is left at 1.0 (no extra adjustment) by default; it doubles as a
# plain override when no proteins are placed at all (nothing to calibrate
# against). Still an approximation -- lipid elemental composition/packing
# differs somewhat from protein -- but far better grounded than a bare
# arbitrary constant.
DEFAULT_MEMBRANE_POTENTIAL_SCALE = 1.0


class CryoETSpecimenGenerator:
    """
    Parameters
    ----------
    protein_specs : list of dict
        One self-contained dict per protein species, e.g.
        ``{"PDB_CODE": "1bxn", "PMER_OCC": 3.0, "PMER_L": 1.2,
        "PMER_L_MAX": 3000.0, "PMER_OVER_TOL": 0.01, "MMER_ISO": 0.1}``.
        Only "PDB_CODE" is required; the rest fall back to polnet's own
        defaults if omitted (see build_protein_params). PDB codes are
        fetched automatically from RCSB on first use (via specter.pdb.PDB)
        and cached in pdb_cache_dir; each is fetched/parsed once and reused
        for both the low-res placeholder and the real high-res potential.
        Same shape as membrane_specs below (one complete dict per species/
        population), rather than a separate code list + override dict.
    membrane_specs : list of dict
        polnet .mbs-equivalent param dicts, e.g.
        ``{"MB_TYPE": "sphere", "MB_THICK_RG": (35, 45), "MB_LAYER_S_RG":
        (8, 12), "MB_OCC_RG": (0.001, 0.003), "MB_OVER_TOL": 0.0,
        "MB_DEN_CF_RG": (0.3, 0.5), "MB_MIN_RAD": 150.0, "MB_MAX_RAD": 300.0}``
        (ellipsoid uses MB_MIN_AXIS/MB_MAX_AXIS/MB_MAX_ECC instead of
        MB_MIN_RAD/MB_MAX_RAD; toroid uses MB_MIN_RAD/MB_MAX_RAD).
    filament_specs : list of dict, optional
        One dict per filament species, converted to a ``filament.FilamentSpec``
        (see there for field meanings), e.g. ``{"code": "1TUB", "step": 85.0,
        "flex_deg": 3.0, "n_filaments": 4}``, or just pass
        ``dataclasses.asdict(filament.MICROTUBULE_SPEC)`` /
        ``dataclasses.asdict(filament.ACTIN_SPEC)`` for the built-in
        microtubule/actin presets. Unlike proteins/membranes, filaments are
        placed by specter's own random-walk code (``filament.place_filaments``),
        not polnet -- no low-res placement pass, no occupancy-aware packing
        against the other species. Default None (no filaments).
    target_shape : tuple of int
        Output volume shape (Z, Y, X) in voxels, at target_v_size.
    target_v_size : float
        Target voxel size in Angstrom.
    low_res_v_size : float, optional
        Voxel size used for polnet's placement pass. Default 10 A.
    low_res_shape : tuple of int, optional
        VOI shape for the placement pass. Default: same physical box as
        target_shape/target_v_size, resampled at low_res_v_size.
    scratch_dir : str or Path, optional
        Directory for low-res placeholder MRCs. Defaults to a temp dir.
    pdb_cache_dir : str or Path, optional
        Directory for downloaded PDB/mmCIF files. Defaults to specter's
        usual "../pdb-data/" (see specter.pdb.PDB).
    membrane_potential_scale : float, optional
        See DEFAULT_MEMBRANE_POTENTIAL_SCALE above.
    parameterization : str, optional
        Atomic scattering-factor parameterization for PotentialBuilder
        ("shtyrov", "kirkland", or "lobato"). Default "shtyrov", matching
        `PotentialBuilder`'s own default.
    seed : int, optional
        Random seed for polnet's placement.
    verbose : bool, optional
        Print pipeline progress (species fetched, sample built, membrane/
        protein counts, timing per stage) and enable polnet's own INFO-level
        console logging (polnet has a real logger -- see polnet.logging_conf
        -- but ships with console_level=WARNING and no handler attached by
        default, so its own progress messages are silently dropped unless
        something calls setup_logger(); this does that). Default True.
    """

    def __init__(
        self,
        protein_specs: list[dict],
        membrane_specs: list[dict],
        target_shape: tuple[int, int, int],
        target_v_size: float,
        filament_specs: list[dict] | None = None,
        low_res_v_size: float = 10.0,
        low_res_shape: tuple[int, int, int] | None = None,
        scratch_dir: str | Path | None = None,
        pdb_cache_dir: str | Path | None = None,
        membrane_potential_scale: float = DEFAULT_MEMBRANE_POTENTIAL_SCALE,
        parameterization: str = "shtyrov",
        seed: int | None = None,
        verbose: bool = True,
    ):
        self.protein_specs = protein_specs
        self.membrane_specs = membrane_specs
        self.filament_specs = [FilamentSpec(**spec) for spec in (filament_specs or [])]
        self.target_shape = target_shape  # (Z, Y, X)
        self.target_v_size = target_v_size
        self.low_res_v_size = low_res_v_size
        self.pdb_cache_dir = str(pdb_cache_dir) if pdb_cache_dir else "../pdb-data/"
        self.membrane_potential_scale = membrane_potential_scale
        self.parameterization = parameterization
        self.seed = seed
        self.verbose = verbose

        if low_res_shape is None:
            # Same physical box, at low_res_v_size, in polnet's (X, Y, Z) order
            phys = (
                np.array(target_shape[::-1]) * target_v_size
            )  # -> (X,Y,Z) physical extent
            lx, ly, lz = (int(np.ceil(p / low_res_v_size)) for p in phys)
            low_res_shape = (lx, ly, lz)
        self.low_res_shape = low_res_shape  # polnet's (X, Y, Z)

        self._scratch_dir = Path(scratch_dir) if scratch_dir else None
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._pdb_cache: dict[str, PDB] = {}

        if self.verbose:
            _setup_polnet_logger(self._scratch(), console_level=logging.INFO)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _scratch(self) -> Path:
        if self._scratch_dir is not None:
            self._scratch_dir.mkdir(parents=True, exist_ok=True)
            return self._scratch_dir
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="cryoet_specimen_")
        return Path(self._tmpdir.name)

    def _get_pdb(self, code: str) -> PDB:
        """Fetch+parse a PDB code once, cached for reuse across the low-res
        placeholder and the real high-res potential build.
        """
        if code not in self._pdb_cache:
            Path(self.pdb_cache_dir).mkdir(parents=True, exist_ok=True)
            # Raw fetch/assembly/progress-bar chatter is silenced regardless of
            # self.verbose -- the per-species "N atoms, built in Xs" summary
            # line below already covers this, and with a few dozen species
            # (targets + cytosolic filler) the underlying PDB fetch mechanics
            # noise multiplies into far more lines than it's worth.
            self._pdb_cache[code] = PDB(
                code, savefolder=self.pdb_cache_dir, verbose=False
            )
        return self._pdb_cache[code]

    def _build_protein_svols(self, scratch: Path) -> list[dict]:
        """Build each protein species' low-res SVOL placeholder and return
        the corresponding .pns-equivalent param dicts, in protein_specs
        order. Runs on its own thread from _run_low_res_placement,
        concurrently with polnet's membrane placement -- see the comment
        there for why that's safe.
        """
        protein_param_dicts = []
        for spec in self.protein_specs:
            code = spec["PDB_CODE"]
            t0 = time.time()
            pdb = self._get_pdb(code)
            filename = pb.build_low_res_svol(
                pdb,
                code,
                self.low_res_v_size,
                scratch,
                parameterization=self.parameterization,
            )
            self._log(
                f"[cryoet]   {code}: {pdb.atomic_numbers.shape[0]} atoms, "
                f"low-res placeholder built in {time.time() - t0:.1f}s"
            )
            protein_param_dicts.append(pb.build_protein_params(spec, filename))
        return protein_param_dicts

    # ------------------------------------------------------------------
    # Step 1-3: low-resolution placement via polnet
    # ------------------------------------------------------------------
    def _run_low_res_placement(self):
        if self.seed is not None:
            import random

            random.seed(self.seed)
            np.random.seed(self.seed)

        self._log(
            f"[cryoet] low-res placement: {self.low_res_shape} voxels "
            f"@ {self.low_res_v_size} A/vx, {len(self.protein_specs)} protein "
            f"species, {len(self.membrane_specs)} membrane spec(s)"
        )
        scratch = self._scratch()
        offset = (4, 4, 4)

        # Membrane placement (polnet's own occupancy-driven packing) and
        # building each protein species' low-res SVOL placeholder
        # (specter's PotentialBuilder) touch disjoint state -- the former
        # only mutates a fresh SyntheticSample, the latter only reads PDB
        # data and writes per-species MRC files to scratch -- so they run
        # concurrently on background threads. Only the protein *packing*
        # step (add_set_cproteins) actually needs both results together: it
        # reads the membrane-updated occupancy grid off the sample AND the
        # SVOL filenames, so it has to wait for both and stays sequential
        # (see build_low_res_sample_with_membranes/add_proteins_to_sample
        # docstrings in polnet_bridge.py).
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            membrane_future = pool.submit(
                pb.build_low_res_sample_with_membranes,
                shape=self.low_res_shape,
                v_size=self.low_res_v_size,
                offset=offset,
                membrane_params=self.membrane_specs,
            )
            svol_future = pool.submit(self._build_protein_svols, scratch)

            sample = membrane_future.result()
            protein_param_dicts = svol_future.result()
        self._log(
            f"[cryoet] membrane placement + protein SVOL builds done in "
            f"{time.time() - t0:.1f}s (ran concurrently)"
        )

        self._log("[cryoet] running polnet protein packing...")
        t0 = time.time()
        pb.add_proteins_to_sample(sample, protein_param_dicts, scratch)
        self._log(f"[cryoet] polnet protein packing done in {time.time() - t0:.1f}s")

        protein_instances = pb.get_protein_instances(sample)
        self._log(f"[cryoet] {len(protein_instances)} protein instance(s) placed")
        if self.verbose and protein_instances:
            self._log(
                f"[cryoet]   chain sizes (SAWLC): {_chain_size_summary(protein_instances)}"
            )

        membrane_instances = []
        for spec in self.membrane_specs:
            membrane_instances.extend(
                pb.fit_membrane_instances(
                    sample,
                    mb_type=spec["MB_TYPE"],
                    thick_rg=spec["MB_THICK_RG"],
                    layer_s_rg=spec["MB_LAYER_S_RG"],
                    den_cf_rg=spec.get("MB_DEN_CF_RG", (1.0, 1.0)),
                )
            )
        self._log(f"[cryoet] {len(membrane_instances)} membrane instance(s) fitted")

        return protein_instances, membrane_instances

    # ------------------------------------------------------------------
    # Step 4: filament placement (specter-native, no polnet involved)
    # ------------------------------------------------------------------
    def _run_filament_placement(self) -> list:
        if not self.filament_specs:
            return []
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        instances = place_filaments(
            self.filament_specs, self.target_shape, self.target_v_size, generator
        )
        self._log(
            f"[cryoet] {len(instances)} filament monomer instance(s) placed "
            f"({len(self.filament_specs)} species)"
        )
        return instances

    # ------------------------------------------------------------------
    # Step 5: high-resolution assembly
    # ------------------------------------------------------------------
    def generate(self) -> torch.Tensor:
        """Run the full pipeline and return the assembled high-resolution
        specimen volume, shape target_shape, specter-native (Z, Y, X).

        Also stores the raw placements on self.protein_instances /
        self.membrane_instances / self.filament_instances (lists of
        ProteinInstance/MembraneInstance, see polnet_bridge.py, and
        FilamentInstance, see filament/_generator.py) after this call, e.g.
        to inspect/plot SAWLC chain membership yourself (group by (code,
        polymer)) rather than just trusting the verbose-mode log summary.
        """
        protein_instances, membrane_instances = self._run_low_res_placement()
        self.protein_instances = protein_instances
        self.membrane_instances = membrane_instances
        self.filament_instances = self._run_filament_placement()

        self._log(
            f"[cryoet] high-res assembly: {self.target_shape} voxels "
            f"@ {self.target_v_size} A/vx"
        )
        volume = torch.zeros(self.target_shape, dtype=torch.float32)

        self._stamp_proteins(volume, protein_instances)
        self._stamp_membranes(volume, membrane_instances)
        self._stamp_filaments(volume, self.filament_instances)

        return volume

    def export_picks(
        self,
        output_dir: str | Path,
        annotation_version: str = "1.0",
        oriented: bool = True,
        include_membranes: bool = True,
        include_filaments: bool = True,
    ) -> dict[str, Path]:
        """Write one copick/CryoET-Data-Portal-style .ndjson pick file per
        placed species, matching the real portal's point-pick schema
        (one JSON object per line: {"type": "point"|"orientedPoint",
        "location": {"x", "y", "z"}[, "xyz_rotation_matrix"]}).

        Coordinates are physical Angstrom (x, y, z), in the same absolute
        frame as the generated volume -- i.e. voxel index (z, y, x) =
        (x, y, z)[::-1] / target_v_size, matching how _stamp_proteins/
        _stamp_membranes insert them. Must be called after generate().

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Directory to write the .ndjson files into.
        annotation_version : str, optional
            Used only in the output filename, matching the portal's
            ``"{name}-{version}_{type}.ndjson"`` convention.
        oriented : bool, optional
            If True, protein picks are written as ``"orientedPoint"`` with
            each instance's rotation matrix included; if False, as plain
            ``"point"`` (location only).
        include_membranes : bool, optional
            If True, also write one ``"point"`` file with each fitted
            membrane's center (no orientation/size -- just a location,
            since membranes aren't point particles).
        include_filaments : bool, optional
            If True, also write one ``"point"``/``"orientedPoint"`` file per
            filament species, one row per monomer instance (same
            oriented/point choice as proteins, via each instance's own
            rotation_matrix -- no quaternion round-trip needed since
            FilamentInstance already stores rotation_matrix directly).

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping of species/"membrane" to written file path.
        """
        if not hasattr(self, "protein_instances"):
            raise RuntimeError("call generate() before export_picks()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        by_code: dict[str, list] = {}
        for inst in self.protein_instances:
            by_code.setdefault(inst.code, []).append(inst)

        point_type = "orientedPoint" if oriented else "point"
        for code, insts in by_code.items():
            path = (
                output_dir / f"{code}-{annotation_version}_{point_type.lower()}.ndjson"
            )
            with open(path, "w") as f:
                for inst in insts:
                    x, y, z = (float(v) for v in inst.position_xyz)
                    row = {"type": point_type, "location": {"x": x, "y": y, "z": z}}
                    if oriented:
                        quat_xyzw = roma.quat_wxyz_to_xyzw(
                            torch.as_tensor(inst.quat_wxyz, dtype=torch.float32)
                        )
                        R = roma.unitquat_to_rotmat(quat_xyzw)
                        row["xyz_rotation_matrix"] = R.numpy().tolist()
                    f.write(json.dumps(row) + "\n")
            written[code] = path

        if include_membranes and self.membrane_instances:
            path = output_dir / f"membrane-{annotation_version}_point.ndjson"
            with open(path, "w") as f:
                for inst in self.membrane_instances:
                    x, y, z = (float(v) for v in inst.center_xyz)
                    f.write(
                        json.dumps(
                            {"type": "point", "location": {"x": x, "y": y, "z": z}}
                        )
                        + "\n"
                    )
            written["membrane"] = path

        if include_filaments and getattr(self, "filament_instances", None):
            by_filament_code: dict[str, list] = {}
            for inst in self.filament_instances:
                by_filament_code.setdefault(inst.code, []).append(inst)
            for code, insts in by_filament_code.items():
                path = (
                    output_dir
                    / f"{code}-{annotation_version}_{point_type.lower()}.ndjson"
                )
                with open(path, "w") as f:
                    for inst in insts:
                        x, y, z = (float(v) for v in inst.position_xyz)
                        row = {
                            "type": point_type,
                            "location": {"x": x, "y": y, "z": z},
                        }
                        if oriented:
                            row["xyz_rotation_matrix"] = (
                                inst.rotation_matrix.numpy().tolist()
                            )
                        f.write(json.dumps(row) + "\n")
                written[code] = path

        return written

    def _stamp_proteins(self, volume: torch.Tensor, instances: list) -> None:
        # One high-res potential per unique species, built once.
        species = sorted({inst.code for inst in instances})
        templates: dict[str, torch.Tensor] = {}
        for code in species:
            t0 = time.time()
            pdb = self._get_pdb(code)  # reuses the cached fetch/parse from low-res
            n = pb.estimate_protein_box_size(pdb.max_diameter, self.target_v_size)
            builder = PotentialBuilder(
                n_xyz=n,
                dx=self.target_v_size,
                atomic_numbers=pdb.atomic_numbers,
                progressbars=False,
                parameterization=self.parameterization,
            )
            # specter-native (Z, Y, X), matches rotate_volume's expectation.
            templates[code] = builder.forward(pdb.coordinates)
            self._log(
                f"[cryoet]   {code}: high-res potential ({n}^3 @ "
                f"{self.target_v_size} A/vx) built in {time.time() - t0:.1f}s"
            )

        self._organic_potential_reference = _estimate_organic_potential_reference(
            templates
        )
        if templates:
            self._log(
                f"[cryoet] organic-matter potential reference: "
                f"{self._organic_potential_reference:.3g} "
                f"(mean over-half-max value, averaged across {len(templates)} "
                f"protein species -- used to calibrate membrane voxel values)"
            )

        t0 = time.time()
        for inst in instances:
            template = templates[inst.code]
            quat_xyzw = roma.quat_wxyz_to_xyzw(
                torch.as_tensor(inst.quat_wxyz, dtype=torch.float32)
            )
            R = roma.unitquat_to_rotmat(quat_xyzw)
            theta = build_affine_matrix(R)
            rotated = rotate_volume(template, theta, padding_mode="zeros")[0]

            # Position: polnet's motif columns are physical (x, y, z); the
            # target volume is specter-native (Z, Y, X) -- reverse here, the
            # one place this conversion needs to happen.
            center_vx_xyz = inst.position_xyz / self.target_v_size
            center_vx_zyx = center_vx_xyz[::-1]
            _insert_clipped(volume, rotated, center_vx_zyx)
        self._log(
            f"[cryoet] stamped {len(instances)} protein instance(s) "
            f"in {time.time() - t0:.1f}s"
        )

    def _stamp_membranes(self, volume: torch.Tensor, instances: list) -> None:
        # Physically-grounded baseline: membranes are ordinary organic matter
        # (lipid C/H/N/O), so their typical local scattering potential should
        # be comparable in magnitude to a protein's -- estimated from the
        # actual protein templates in _stamp_proteins (1.0 if no proteins
        # were placed at all, e.g. a membrane-only specimen; membrane_
        # potential_scale is then the only lever available and acts as a
        # plain override rather than a fine-tuning multiplier on top).
        # inst.cf (MB_DEN_CF_RG) modulates it per membrane spec, matching
        # polnet's own contrast-factor semantics.
        reference = getattr(self, "_organic_potential_reference", 1.0)
        t0 = time.time()
        for inst in instances:
            local = _build_membrane_density_zyx(inst, self.target_v_size)
            local *= reference * inst.cf * self.membrane_potential_scale
            center_vx_zyx = (inst.center_xyz / self.target_v_size)[::-1]
            _insert_clipped(
                volume, torch.as_tensor(local, dtype=torch.float32), center_vx_zyx
            )
        self._log(
            f"[cryoet] stamped {len(instances)} membrane instance(s) "
            f"in {time.time() - t0:.1f}s"
        )

    def _stamp_filaments(self, volume: torch.Tensor, instances: list) -> None:
        # One high-res potential per unique monomer species, built once --
        # same caching shape as _stamp_proteins, except the coordinates are
        # first PCA-realigned (align_principal_axis_to_z) so the monomer's
        # own longest axis -- its stacking axis, see filament/_placement.py
        # -- lands along local +Z before rendering, matching what
        # filament_orientations' rotations assume they're rotating.
        species = sorted({inst.code for inst in instances})
        templates: dict[str, torch.Tensor] = {}
        for code in species:
            t0 = time.time()
            pdb = self._get_pdb(code)
            n = pb.estimate_protein_box_size(pdb.max_diameter, self.target_v_size)
            builder = PotentialBuilder(
                n_xyz=n,
                dx=self.target_v_size,
                atomic_numbers=pdb.atomic_numbers,
                progressbars=False,
                parameterization=self.parameterization,
            )
            aligned_coordinates = align_principal_axis_to_z(pdb.coordinates)
            templates[code] = builder.forward(aligned_coordinates)
            self._log(
                f"[cryoet]   {code}: high-res filament monomer potential "
                f"({n}^3 @ {self.target_v_size} A/vx) built in "
                f"{time.time() - t0:.1f}s"
            )

        t0 = time.time()
        for inst in instances:
            template = templates[inst.code]
            theta = build_affine_matrix(inst.rotation_matrix)
            rotated = rotate_volume(template, theta, padding_mode="zeros")[0]
            center_vx_zyx = torch.flip(
                inst.position_xyz / self.target_v_size, dims=[0]
            ).numpy()
            _insert_clipped(volume, rotated, center_vx_zyx)
        self._log(
            f"[cryoet] stamped {len(instances)} filament monomer instance(s) "
            f"in {time.time() - t0:.1f}s"
        )


def _chain_size_summary(instances: list) -> str:
    """Summarize SAWLC chain lengths per species, e.g. '1bxn: {1: 2, 5: 1}'
    meaning two singleton chains and one 5-monomer chain for species 1bxn.
    A chain is identified by (code, polymer) -- polymer indices are only
    unique within a species, not across species (see ProteinInstance).
    """
    by_code: dict[str, Counter] = {}
    for inst in instances:
        by_code.setdefault(inst.code, Counter())[inst.polymer] += 1
    parts = []
    for code, counts in by_code.items():
        chain_length_histogram = Counter(counts.values())
        parts.append(f"{code}: {dict(sorted(chain_length_histogram.items()))}")
    return "; ".join(parts)


def _estimate_organic_potential_reference(templates: dict[str, torch.Tensor]) -> float:
    """Estimate a typical scattering-potential magnitude for ordinary,
    densely-packed organic matter (protein interior), from the actual
    high-res protein templates already built for this specimen.

    Per species: mean potential over voxels at or above half that species'
    own peak value (a standard "occupied region" measure -- robust to
    background zeros and to a single unrepresentative peak voxel, e.g. a
    heavy atom). Averaged across species if more than one is present.

    A lipid bilayer is also ordinary organic matter (mostly C/H/N/O, packed
    at a broadly similar density to protein interior), so this is a
    physically-motivated approximation for the membrane's baseline voxel
    value -- not exact (lipid elemental composition and packing differ
    somewhat from protein), but far better grounded than an arbitrary
    constant.

    Returns 1.0 if no templates are available (e.g. a membrane-only
    specimen with no proteins placed at all).
    """
    if not templates:
        return 1.0
    per_species = []
    for template in templates.values():
        peak = template.max()
        if peak <= 0:
            continue
        occupied = template[template >= 0.5 * peak]
        per_species.append(occupied.mean().item())
    return float(np.mean(per_species)) if per_species else 1.0


def _insert_clipped(
    volume: torch.Tensor, local: torch.Tensor, pos_zyx: np.ndarray
) -> None:
    """Accumulate `local` (small, odd-shaped, centered) into `volume` at
    pos_zyx, clipping to volume bounds. Same clip geometry as crowding.py's
    insert_particles_into_micrograph (see arrays.clip_insert_bounds), but
    called one particle at a time so `volume` can be a large CPU (or
    memmap-backed) tensor without ever holding a full batch of rotated
    copies at once, and max- rather than sum-merged (avoids double-counting
    where two clipped inserts happen to graze the same boundary voxel).

    pos_zyx is an ABSOLUTE voxel index into `volume` (matching polnet's own
    motif/insert_svol_tomo convention -- positions are VOI-corner-relative,
    0..extent -- NOT centered like specter's own crowding.py convention, so
    no volume-center offset is added here).
    """
    bounds = clip_insert_bounds(pos_zyx.tolist(), local.shape, volume.shape)
    if bounds is None:
        return
    dst, src = bounds
    volume[dst] = torch.maximum(volume[dst], local[src])


def _build_membrane_density_zyx(inst, v_size: float) -> np.ndarray:
    """Analytically construct one membrane's bilayer density directly in a
    local array using specter-native axis order (axis0=physical Z,
    axis1=physical Y, axis2=physical X), so rotate_volume's rotation
    matrices (which assume that same mapping -- see module docstring in
    polnet_bridge.py) apply correctly with no further conversion.

    Same double-shell + Gaussian-blur construction polnet itself uses, just
    evaluated fresh at v_size from the fitted analytic parameters instead of
    at the (coarser) resolution polnet ran its placement at.
    """
    from scipy.ndimage import gaussian_filter

    t_v = 0.5 * inst.thick / v_size
    s_v = inst.layer_s / v_size

    if inst.mb_type == "sphere":
        rad_v = inst.radius / v_size
        ao, ai = rad_v + t_v, rad_v - t_v
        r = int(np.ceil(ao + 3 * s_v)) + 2
        Z = np.arange(-r, r + 1, dtype=np.float64)[:, None, None]
        Y = np.arange(-r, r + 1, dtype=np.float64)[None, :, None]
        X = np.arange(-r, r + 1, dtype=np.float64)[None, None, :]
        G = _shell(X, Y, Z, ao + 1, ao + 1, ao + 1, ao - 1, ao - 1, ao - 1)
        G += _shell(X, Y, Z, ai + 1, ai + 1, ai + 1, ai - 1, ai - 1, ai - 1)

    elif inst.mb_type == "ellipsoid":
        # Local pre-rotation frame: physical X gets semi_axes[0], physical Y
        # gets semi_axes[1], physical Z gets semi_axes[2] (matches
        # rot_matrix's column assignment in polnet_bridge._fit_ellipsoid --
        # column i is the physical direction e_i=[x,y,z] unit vector i maps
        # to). Physical X/Y/Z map to array axes 2/1/0 respectively.
        a, b, c = inst.semi_axes / v_size
        ao, bo, co = a + t_v, b + t_v, c + t_v
        ai, bi, ci = a - t_v, b - t_v, c - t_v
        r = int(np.ceil(max(ao, bo, co) + 3 * s_v)) + 2
        Z = np.arange(-r, r + 1, dtype=np.float64)[:, None, None]
        Y = np.arange(-r, r + 1, dtype=np.float64)[None, :, None]
        X = np.arange(-r, r + 1, dtype=np.float64)[None, None, :]
        G = _shell(X, Y, Z, ao + 1, bo + 1, co + 1, ao - 1, bo - 1, co - 1)
        G += _shell(X, Y, Z, ai + 1, bi + 1, ci + 1, ai - 1, bi - 1, ci - 1)
        G = _rotate_local(G, inst.rot_matrix)

    elif inst.mb_type == "toroid":
        # Local pre-rotation frame: hole axis along physical Z (array
        # axis0), rho computed from physical X (array axis2) and Y
        # (array axis1) -- matches rot_matrix's column assignment in
        # polnet_bridge._fit_torus (columns = u, v, axis_dir -> e_x, e_y,
        # e_z respectively).
        rad_a, rad_b = inst.rad_a / v_size, inst.rad_b / v_size
        bo, bi = rad_b + t_v, rad_b - t_v
        r = int(np.ceil(rad_a + bo + 1 + 3 * s_v)) + 2
        Z = np.arange(-r, r + 1, dtype=np.float64)[:, None, None]
        Y = np.arange(-r, r + 1, dtype=np.float64)[None, :, None]
        X = np.arange(-r, r + 1, dtype=np.float64)[None, None, :]
        rho = np.sqrt(X**2 + Y**2)
        G = _torus_shell(rho, Z, rad_a, bo + 1, bo - 1)
        G += _torus_shell(rho, Z, rad_a, bi + 1, bi - 1)
        G = _rotate_local(G, inst.rot_matrix)

    else:
        raise ValueError(f"Unsupported membrane type: {inst.mb_type!r}")

    density = gaussian_filter(G.astype(np.float64), s_v)
    density -= density.min()
    m = density.max()
    if m > 0:
        density /= m
    return density.astype(np.float32)


def _shell(X, Y, Z, ao, bo, co, ai, bi, ci) -> np.ndarray:
    r_o = (X / ao) ** 2 + (Y / bo) ** 2 + (Z / co) ** 2
    r_i = (X / ai) ** 2 + (Y / bi) ** 2 + (Z / ci) ** 2
    return np.logical_and(r_i >= 1, r_o <= 1).astype(np.float64)


def _torus_shell(rho, Z, rad_a, b_outer, b_inner) -> np.ndarray:
    r_o = (rad_a - rho) ** 2 + Z**2 <= b_outer**2
    r_i = (rad_a - rho) ** 2 + Z**2 >= b_inner**2
    return np.logical_and(r_i, r_o).astype(np.float64)


def _rotate_local(arr: np.ndarray, rot_matrix: np.ndarray) -> np.ndarray:
    """Rotate a small local numpy array by a physical-xyz rotation matrix,
    via the same rotate_volume machinery used for proteins (keeps the axis
    convention identical everywhere in this module).
    """
    t = torch.as_tensor(arr, dtype=torch.float32)
    R = torch.as_tensor(rot_matrix, dtype=torch.float32)
    theta = build_affine_matrix(R)
    return rotate_volume(t, theta, padding_mode="zeros")[0].numpy()
