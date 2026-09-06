"""
Tests for PDB parsing, in particular bonded-species atom typing.
"""

import gzip
import os
import shutil
import time
import warnings
from pathlib import Path

import pytest
import torch

import specter.pdb as pdb_module
from specter.config import default_pdb_cache_dir
from specter.pdb import PDB

# Two tiers of fixture. The structures the suite cannot run without are
# tracked in tests/test_data/. These larger ones only add coverage for
# awkward depositions, so they are read from the user's own download
# cache and skipped when absent rather than committed to git -- together
# they are ~8 MB, against 4.3 MB for everything else combined. Fetch them
# with e.g. `specter simulate particles --pdb_source 7a4m` to enable.

# Myoglobin (oxy form): standard amino acids, a heme group coordinated by a
# His side chain and a bound O2 — exercises intra-residue bonds
# (_chem_comp_bond), the carboxyl/amide flag override, and inter-residue
# metal-coordination bonds (_struct_conn), all without needing an external
# CCP4 Monomer Library.
_FIXTURE = Path(__file__).parent / "test_data" / "1mbo.cif"


@pytest.fixture(scope="module")
def species():
    return PDB.get_atom_species(str(_FIXTURE), verbose=False)


def _atom_species_by_name(structure_species, filepath, resname, resseq, atom_name):
    """
    Look up the computed species string for a specific (residue, atom).

    Mirrors get_atom_species's altloc de-duplication (first-seen wins) so
    this stays index-aligned with the list it's checking against.
    """
    import gemmi

    st = gemmi.read_structure(str(filepath))
    st.setup_entities()

    names = []
    seen = set()
    for cra in st[0].all():
        if cra.atom.occ == 0.0:
            continue
        key = (cra.chain.name, cra.residue.seqid.num or 0, cra.atom.name)
        if key in seen:
            continue
        seen.add(key)
        names.append((cra.residue.name, cra.residue.seqid.num, cra.atom.name))

    for (resn, seq, atn), sp in zip(names, structure_species):
        if resn == resname and seq == resseq and atn == atom_name:
            return sp
    raise AssertionError(f"Atom {resname}{resseq}:{atom_name} not found")


class _FakeResponse:
    def __init__(self, content: bytes = b"", json_data: dict | None = None):
        self.content = content
        self._json_data = json_data or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_fetch_pdb_file_creates_missing_pdb_cache_dir(tmp_path, monkeypatch):
    # Regression test: pdb_cache_dir is a plain filesystem path (resolved
    # against the caller's own cwd, not the specter repo root), and
    # fetch_pdb_file used to crash with a raw FileNotFoundError from
    # open(file_path, "w") if that directory didn't already exist, rather
    # than creating it on demand.
    fake_cif = "data_1ABC\n#\n"
    compressed = gzip.compress(fake_cif.encode())

    def fake_get(url, *args, **kwargs):
        if "rest/v1/core/entry" in url:
            return _FakeResponse(
                json_data={"rcsb_entry_container_identifiers": {"assembly_ids": []}}
            )
        return _FakeResponse(content=compressed)

    monkeypatch.setattr(pdb_module.requests, "get", fake_get)

    missing_dir = tmp_path / "not" / "yet" / "created"
    assert not missing_dir.exists()

    filepath = PDB.fetch_pdb_file("1abc", pdb_cache_dir=str(missing_dir), verbose=False)

    assert missing_dir.exists()
    assert Path(filepath).read_text() == fake_cif


def test_canonical_pdb_source_folds_accession_case_only():
    # Accession codes are case-insensitive at RCSB, so they fold to one
    # spelling; file paths are case-sensitive on Linux and must not.
    assert pdb_module.canonical_pdb_source("1fa2") == "1FA2"
    assert pdb_module.canonical_pdb_source("1FA2") == "1FA2"
    assert pdb_module.canonical_pdb_source("1Fa2") == "1FA2"
    assert pdb_module.canonical_pdb_source("/Data/Structs/Mine.cif") == (
        "/Data/Structs/Mine.cif"
    )
    # Four characters but not alphanumeric: a path, not a code.
    assert pdb_module.canonical_pdb_source("a.cif") == "a.cif"


def test_fetch_pdb_file_downloads_one_entry_per_accession_regardless_of_case(
    tmp_path, monkeypatch
):
    # Regression test: the cache filename was built verbatim from the
    # caller's spelling, so a config naming one structure two ways (a
    # target as "1FA2", a filler as "1fa2") downloaded, cached and later
    # re-parsed the same RCSB entry twice.
    fake_cif = "data_1ABC\n#\n"
    compressed = gzip.compress(fake_cif.encode())
    downloads = []

    def fake_get(url, *args, **kwargs):
        if "rest/v1/core/entry" in url:
            return _FakeResponse(
                json_data={"rcsb_entry_container_identifiers": {"assembly_ids": []}}
            )
        downloads.append(url)
        return _FakeResponse(content=compressed)

    monkeypatch.setattr(pdb_module.requests, "get", fake_get)

    first = PDB.fetch_pdb_file("1abc", pdb_cache_dir=str(tmp_path), verbose=False)
    second = PDB.fetch_pdb_file("1ABC", pdb_cache_dir=str(tmp_path), verbose=False)

    assert first == second
    assert len(downloads) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["1ABC-assembly1.cif"]


def test_fetch_pdb_file_reuses_a_differently_cased_cache_entry(tmp_path, monkeypatch):
    # A cache populated before keys were canonicalized still holds entries
    # under their old spelling. That is the same file, so it must be reused
    # rather than re-downloaded into a second copy.
    (tmp_path / "1abc-assembly1.cif").write_text("data_1ABC\n#\n")

    def fake_get(url, *args, **kwargs):
        if "rest/v1/core/entry" in url:
            return _FakeResponse(
                json_data={"rcsb_entry_container_identifiers": {"assembly_ids": []}}
            )
        raise AssertionError("must not re-download an entry already cached")

    monkeypatch.setattr(pdb_module.requests, "get", fake_get)

    path = PDB.fetch_pdb_file("1ABC", pdb_cache_dir=str(tmp_path), verbose=False)

    assert Path(path).name == "1abc-assembly1.cif"
    assert len(list(tmp_path.iterdir())) == 1


def test_fetch_pdb_file_does_not_call_the_network_when_quiet(tmp_path, monkeypatch):
    # Regression test: get_available_assemblies returns None and only ever
    # prints, but ran on every fetch -- before the cache check, and with its
    # result discarded when verbose=False. That HTTPS round trip was the
    # entire cost of a "cache hit" for an already-downloaded structure.
    (tmp_path / "1ABC-assembly1.cif").write_text("data_1ABC\n#\n")

    def fake_get(url, *args, **kwargs):
        raise AssertionError(f"no request should be made, got {url}")

    monkeypatch.setattr(pdb_module.requests, "get", fake_get)
    path = PDB.fetch_pdb_file("1ABC", pdb_cache_dir=str(tmp_path), verbose=False)
    assert Path(path).name == "1ABC-assembly1.cif"


@pytest.mark.skipif(not _FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_parsed_cache_returns_an_identical_structure(tmp_path):
    """A cached parse must be indistinguishable from a fresh one.

    Re-parsing dominates the cost of loading an already-downloaded
    structure (16.3 s for a 220k-atom assembly against 0.09 s to read the
    arrays back), but a cache that returned anything different would
    silently render the wrong molecule.
    """
    shutil.copy(_FIXTURE, tmp_path / "1mbo.cif")
    src = str(tmp_path / "1mbo.cif")

    cold = PDB(
        src, pdb_cache_dir=str(tmp_path), verbose=False, compute_atom_species=True
    )
    assert cold._parse_was_cached is False
    warm = PDB(
        src, pdb_cache_dir=str(tmp_path), verbose=False, compute_atom_species=True
    )
    assert warm._parse_was_cached is True

    assert torch.equal(cold.atomic_numbers, warm.atomic_numbers)
    assert torch.equal(cold.coordinates, warm.coordinates)
    assert cold.atom_species == warm.atom_species
    assert torch.equal(cold.b_factors, warm.b_factors)
    assert cold.max_diameter == warm.max_diameter


@pytest.mark.skipif(not _FIXTURE.exists(), reason="bundled PDB fixture missing")
@pytest.mark.parametrize("compute_atom_species", [False, True])
def test_b_factors_align_with_the_atoms_they_describe(compute_atom_species):
    """Both parse paths must return one B-factor per atom, same order.

    The Biopython walk and the gemmi typed model build the atom list
    independently, and `PotentialBuilder(b_factors=...)` indexes it
    positionally -- a length that matched but an order that did not would
    damp the wrong atoms with no error anywhere.
    """
    pdb = PDB(str(_FIXTURE), verbose=False, compute_atom_species=compute_atom_species)

    assert pdb.b_factors.shape == (pdb.atomic_numbers.shape[0],)
    assert torch.isfinite(pdb.b_factors).all()
    assert (pdb.b_factors >= 0).all()
    # 1mbo is a real deposition, so its column varies rather than sitting at
    # one refined-flat value -- which is the whole reason a per-atom B is
    # worth carrying at all.
    assert pdb.b_factors.std() > 0


@pytest.mark.skipif(not _FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_both_parse_paths_agree_on_b_factors():
    """Typing a structure must not change the B-factors it reports.

    Without a monomer library the two paths describe the same atom list, so
    any disagreement is a misalignment between them.
    """
    plain = PDB(str(_FIXTURE), verbose=False)
    typed = PDB(str(_FIXTURE), verbose=False, compute_atom_species=True)

    assert typed.b_factors.shape == plain.b_factors.shape
    torch.testing.assert_close(typed.b_factors, plain.b_factors)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="bundled PDB fixture missing")
@pytest.mark.parametrize(
    "differing",
    [
        {"compute_atom_species": False},
        {"compute_atom_species": True, "readd_hydrogens": False},
        {"compute_atom_species": True, "readd_hydrogens": True},
    ],
)
def test_parsed_cache_misses_when_a_parse_flag_differs(tmp_path, differing):
    """Every flag that changes the parse is part of the key.

    A stale hit here would be the worst kind of failure: a structure that
    loads fast, looks plausible, and is wrong.
    """
    shutil.copy(_FIXTURE, tmp_path / "1mbo.cif")
    src = str(tmp_path / "1mbo.cif")
    PDB(src, pdb_cache_dir=str(tmp_path), verbose=False, compute_atom_species=True)

    other = PDB(src, pdb_cache_dir=str(tmp_path), verbose=False, **differing)
    assert other._parse_was_cached is False


@pytest.mark.skipif(not _FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_parsed_cache_misses_when_the_source_file_changes(tmp_path):
    """Editing a structure in place must not keep serving the old parse."""
    shutil.copy(_FIXTURE, tmp_path / "1mbo.cif")
    src = str(tmp_path / "1mbo.cif")
    PDB(src, pdb_cache_dir=str(tmp_path), verbose=False)
    assert PDB(src, pdb_cache_dir=str(tmp_path), verbose=False)._parse_was_cached

    os.utime(src, (time.time() + 10, time.time() + 10))
    assert (
        PDB(src, pdb_cache_dir=str(tmp_path), verbose=False)._parse_was_cached is False
    )


@pytest.mark.skipif(not _FIXTURE.exists(), reason="bundled PDB fixture missing")
def test_parsed_cache_still_applies_origin_and_survives_corruption(tmp_path):
    """`origin` is applied after the cache, so it must still take effect on a
    hit -- it is deliberately not part of the key, since re-centering costs
    nothing. And a truncated entry (a killed writer, a full disk) must fall
    back to parsing rather than raising."""
    shutil.copy(_FIXTURE, tmp_path / "1mbo.cif")
    src = str(tmp_path / "1mbo.cif")
    base = PDB(src, pdb_cache_dir=str(tmp_path), verbose=False)
    shifted = PDB(
        src, pdb_cache_dir=str(tmp_path), verbose=False, origin=(5.0, 6.0, 7.0)
    )
    assert shifted._parse_was_cached
    assert not torch.allclose(base.coordinates, shifted.coordinates)

    entries = list((tmp_path / "parsed").glob("*.pt"))
    assert entries
    entries[0].write_bytes(b"not a torch file")
    recovered = PDB(src, pdb_cache_dir=str(tmp_path), verbose=False)
    assert recovered._parse_was_cached is False
    assert torch.equal(recovered.atomic_numbers, base.atomic_numbers)


def test_get_atom_species_requires_mmcif(tmp_path):
    """A legacy .pdb source has no chem-comp bond dictionary; species is None."""
    fake_pdb = tmp_path / "fake.pdb"
    fake_pdb.write_text(
        "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    )
    with pytest.warns(UserWarning):
        result = PDB.get_atom_species(str(fake_pdb), verbose=False)
    assert result == [None]


def test_carboxyl_oxygen_flag(species):
    """Asp OD1/OD2 side-chain oxygens get the ', carboxyl' species suffix."""
    sp = _atom_species_by_name(species, _FIXTURE, "ASP", 20, "OD1")
    assert sp is not None
    assert sp.startswith("O(")
    assert sp.endswith(", carboxyl)")


def test_heme_iron_coordination(species):
    """
    Heme Fe shows up bonded to porphyrin N atoms (+ His/O2 axial ligands),
    confirming _struct_conn metal-coordination bonds are picked up.

    This is the no-monomer-library path, where those links are kept: without a
    library the porphyrin Fe-N bonds are not fully defined by the file's own
    _chem_comp_bond, so dropping the links would leave Fe(NNN) here and Fe(N)
    for 1A6M. See test_metal_links_dropped_with_library for the other path.
    """
    sp = _atom_species_by_name(species, _FIXTURE, "HEM", 155, "FE")
    assert sp is not None
    assert sp.startswith("Fe(")
    assert "N" in sp

    # matches the bundled Shtyrov species table's Fe environment format
    from specter.atom import load_shtyrov_species_parameters
    from importlib import resources

    path = resources.files("specter.atom_data").joinpath("params_cat.json")
    with resources.as_file(path) as fpath:
        table = load_shtyrov_species_parameters(str(fpath))
    assert "Fe(NNNN)" in table  # the 4-coordinate deoxy-heme entry exists


def test_unresolved_atoms_are_none(species):
    """Atoms whose component/bonds can't be resolved fall back to None, not an error."""
    assert any(s is None for s in species)


def test_species_aligns_with_atomic_numbers():
    """atom_species (opt-in) is the same length as atomic_numbers/coordinates."""
    pdb = PDB(str(_FIXTURE), verbose=False, compute_atom_species=True)
    assert pdb.atom_species is not None
    assert len(pdb.atom_species) == pdb.atomic_numbers.shape[0]
    assert len(pdb.atom_species) == pdb.coordinates.shape[0]


# Beta-amylase: 208 of its deposited atoms carry occupancy 0.0, which is also
# the marker get_atom_species stamps on the dummy hydrogens it adds while
# building the bond graph. Typing once dropped the deposited ones too, so
# atom_species came back 208 entries short of atomic_numbers and the Shtyrov
# potential path died on the length mismatch.
_ZERO_OCC_FIXTURE = Path(default_pdb_cache_dir()) / "1fa2-assembly1.cif"


@pytest.mark.skipif(
    not _ZERO_OCC_FIXTURE.exists(), reason="1fa2 not in the local PDB cache"
)
def test_zero_occupancy_atoms_are_still_typed():
    """Deposited zero-occupancy atoms are not mistaken for added dummy atoms."""
    import gemmi

    st = gemmi.read_structure(str(_ZERO_OCC_FIXTURE))
    st.setup_entities()
    assert sum(1 for cra in st[0].all() if cra.atom.occ == 0.0) > 0, (
        "fixture no longer has zero-occupancy atoms; pick another structure"
    )

    pdb = PDB(str(_ZERO_OCC_FIXTURE), verbose=False, compute_atom_species=True)
    assert pdb.atom_species is not None
    assert len(pdb.atom_species) == pdb.atomic_numbers.shape[0]


# Each of these once returned an atom_species list of the wrong length, which
# surfaced far downstream as an opaque IndexError inside the Shtyrov potential
# builder (`shape of the mask [N] does not match the indexed tensor [M]`) --
# and shtyrov is the default parameterization, so these structures could not be
# rendered at all. One entry per distinct root cause:
#   3DY4  residues numbered 10, 10A, 10B... -- the atom key ignored the
#         insertion code, so distinct atoms collided and were dropped as
#         though they were alternate conformers.
#   7a4m  carries explicit hydrogens -- HydrogenChange.ReAdd strips every
#         existing H before re-adding it from monomer-library geometry, so
#         without a library the strip happened and the re-add did not.
_ALIGNMENT_FIXTURES = ["3DY4-assembly1.cif", "7a4m.cif"]


@pytest.mark.parametrize("name", _ALIGNMENT_FIXTURES)
def test_atom_species_aligns_for_awkward_structures(name):
    """atom_species stays index-aligned with atomic_numbers, element for element."""
    from specter.atom import atom_symbol

    path = Path(default_pdb_cache_dir()) / name
    if not path.exists():
        pytest.skip(f"{name} not in the local PDB cache")

    pdb = PDB(str(path), verbose=False, compute_atom_species=True)
    assert pdb.atom_species is not None
    assert len(pdb.atom_species) == pdb.atomic_numbers.shape[0]

    # Equal lengths alone would still pass if the two lists were shifted
    # relative to each other, so check the elements agree entry by entry.
    mismatched = [
        i
        for i, (z, s) in enumerate(zip(pdb.atomic_numbers.tolist(), pdb.atom_species))
        if s is not None and str(atom_symbol(int(z))).upper() != s.split("(")[0].upper()
    ]
    assert not mismatched, f"{len(mismatched)} atoms typed as the wrong element"


# A monomer library is what supplies the hydrogens a deposited structure omits,
# and without it every H-containing Shtyrov species ("C(HHHC)", "O(HH)") misses
# the table and falls back to per-element Peng. Typing then covers ~56% of a
# protein's atoms instead of ~99%. These tests need a real library, so they are
# skipped wherever one is not configured.
def _monomer_library() -> str | None:
    from os import environ

    env = environ.get("CLIBD_MON")
    if env and Path(env).is_dir():
        return env
    bundled = Path.home() / "sffit" / "monomers"
    return str(bundled) if bundled.is_dir() else None


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
def test_monomer_library_adds_hydrogens_and_stays_aligned():
    """With a library, all three arrays come from the H-completed model."""
    monlib = _monomer_library()

    plain = PDB(str(_FIXTURE), verbose=False, compute_atom_species=True)
    assert (plain.atomic_numbers == 1).sum().item() == 0, (
        "fixture already carries hydrogens; pick one without them"
    )

    completed = PDB(
        str(_FIXTURE),
        verbose=False,
        compute_atom_species=True,
        monomer_library_path=monlib,
    )
    n_h = (completed.atomic_numbers == 1).sum().item()
    assert n_h > 0, "library did not add any hydrogens"

    # The three arrays are taken from one iteration, so they cannot disagree.
    assert len(completed.atom_species) == completed.atomic_numbers.shape[0]
    assert completed.coordinates.shape[0] == completed.atomic_numbers.shape[0]
    assert completed.atomic_numbers.shape[0] > plain.atomic_numbers.shape[0]


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
def test_ambiguous_hydrogens_are_not_rendered():
    """Zero-occupancy H (rotatable -OH, His tautomers) must not reach the model.

    PotentialBuilder applies no occupancy weighting, so keeping them would
    render both tautomer hydrogens of every histidine. sffit selects ";q>0"
    for the same reason.
    """
    import gemmi

    monlib = _monomer_library()
    st = gemmi.read_structure(str(_FIXTURE))
    st.setup_entities()
    lib = gemmi.read_monomer_lib(monlib, st[0].get_all_residue_names())
    gemmi.prepare_topology(st, lib, h_change=gemmi.HydrogenChange.ReAdd)
    n_zero = sum(1 for cra in st[0].all() if cra.atom.occ == 0.0)
    assert n_zero > 0, "fixture no longer exercises the ambiguous-hydrogen path"

    completed = PDB(
        str(_FIXTURE),
        verbose=False,
        compute_atom_species=True,
        monomer_library_path=monlib,
    )
    n_kept = completed.atomic_numbers.shape[0]
    n_all = sum(1 for _ in st[0].all())
    assert n_kept <= n_all - n_zero, (
        f"{n_all - n_zero - n_kept} ambiguous hydrogens were not excluded"
    )


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
def test_metal_links_dropped_with_library():
    """With a library, the heme iron types as Fe(NNNN) -- an entry in the table.

    sffit deletes ConnectionType.MetalC before typing, so the iron keeps only
    the four porphyrin nitrogens its HEM component defines rather than also
    picking up the axial His/O2 ligands. Keeping them gives Fe(NNNNNO...),
    which no fitted table contains, so the iron would fall back to Peng.
    """
    from specter.pdb import PDB as _PDB

    _, _, species, _, used = _PDB._build_typed_model(
        str(_FIXTURE), _monomer_library(), False
    )
    assert used, "library did not load"
    iron = [s for s in species if s is not None and s.startswith("Fe(")]
    assert iron == ["Fe(NNNN)"], iron


# 7a4m carries its own hydrogens, so it distinguishes the two modes: ReAdd
# replaces them with the library's ideal geometry, NoChange keeps them where
# they were deposited.
_H_FIXTURE = Path(default_pdb_cache_dir()) / "7a4m.cif"


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
@pytest.mark.skipif(not _H_FIXTURE.exists(), reason="7a4m not in the local PDB cache")
def test_readd_hydrogens_false_keeps_deposited_coordinates():
    """readd_hydrogens=False leaves a file's own hydrogens where they were."""
    import gemmi

    st = gemmi.read_structure(str(_H_FIXTURE))
    st.setup_entities()
    deposited = {
        (round(c.atom.pos.x, 3), round(c.atom.pos.y, 3), round(c.atom.pos.z, 3))
        for c in st[0].all()
        if c.atom.element.name == "H"
    }
    assert deposited, "fixture no longer carries hydrogens"

    def h_coords(readd):
        znum, pos, _, _, used = PDB._build_typed_model(
            str(_H_FIXTURE), _monomer_library(), False, readd_hydrogens=readd
        )
        assert used
        return {
            (round(x, 3), round(y, 3), round(z, 3))
            for (x, y, z), n in zip(pos, znum)
            if n == 1
        }

    kept = h_coords(False)
    ideal = h_coords(True)

    # Keeping them means most land exactly on the deposited positions...
    assert len(kept & deposited) / len(kept) > 0.9
    # ...while re-adding them from ideal geometry moves nearly all of them.
    assert len(ideal & deposited) / len(ideal) < 0.5


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
def test_readd_hydrogens_false_types_without_adding_density():
    """On a hydrogen-free structure, readd_hydrogens=False types but adds no atoms.

    The species descriptor comes from the bond graph, not from coordinates, so
    hydrogens added purely as zero-occupancy dummies still resolve their
    neighbours' species while contributing no density.
    """
    import json
    from importlib import resources

    path = resources.files("specter.atom_data").joinpath("params_cat.json")
    with resources.as_file(path) as fpath:
        table = set(json.loads(Path(fpath).read_text()))

    def coverage(mon, readd):
        znum, _, species, _, _ = PDB._build_typed_model(
            str(_FIXTURE), mon, False, readd_hydrogens=readd
        )
        hit = sum(1 for s in species if s in table)
        return len(znum), sum(1 for z in znum if z == 1), hit / len(znum)

    n_plain, h_plain, cov_plain = coverage(None, True)
    n_typed, h_typed, cov_typed = coverage(_monomer_library(), False)

    assert (n_typed, h_typed) == (n_plain, h_plain), "density changed"
    assert cov_typed > cov_plain + 0.3, (cov_plain, cov_typed)


@pytest.mark.skipif(_monomer_library() is None, reason="no monomer library available")
@pytest.mark.skipif(not _H_FIXTURE.exists(), reason="7a4m not in the local PDB cache")
def test_readd_hydrogens_auto_follows_the_file():
    """ "auto" keeps deposited hydrogens, and adds them only when there are none."""
    mon = _monomer_library()

    def counts(cif, mode):
        znum, _, _, _, _ = PDB._build_typed_model(
            str(cif), mon, False, readd_hydrogens=mode
        )
        return len(znum), sum(1 for z in znum if z == 1)

    # 1mbo carries no hydrogens: "auto" should behave like True and add them.
    assert counts(_FIXTURE, "auto") == counts(_FIXTURE, True)
    assert counts(_FIXTURE, "auto")[1] > 0

    # 7a4m carries its own: "auto" should behave like False and keep them.
    assert counts(_H_FIXTURE, "auto") == counts(_H_FIXTURE, False)
    assert counts(_H_FIXTURE, "auto") != counts(_H_FIXTURE, True)


def test_monomer_library_path_expands_and_reports_clearly(tmp_path, monkeypatch):
    """A '~' or '$VAR' path is expanded; a wrong one names itself and its source.

    gemmi does no shell expansion, so "~/monomers" -- the natural thing to put
    in a TOML -- used to reach read_monomer_lib verbatim and fail with a bare
    FileNotFoundError naming a path the user never wrote.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MONLIB_ROOT", str(tmp_path))

    # neither of these exists, so the error tells us which path was resolved
    for given, source in [
        ("~/monomers", "monomer_library_path"),
        ("$MONLIB_ROOT/monomers", "monomer_library_path"),
    ]:
        with pytest.raises(FileNotFoundError) as exc:
            PDB._build_typed_model(str(_FIXTURE), given, False)
        assert str(tmp_path / "monomers") in str(exc.value), str(exc.value)
        assert source in str(exc.value)

    # a stale $CLIBD_MON is named as such rather than blamed on the argument
    monkeypatch.setenv("CLIBD_MON", str(tmp_path / "gone"))
    with pytest.raises(FileNotFoundError, match=r"\$CLIBD_MON"):
        PDB._build_typed_model(str(_FIXTURE), None, False)


def test_missing_monomer_library_reported_once_per_process(monkeypatch):
    """
    The missing-library warning is a property of the environment, so a run
    typing many structures reports it once, not once per structure.

    A tomogram loading 27 species emitted 27 copies of a ~90-word warning.
    Python's own once-per-location dedup does not cover the two places this
    actually showed up -- IPython clears `__warningregistry__` between cells,
    and each spawned PDB worker starts with fresh module state -- so the
    suppression has to be specter's own, and cannot be tested by relying on
    the default warning filter.
    """
    monkeypatch.delenv("CLIBD_MON", raising=False)
    monkeypatch.setattr(pdb_module, "_monomer_library_warned", False)

    with pytest.warns(RuntimeWarning, match="No monomer library configured"):
        PDB._build_typed_model(str(_FIXTURE), None, verbose=False)

    # "always" defeats the interpreter's own dedup, so anything caught here is
    # specter emitting a second copy rather than the filter letting one through.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            PDB._build_typed_model(str(_FIXTURE), None, verbose=False)
    assert [w for w in caught if "No monomer library" in str(w.message)] == []


def test_monomer_library_warning_can_be_suppressed_for_workers(monkeypatch):
    """
    A spawned worker stays silent so its parent can report on its behalf.

    `_parallel_render._fetch_one_pdb` calls this: without it each of the
    (up to 8) worker processes rediscovers the same missing library and
    reports it independently, which is where the duplicate warnings a
    tomogram run printed actually came from.
    """
    monkeypatch.delenv("CLIBD_MON", raising=False)
    monkeypatch.setattr(pdb_module, "_monomer_library_warned", False)
    pdb_module.suppress_monomer_library_warning()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PDB._build_typed_model(str(_FIXTURE), None, verbose=False)
    assert [w for w in caught if "No monomer library" in str(w.message)] == []


def test_missing_path_still_reports_both_accepted_forms(tmp_path: Path) -> None:
    """A typo'd path must not be silently treated as an accession code."""
    with pytest.raises(ValueError, match="4-character PDB ID or a valid file path"):
        PDB(str(tmp_path / "does_not_exist.cif"), verbose=False)
