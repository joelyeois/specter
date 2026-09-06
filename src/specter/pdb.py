"""
`PDB`: fetches, caches and parses a structure into atomic numbers,
coordinates and, for Shtyrov factors, bonded-species types.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import sys
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import gemmi
import numpy as np
import requests
import torch
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning  # correct import
from .progress import track
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure

from .atom import atom_number
from .config import default_pdb_cache_dir

# A user-level cache (`~/.cache/specter/pdb` by default), NOT a project
# directory: a structure fetched by accession code is identical for every
# project, so caching per user downloads it once rather than once per working
# directory. Set $SPECTER_PDB_CACHE (or $XDG_CACHE_HOME) to relocate it, e.g.
# onto scratch. Only downloads land here -- a structure passed by path is read
# where it lies, which is what keeps `specter cache clean` safe. See
# `default_pdb_cache_dir` in config/_paths.py.
DEFAULT_PDB_CACHE_DIR = default_pdb_cache_dir()

# Suppress only PDBConstructionWarnings
warnings.simplefilter("ignore", PDBConstructionWarning)


# Whether this process has already reported that no monomer library is
# configured. The condition is a property of the environment, not of any one
# structure, so a tomogram loading 27 species has nothing to say 27 times.
# Python's own once-per-location dedup does not cover it: IPython clears
# `__warningregistry__` between notebook cells, and every worker spawned by
# `specimen/_parallel_render.py` starts with fresh module state, so the same
# environment-level fact was reported once per worker per cell.
_monomer_library_warned = False


def _cache_file_ignoring_case(pdb_cache_dir: str, filename: str) -> str | None:
    """
    Find `filename` in `pdb_cache_dir` ignoring case, or None.

    Parameters
    ----------
    pdb_cache_dir : str
        Directory to search. A missing directory yields None.
    filename : str
        The canonically-spelled filename being looked for.

    Returns
    -------
    str or None
        Full path to the case-insensitive match, or None if there is none.
    """
    try:
        entries = os.listdir(pdb_cache_dir)
    except OSError:
        return None
    wanted = filename.lower()
    for entry in entries:
        if entry.lower() == wanted:
            return os.path.join(pdb_cache_dir, entry)
    return None


def canonical_pdb_source(pdb_source: str) -> str:
    """
    Normalize a `pdb_source` so the same structure has one cache key.

    Parameters
    ----------
    pdb_source : str
        A 4-character PDB accession code, or a path to a structure file.

    Returns
    -------
    str
        The accession code upper-cased, or `pdb_source` unchanged when it
        is a file path.

    Notes
    -----
    RCSB accession codes are case-insensitive -- ``1fa2`` and ``1FA2`` name
    one entry and both download -- so a config spelling the same structure
    two ways (a target as ``1FA2``, a filler as ``1fa2``) otherwise
    downloads it twice, caches it under two filenames, and parses/types it
    twice. Upper case is the canonical form because it is what RCSB and
    the literature print. File paths are returned untouched: they are
    case-sensitive on Linux, so folding their case would break them.

    The accession-code test matches `PDB.__init__`'s own, deliberately --
    see the comment there for why a 4-character alphanumeric string is
    read as an ID rather than as a filename.
    """
    if len(pdb_source) == 4 and pdb_source.isalnum():
        return pdb_source.upper()
    return pdb_source


#: Bumped whenever anything that shapes a parsed structure changes -- the
#: gemmi typing rules, the Biopython atom walk, the hydrogen handling. It is
#: part of every parsed-cache key, so an entry written by older code misses
#: instead of silently returning a structure the current code would not
#: produce. Raise it when editing `_build_typed_model`,
#: `get_atoms_and_coordinates` or `get_atom_species` -- and note the
#: entry carries every field `PDB` reads off a structure, so adding one (as
#: `b_factors` did at version 2) is itself a reason to raise it.
_PARSED_CACHE_VERSION = 2


def _parsed_cache_path(
    pdb_cache_dir: str,
    filepath: str,
    compute_atom_species: bool,
    readd_hydrogens: bool | str,
    monomer_library_path: str | None,
) -> str | None:
    """
    Where the parsed form of `filepath` would be cached, or None.

    Parsing a structure is a pure function of the file's bytes and the three
    flags that steer it, so its result can be reused across runs -- which is
    worth a great deal: re-parsing a 220k-atom assembly costs 16.7 s against
    0.04 s to load the arrays back.

    The key covers the source file's identity (path, size, mtime) as well as
    the flags, so editing or replacing a structure in place misses rather
    than returning the previous parse.

    Parameters
    ----------
    pdb_cache_dir : str
        Root of the structure cache; entries live in its `parsed/` subfolder.
    filepath : str
        The structure file that would be parsed.
    compute_atom_species : bool
        Whether bonded-species typing runs (changes the result).
    readd_hydrogens : bool or str
        Hydrogen handling (changes the atom list).
    monomer_library_path : str, optional
        Already-resolved library directory, or None.

    Returns
    -------
    str or None
        Full path to the cache entry, or None when the source file cannot be
        stat-ed (in which case parsing simply proceeds uncached).
    """
    try:
        st = os.stat(filepath)
    except OSError:
        return None
    key = "\0".join(
        [
            str(_PARSED_CACHE_VERSION),
            os.path.realpath(filepath),
            str(st.st_size),
            str(st.st_mtime_ns),
            str(bool(compute_atom_species)),
            str(readd_hydrogens),
            monomer_library_path or "",
        ]
    )
    digest = hashlib.sha256(key.encode()).hexdigest()
    return os.path.join(pdb_cache_dir, "parsed", f"{digest}.pt")


def _load_parsed_structure(
    path: str | None,
) -> tuple[torch.Tensor, torch.Tensor, list[str | None] | None, torch.Tensor] | None:
    """
    Read a cached parse, or None if there isn't a usable one.

    Any failure -- missing file, truncated write, a torch version that
    can't read it -- returns None so the caller re-parses. A cache is an
    optimisation; it must never be the reason a structure fails to load.
    """
    if path is None or not os.path.exists(path):
        return None
    try:
        blob = torch.load(path, weights_only=False)
        return (
            blob["atomic_numbers"],
            blob["coordinates"],
            blob["atom_species"],
            blob["b_factors"],
        )
    except Exception:
        return None


def _store_parsed_structure(
    path: str | None,
    atomic_numbers: torch.Tensor,
    coordinates: torch.Tensor,
    atom_species: list[str | None] | None,
    b_factors: torch.Tensor,
) -> None:
    """
    Write a parse to the cache, best-effort.

    Written to a temporary file and moved into place, because
    `specimen._parallel_render` parses across several PROCESSES and two of
    them can reach the same entry at once -- `os.replace` is atomic, so a
    reader never sees a half-written file. A failure here (read-only cache,
    full disk) is swallowed: the caller already has the parse it needs.
    """
    if path is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(
            {
                "atomic_numbers": atomic_numbers,
                "coordinates": coordinates,
                "atom_species": atom_species,
                "b_factors": b_factors,
            },
            tmp,
        )
        os.replace(tmp, path)
    except Exception:
        pass


def _resolve_monomer_library_path(
    monomer_library_path: str | None = None,
) -> str | None:
    """
    Resolve the Monomer Library directory to type Shtyrov species against.

    Parameters
    ----------
    monomer_library_path : str, optional
        An explicit path; falls back to `$CLIBD_MON` when not given.

    Returns
    -------
    str or None
        An existing directory, or None if neither source names one.

    Raises
    ------
    FileNotFoundError
        If a path is configured but is not a directory.
    """
    monlib_path = monomer_library_path or os.environ.get("CLIBD_MON")
    if not monlib_path:
        return None
    # gemmi does no shell expansion, so a perfectly natural "~/monomers" in a
    # TOML would otherwise reach it verbatim and fail deep inside
    # read_monomer_lib with a bare FileNotFoundError naming a path the user
    # never wrote.
    monlib_path = os.path.expanduser(os.path.expandvars(monlib_path))
    if not os.path.isdir(monlib_path):
        raise FileNotFoundError(
            f"Monomer library not found at {monlib_path!r} (from "
            + ("monomer_library_path" if monomer_library_path else "$CLIBD_MON")
            + "). Point it at a clone of "
            "https://github.com/MonomerLibrary/monomers, or unset it "
            "to run without hydrogen typing."
        )
    return monlib_path


def _warn_missing_monomer_library() -> None:
    """
    Report that Shtyrov typing is running without a monomer library, at most
    once per process. See `_monomer_library_warned`.
    """
    global _monomer_library_warned
    if _monomer_library_warned:
        return
    _monomer_library_warned = True
    warnings.warn(
        "No monomer library configured (set $CLIBD_MON or pass "
        "monomer_library_path). gemmi ships no component definitions "
        "of its own, so the topology is built from the file's own "
        "bonds alone and no hydrogens can be added: H-containing "
        "Shtyrov species (e.g. 'O(HH)', 'C(HHHC)') will not resolve, "
        "and those atoms fall back to per-element Peng -- around 44% "
        "of a hydrogen-free protein. Install the Monomer Library "
        "(https://github.com/MonomerLibrary/monomers) to type them.",
        RuntimeWarning,
        stacklevel=3,
    )


def suppress_monomer_library_warning() -> None:
    """
    Mark the missing-monomer-library warning as already reported, so this
    process stays silent about it.

    For worker processes whose parent has already reported it on their
    behalf; see `_monomer_library_warned`.
    """
    global _monomer_library_warned
    _monomer_library_warned = True


# (chain, residue number, insertion code, atom name)
_AtomKey = tuple[str, int, str, str]


def _atom_key(
    chain: "gemmi.Chain", residue: "gemmi.Residue", atom: "gemmi.Atom"
) -> _AtomKey:
    """
    Identify one atom uniquely within a gemmi model.

    The insertion code is part of the identity, not decoration: an entry like
    3DY4 numbers whole stretches of residues `10`, `10A`, `10B`, ... , so a key
    without it collapses thousands of genuinely distinct atoms onto each other.
    `get_atom_species` treats a repeated key as an alternate conformer to be
    dropped, so those atoms went missing and the returned species list no
    longer lined up with `atomic_numbers`.
    """
    return (chain.name, residue.seqid.num or 0, residue.seqid.icode, atom.name)


class PDB:
    def __init__(
        self,
        pdb_source: str,
        assembly: bool = True,
        pdb_cache_dir: str = DEFAULT_PDB_CACHE_DIR,
        origin: tuple[float, float, float] | None = None,
        verbose: bool = True,
        compute_atom_species: bool = False,
        monomer_library_path: str | None = None,
        readd_hydrogens: bool | str = "auto",
    ) -> None:
        """
        Create a PDB object from either a PDB ID or a local file path.

        Parameters
        ----------
        pdb_source : str
            Either a 4-character PDB ID (e.g., '1abc') or a local file path
            to a PDB/mmCIF structure file.
        assembly : bool, optional
            Whether to fetch the biological assembly when using a PDB ID.
            Default is True.
        pdb_cache_dir : str, optional
            Folder to store downloaded PDB/mmCIF files. Default is
            `$SPECTER_PDB_CACHE`, else `$XDG_CACHE_HOME/specter/pdb`, else
            `~/.cache/specter/pdb` -- see `config.default_pdb_cache_dir`.
        origin : tuple[float, float, float] or None, optional
            Custom origin to subtract from coordinates. If None, coordinates
            are centered on their geometric center. If a tuple, that point is
            subtracted directly without auto-centering.
        verbose : bool, optional
            Print fetch/assembly status and show the atom-extraction progress
            bar. Default True; set False to silence per-fetch chatter (e.g.
            when constructing many PDB objects in a loop with your own
            higher-level progress reporting).
        compute_atom_species : bool, optional
            Also compute `atom_species` (bonded-neighbor species descriptors,
            e.g. for the Shtyrov potential parameterization). Requires an
            mmCIF source and re-parses it via gemmi, so it is opt-in.
            Default is False.
        monomer_library_path : str, optional
            Path to a Monomer Library (https://github.com/MonomerLibrary/
            monomers), falling back to `$CLIBD_MON`. Only consulted when
            `compute_atom_species=True`. When one resolves, gemmi completes
            the model with the hydrogens the library defines, and
            `atomic_numbers`/`coordinates`/`atom_species` are then all taken
            from that completed model -- so a typical hydrogen-free
            deposition roughly doubles in atom count, and H-containing
            species (`"C(HHHC)"`, `"O(HH)"`) resolve instead of falling back
            to per-element Peng. Without a library the file's own atoms are
            used unchanged, which is the historical behaviour.
        readd_hydrogens : {"auto", True, False}, optional
            Whether to replace existing hydrogens with the monomer library's
            ideal geometry. Only meaningful when a library resolves.

            ``"auto"`` (default) keeps the hydrogens a structure already
            carries and adds them only when it carries none. Deposited
            hydrogen positions are information the file provides, so there is
            no reason to move them; a structure with none still gets the
            typing and density that make the Shtyrov species resolvable.

            ``True`` always re-adds from ideal geometry, matching `sffit`'s
            default and so the configuration the scattering factors were
            fitted in. ``False`` never re-adds: existing hydrogens stay put,
            and hydrogens the file lacks become zero-occupancy atoms that
            inform their neighbours' species without being rendered, which
            improves typing without changing the atom set at all.

            Typing improves under every setting -- a species descriptor is
            built from the bond graph, not from positions, so the fitted
            factors apply whichever coordinates are used. What differs is only
            whether hydrogen density is added, and from ideal or deposited
            geometry.

            A partially hydrogenated structure is treated as carrying them, so
            ``"auto"`` keeps what is there rather than replacing the lot;
            pass ``True`` explicitly to complete it from ideal geometry.

        Attributes
        ----------
        pdb_id : str
            The PDB ID if pdb_source is a PDB ID.
        filepath : str
            Path to the PDB/mmCIF file.
        atomic_numbers : torch.Tensor
            Atomic numbers of all atoms in the structure, shape (N,).
        coordinates : torch.Tensor
            Coordinates shifted by origin, shape (N, 3).
        atom_species : list of str or None, optional
            Bonded-neighbor species descriptor per atom (e.g. `"O(HH)"`,
            `"C(HHHC)"`), only set when `compute_atom_species=True`.
        b_factors : torch.Tensor
            Deposited isotropic B-factor per atom in Å², shape (N,), aligned
            with `atomic_numbers`/`coordinates`. Always read; whether anything
            renders with it is the caller's choice (see
            `PotentialBuilder(b_factors=...)`). Hydrogens a monomer library
            adds carry whatever B gemmi assigns them, which is 0 for an atom
            built from ideal geometry.
        """

        # Determine whether pdb_source is a PDB ID or file path. The ID check
        # comes first deliberately, and the ambiguity it implies is narrower
        # than it looks: a readable structure file must end in .cif or .pdb
        # (see _load_structure), so any real one is longer than 4 characters
        # and reaches the isfile branch regardless. Only a bare, extensionless
        # 4-character filename collides -- which specter could not parse
        # anyway -- whereas a user typing a 4-character argument essentially
        # always means the accession code.
        if (
            isinstance(pdb_source, str)
            and len(pdb_source) == 4
            and pdb_source.isalnum()
        ):
            # Treat as PDB ID
            self.pdb_id = pdb_source
            self.filepath = PDB.fetch_pdb_file(
                pdb_source,
                pdb_cache_dir=pdb_cache_dir,
                assembly=assembly,
                verbose=verbose,
            )
            self.assembly = assembly
            self.pdb_cache_dir = pdb_cache_dir
        elif os.path.isfile(pdb_source):
            self.filepath = pdb_source
        else:
            raise ValueError(
                f"Invalid pdb_source: '{pdb_source}'. Must be a 4-character PDB ID or a valid file path."
            )

        # Biopython parses lazily via the `structure` property: when a monomer
        # library completes the model below, its atom list supersedes
        # Biopython's entirely and parsing the file twice would be pure waste
        # (20 s of the 30 s a 532k-atom assembly takes).
        self._structure: "Structure | None" = None

        # bonded-neighbor species descriptors (e.g. for Shtyrov potentials)
        self.atom_species: list[str | None] | None = None

        # deposited isotropic B-factors, Å², set by every path below
        self.b_factors: torch.Tensor

        # Parsing is by far the dominant cost of loading a structure that is
        # already downloaded (16.7 s for a 220k-atom assembly, roughly 60% of
        # it Biopython and 40% gemmi typing), and it is a pure function of the
        # file plus the flags below -- so reuse it across runs. `origin` and
        # `max_diameter` are re-derived from the cached coordinates further
        # down, and so are deliberately NOT part of the key.
        resolved_library = _resolve_monomer_library_path(monomer_library_path)
        parsed_path = _parsed_cache_path(
            pdb_cache_dir,
            self.filepath,
            compute_atom_species,
            readd_hydrogens,
            resolved_library,
        )
        cached = _load_parsed_structure(parsed_path)
        if cached is not None:
            (
                self.atomic_numbers,
                self.coordinates,
                self.atom_species,
                self.b_factors,
            ) = cached
            if compute_atom_species and resolved_library is None:
                # Raised inside _build_typed_model on the parsing path, so a
                # cache hit would otherwise silently stop reporting an
                # environment problem that still applies.
                _warn_missing_monomer_library()
            self._parse_was_cached = True
        else:
            self._parse_was_cached = False
            self._parse_structure(
                compute_atom_species=compute_atom_species,
                readd_hydrogens=readd_hydrogens,
                monomer_library_path=monomer_library_path,
                verbose=verbose,
            )
            _store_parsed_structure(
                parsed_path,
                self.atomic_numbers,
                self.coordinates,
                self.atom_species,
                self.b_factors,
            )

        # center coordinates
        if origin is None:
            self.coordinates = PDB.center_coordinates(self.coordinates)
        else:
            self.coordinates = self.coordinates - torch.tensor(
                origin, dtype=self.coordinates.dtype
            )

        # estimate max diameter
        self.max_diameter = PDB.estimate_max_diameter(self.coordinates)
        """float: Maximum diameter of the structure based on convex hull."""

    def _parse_structure(
        self,
        compute_atom_species: bool,
        readd_hydrogens: bool | str,
        monomer_library_path: str | None,
        verbose: bool,
    ) -> None:
        """
        Populate `atomic_numbers`/`coordinates`/`atom_species`/`b_factors`
        from the file.

        Split out of `__init__` so the parsed-cache path above can skip it
        wholesale. Sets exactly the four attributes the cache stores, in
        their pre-centering form.
        """
        used_library = False
        if compute_atom_species:
            znum, pos, species, bfac, used_library = PDB._build_typed_model(
                self.filepath,
                monomer_library_path,
                verbose=verbose,
                readd_hydrogens=readd_hydrogens,
            )
            self.atom_species = species
            if used_library:
                # The library completed the model with hydrogens gemmi placed
                # from ideal geometry, so Biopython's atom list -- parsed from
                # the file, which has none -- no longer describes the same
                # molecule. Take all three arrays from the completed model
                # instead, the way sffit does, so they align by construction
                # rather than by comparison. This is what makes H-containing
                # species resolvable at all; see `monomer_library_path`.
                self.atomic_numbers = torch.tensor(znum, dtype=torch.long)
                self.coordinates = torch.tensor(pos, dtype=torch.float32)
                self.b_factors = torch.tensor(bfac, dtype=torch.float32)

        if not used_library:
            # get atomic elements, coordinates and B-factors
            (
                self.atomic_numbers,
                self.coordinates,
                self.b_factors,
            ) = PDB.get_atoms_and_coordinates(self.structure, verbose=verbose)

        if compute_atom_species and not used_library:
            # Without a library the two parsers should still agree atom for
            # atom. Catch any residual disagreement here, where the structure
            # can be named, rather than letting it surface as an opaque
            # mask/tensor shape error deep in PotentialBuilder. A multi-model
            # file trips this: Biopython walks every model, gemmi types only
            # the first. With a library the two arrays come from one pass, so
            # there is nothing to reconcile.
            assert self.atom_species is not None, (
                "compute_atom_species=True always sets self.atom_species above"
            )
            if len(self.atom_species) != self.atomic_numbers.shape[0]:
                n_models = len(list(self.structure))
                raise ValueError(
                    f"{self.filepath}: bonded-species typing produced "
                    f"{len(self.atom_species)} entries for "
                    f"{self.atomic_numbers.shape[0]} atoms"
                    + (
                        f" -- the file holds {n_models} models, and only the "
                        "first can be typed. Extract a single model, or use "
                        "scattering_factors='kirkland'/'lobato'."
                        if n_models > 1
                        else ". Use scattering_factors="
                        "'kirkland'/'lobato' for this structure."
                    )
                )

    @property
    def structure(self) -> "Structure":
        """
        Bio.PDB.Structure.Structure: the parsed structure, read on first use.

        Parsed lazily because the monomer-library path never needs it: gemmi
        supplies the atom list there, and eagerly parsing the same file with
        Biopython as well doubled the cost of building a large assembly.
        """
        if self._structure is None:
            self._structure = PDB.get_pdb_structure(self.filepath)
        return self._structure

    @staticmethod
    def fetch_pdb_file(
        pdb_id: str,
        ext: str = "cif",
        pdb_cache_dir: str = DEFAULT_PDB_CACHE_DIR,
        assembly: bool | int = True,
        verbose: bool = True,
    ) -> str:
        """
        Download a PDB file and save it in a given location.

        Parameters
        ----------
        pdb_id : str
            A valid PDB ID.
        ext : str
            File ext ('cif' or 'pdb').
        pdb_cache_dir : str
            Destination folder.
        assembly : bool or int
            - True  → fetch default biological assembly (assembly 1 if available).
            - False → fetch asymmetric unit.
            - int   → fetch that specific assembly if available, fallback to default
               PDBx/mmCIF file.
        verbose : bool, optional
            Print fetch/assembly status. Default True.

        Returns
        -------
        str
            Path to the saved PDB file
        """
        # Fold `1fa2` and `1FA2` onto one cache entry -- see
        # `canonical_pdb_source`. Done here rather than at the call site so
        # every caller gets it, including `specimen.filament._tubulin`'s
        # direct fetches.
        pdb_id = canonical_pdb_source(pdb_id)

        # Guarded on `verbose` because that is the ONLY thing it does: it
        # returns None and its sole effect is a print. Unguarded it spent a
        # ~0.6 s HTTPS round trip per structure -- on every call, before the
        # cache check below, and discarded -- which was the entire cost of a
        # "cache hit" for the 26 already-downloaded structures a tomogram
        # loads.
        if verbose:
            PDB.get_available_assemblies(pdb_id, verbose=verbose)

        # Decide what to fetch
        if assembly is True:
            if verbose:
                print(f"{pdb_id}: Fetching default Biological Assembly 1")
            filename = f"{pdb_id}-assembly{1}.{ext}"
        elif assembly is False:
            if verbose:
                print(f"{pdb_id}: Fetching default PDBx/mmCIF file.")
            filename = f"{pdb_id}.{ext}"
        elif isinstance(assembly, int):
            if verbose:
                print(f"{pdb_id}: Fetching Biological Assembly {assembly}")
            filename = f"{pdb_id}-assembly{assembly}.{ext}"
        else:
            raise ValueError("assembly must be True, False, or int")

        # Build filepath
        file_path = os.path.join(pdb_cache_dir, filename)

        # Return existing file if available
        if os.path.exists(file_path):
            if verbose:
                print(f"File already exists: {file_path}, skip fetching.")
            return file_path

        # A cache written before keys were canonicalized, or by hand, may
        # hold this entry under a different spelling of the accession code
        # (`1fa2-assembly1.cif`). It is the same file: reuse it rather than
        # re-downloading and leaving the cache with two copies of one
        # structure. Only reached on a miss, so this costs one listdir.
        cached = _cache_file_ignoring_case(pdb_cache_dir, filename)
        if cached is not None:
            if verbose:
                print(f"File already exists: {cached}, skip fetching.")
            return cached

        # Fetch
        if verbose:
            print("File does not exist, fetching.")
        url = "https://files.rcsb.org/download/" + filename + ".gz"
        r = requests.get(url)
        r.raise_for_status()

        # Decompress in memory
        with gzip.open(io.BytesIO(r.content), "rt") as f:
            cif_content = f.read()

        # Save to file -- pdb_cache_dir is just a plain relative-or-absolute
        # path (resolved against the current process's cwd, not the
        # specter repo root), so create it on demand rather than crashing
        # with a raw FileNotFoundError if it doesn't exist yet.
        os.makedirs(pdb_cache_dir, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(cif_content)

        if verbose:
            print(f"Downloaded to: {file_path}")
        return file_path

    @staticmethod
    def get_available_assemblies(pdb_id: str, verbose: bool = True) -> None:
        """
        Print the available biological assembly IDs for a PDB entry.

        Parameters
        ----------
        pdb_id : str
            4-character PDB ID.
        verbose : bool, optional
            Print the result (or fetch error). Default True.

        Notes
        -----
        Prints the available assemblies to console. If the PDB entry cannot be
        accessed or does not have assembly information, prints an error message.
        """
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            assemblies = data.get("rcsb_entry_container_identifiers", {}).get(
                "assembly_ids", []
            )
            if verbose:
                print("Assemblies available: " + ", ".join(assemblies))
        except Exception as e:
            if verbose:
                print(f"Error fetching assemblies for {pdb_id}: {e}")
            return

    @staticmethod
    def get_pdb_structure(filepath: str) -> Structure:
        """
        Parse a PDB or mmCIF file and return the structure object.

        Parameters
        ----------
        filepath : str
            Path to PDB (.pdb) or mmCIF (.cif) file.

        Returns
        -------
        structure : Bio.PDB.Structure.Structure
            Parsed structure object from Biopython.

        Raises
        ------
        ValueError
            If the file format is not 'pdb' or 'cif'.
        """
        ext = filepath[-3:]
        parser: PDBParser | MMCIFParser
        if ext == "pdb":
            parser = PDBParser()
        elif ext == "cif":
            parser = MMCIFParser()
        else:
            raise ValueError(f"Invalid file format '{ext}'. Must be 'cif' or 'pdb'.")
        structure = parser.get_structure("structure", filepath)
        return structure

    @staticmethod
    def get_atoms_and_coordinates(
        structure: Structure | str,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract atomic elements, coordinates and B-factors from PDB structure.

        All three come from one walk of the same atom list, so they are
        aligned by construction -- which is what `PotentialBuilder` relies on,
        since it indexes `b_factors` positionally against `atomic_numbers`.

        Parameters
        ----------
        structure : Bio.PDB.Structure.Structure or str
            Either a parsed Biopython structure object or a filepath to a
            PDB/mmCIF file. If a filepath is provided, the structure will be
            loaded automatically.
        verbose : bool, optional
            Show the extraction progress bar. Default True.

        Returns
        -------
        elements : torch.Tensor
            Atomic numbers for each atom, shape (N,).
        coords : torch.Tensor
            Atomic coordinates (x, y, z) for each atom, shape (N, 3).
        b_factors : torch.Tensor
            Deposited isotropic B-factor for each atom in Å², shape (N,).

        Notes
        -----
        Uses Rich progress bar to display extraction progress for large structures.
        """

        # if filepath, load the structure.
        if isinstance(structure, str):
            structure = PDB.get_pdb_structure(structure)

        # Extract atomic data
        coords_list = []
        element_symbols = []
        b_factor_list = []
        for atom in track(
            structure.get_atoms(),
            description="Extracting atom coordinates and element",
            disable=not verbose,
        ):
            element_symbol = atom.element.strip().upper()
            element_symbols.append(element_symbol)

            coord = atom.get_coord()
            coords_list.append(coord)
            b_factor_list.append(atom.get_bfactor())
        coords = torch.as_tensor(np.array(coords_list))
        elements = atom_number(element_symbols)
        b_factors = torch.tensor(b_factor_list, dtype=torch.float32)

        return elements, coords, b_factors

    @staticmethod
    def _build_typed_model(
        filepath: str,
        monomer_library_path: str | None = None,
        verbose: bool = True,
        readd_hydrogens: bool | str = "auto",
    ) -> tuple[list[int], list[list[float]], list[str | None], list[float], bool]:
        """
        Build the topology-completed model and type every atom in one pass.

        Returns each atom's element, position, bonded-neighbor species
        descriptor and B-factor from the *same* gemmi model, so the four stay
        aligned by construction. :meth:`get_atom_species` returns only the
        descriptors.

        Mirrors the atom-typing approach used by `sffit
        <https://github.com/as2875/sffit>`_ to fit bonded-species electron
        scattering factors: builds a bond topology with
        ``gemmi.prepare_topology`` and describes each atom as its element
        plus the elements of its directly bonded neighbors, e.g. `"O(HH)"`
        for a water oxygen or `"C(HHHC)"` for a methyl carbon, with the same
        carboxyl/amide override sffit uses for Asp/Glu/Asn/Gln side-chain
        oxygens.

        Parameters
        ----------
        filepath : str
            Path to an mmCIF file. Bond templates come from the file's own
            embedded chemical-component definitions (`_chem_comp_bond`) plus
            explicit links (`_struct_conn`, e.g. disulfides or metal
            coordination) and standard polymer backbone connectivity, all
            resolved by gemmi. The legacy PDB format carries none of this,
            so this function requires mmCIF.
        monomer_library_path : str, optional
            Path to a CCP4 Monomer Library (as used by REFMAC/Coot/
            Servalcat). If None, falls back to the `CLIBD_MON` environment
            variable, and if that is unset too, to gemmi's built-in
            defaults. Bond topology for standard residues is available
            either way, but hydrogens can only be added with a real monomer
            library — without one, H-containing species (e.g. `"O(HH)"`,
            `"C(HHHC)"`) will not be resolved and those atoms fall back to
            `None`.
        verbose : bool, optional
            Print how many atoms were successfully typed. Default True.

        Returns
        -------
        species : list of str or None
            One entry per atom, in the same file/chain/residue/atom order
            as `get_atoms_and_coordinates`. `None` where no neighbors could
            be determined (e.g. isolated ions, or unresolved components).
        b_factors : list of float
            Isotropic B-factor per atom, Å². Atoms the monomer library adds
            carry gemmi's own value for them (0 for one built from ideal
            geometry), since the file supplies no B for an atom it does not
            contain.
        """
        if not filepath.lower().endswith("cif"):
            warnings.warn(
                "get_atom_species requires an mmCIF source (bond templates "
                "are not available in legacy PDB format); returning None "
                "for all atoms.",
                stacklevel=2,
            )
            structure = PDB.get_pdb_structure(filepath)
            znum_t, pos_t, bfac_t = PDB.get_atoms_and_coordinates(
                structure, verbose=False
            )
            return (
                znum_t.tolist(),
                pos_t.tolist(),
                [None] * int(znum_t.shape[0]),
                bfac_t.tolist(),
                False,
            )

        monlib_path = _resolve_monomer_library_path(monomer_library_path)

        st = gemmi.read_structure(filepath)
        st.setup_entities()

        if monlib_path:
            try:
                monlib = gemmi.read_monomer_lib(
                    monlib_path, st[0].get_all_residue_names()
                )
            except RuntimeError as err:
                warnings.warn(str(err), RuntimeWarning, stacklevel=2)
                monlib = gemmi.MonLib()
        else:
            _warn_missing_monomer_library()
            monlib = gemmi.MonLib()

        warnings_sink = sys.stderr if verbose else io.StringIO()

        # First pass: complete the model (adds hydrogens as dummy,
        # zero-occupancy atoms if the monomer library has geometry for
        # them), mirroring sffit's from_gemmi().
        #
        # This only runs with a real monomer library. `HydrogenChange.ReAdd`
        # *removes* every existing hydrogen before re-adding it from library
        # geometry, so without a library to re-add from it is not a no-op: it
        # strips the file's own hydrogens outright (22FX loses 4968 atoms),
        # leaving fewer species than there are atoms in `atomic_numbers`.
        #
        # The added atoms are tracked by identity rather than recognised by
        # their zero occupancy later: deposited structures may legitimately
        # contain zero-occupancy atoms (1FA2 has 208 of them), and dropping
        # those as "dummies" would misalign the two lists the same way.
        # A library is what supplies hydrogens; without one the completed
        # model is just the file's own atoms.
        used_library = bool(len(monlib.monomers))

        # "auto": keep hydrogens a structure already carries, add them when it
        # carries none. Deposited hydrogens are information the file provides,
        # and there is no reason to move them -- the species descriptor comes
        # from the bond graph, so the fitted factors apply either way.
        if readd_hydrogens == "auto":
            readd_hydrogens = not any(
                cra.atom.element.atomic_number == 1 for cra in st[0].all()
            )
        else:
            readd_hydrogens = bool(readd_hydrogens)

        # Drop explicit metal-coordination links, as sffit does before typing,
        # but only when a library is in use -- which is the only configuration
        # sffit supports. The library's HEM component defines the four
        # porphyrin Fe-N bonds itself, so removing the _struct_conn links
        # leaves Fe(NNNN), the 4-coordinate entry the table provides, instead
        # of Fe(NNNNNOO), which no table contains. Without a library those
        # internal bonds come only from the file's own (incomplete)
        # _chem_comp_bond, and removing the links strips what was compensating:
        # 1mbo degrades to Fe(NNN) and 1A6M to Fe(N).
        #
        # Deliberately NOT adopted from sffit's from_gemmi: `expand_ncs`, since
        # specter fetches biological assemblies from RCSB and would
        # double-expand them; and keeping alternate conformers, since
        # PotentialBuilder applies no occupancy weighting and would render both
        # at full strength.
        if used_library:
            for i in reversed(range(len(st.connections))):
                if st.connections[i].type is gemmi.ConnectionType.MetalC:
                    del st.connections[i]

        # prepare_topology rewrites each connection's link_id as a side effect,
        # so keep the list to restore before the second pass (sffit does the
        # same). Snapshotted *after* the metal links are dropped: taking it
        # earlier would restore them for the second pass, which is the one
        # that builds the bond graph the typing actually reads.
        conlist = gemmi.ConnectionList(st.connections)

        added_keys: set[_AtomKey] = set()
        if used_library:
            # sffit couples these two: with ReAdd, gemmi has already placed the
            # hydrogens, so only missing *heavy* atoms are backfilled. With
            # NoChange the file's own hydrogens are kept and the ones it lacks
            # are backfilled as zero-occupancy dummies -- present for typing,
            # never rendered.
            topo = gemmi.prepare_topology(
                st,
                monlib,
                h_change=(
                    gemmi.HydrogenChange.ReAdd
                    if readd_hydrogens
                    else gemmi.HydrogenChange.NoChange
                ),
                warnings=warnings_sink,
            )
            for m in topo.find_missing_atoms(including_hydrogen=not readd_hydrogens):
                # A residue the library has no entry for (including the
                # blank-named components some entries carry) can't be
                # completed; its atoms keep whatever bonds the file provides.
                if m.res_id.name not in monlib.monomers:
                    continue
                mon = monlib.monomers[m.res_id.name]
                monat = mon.find_atom(m.atom_name)
                if monat is None:
                    continue
                atom = gemmi.Atom()
                atom.occ = 0.0
                atom.element = monat.el
                atom.name = m.atom_name
                cra = st[0].find_cra(m)
                added_keys.add(_atom_key(cra.chain, cra.residue, atom))
                cra.residue.add_atom(atom)

        # Second pass: final bond graph, now including any added atoms.
        st.connections = conlist
        topo = gemmi.prepare_topology(
            st, monlib, h_change=gemmi.HydrogenChange.NoChange, warnings=warnings_sink
        )

        # Atom objects returned by gemmi are stable dict keys for the same
        # underlying atom (mirrors sffit's `lookup` pattern in from_gemmi).
        # Neighbors are tracked by atom identity, not element, so that e.g.
        # a methyl carbon's three separate hydrogen neighbors are each
        # counted (not collapsed into one "H").
        identity = {
            cra.atom: _atom_key(cra.chain, cra.residue, cra.atom) for cra in st[0].all()
        }
        element_of = {key: atom.element.name for atom, key in identity.items()}
        neighbor_keys: dict[_AtomKey, set[_AtomKey]] = defaultdict(set)
        for bond in topo.bonds:
            a, b = bond.atoms
            ka, kb = identity[a], identity[b]
            neighbor_keys[ka].add(kb)
            neighbor_keys[kb].add(ka)

        species: list[str | None] = []
        atomic_numbers: list[int] = []
        positions: list[list[float]] = []
        b_factors: list[float] = []
        n_matched = 0
        seen: set[_AtomKey] = set()
        for cra in st[0].all():
            key = _atom_key(cra.chain, cra.residue, cra.atom)
            if key in added_keys:
                # dummy atom added only to inform its neighbors' typing
                continue
            if key in seen:
                # alternate conformer of an atom already typed (gemmi's
                # `all()` visits every altloc; Biopython/get_atoms_and_
                # coordinates collapses them to one atom per position, so
                # we keep only the first here to stay index-aligned with
                # atomic_numbers/coordinates — bonding topology doesn't
                # differ between altlocs of the same atom in practice).
                continue
            if used_library and cra.atom.occ == 0.0:
                # gemmi zero-occupancies the hydrogens whose presence or
                # position is ambiguous -- a rotatable Ser/Thr/Tyr hydroxyl H,
                # or both tautomer hydrogens of a histidine, only one of which
                # is really there. sffit drops these from its fit (it selects
                # ";q>0") and so must we: PotentialBuilder has no occupancy
                # weighting, so keeping them would render every His
                # doubly protonated. They still inform their neighbours'
                # descriptors, since the bond graph above spans every atom.
                continue
            seen.add(key)
            atomic_numbers.append(cra.atom.element.atomic_number)
            positions.append([cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z])
            b_factors.append(cra.atom.b_iso)
            neighbors = neighbor_keys.get(key)
            if not neighbors:
                species.append(None)
                continue

            elem = cra.atom.element.name
            neighbor_elems = [element_of[nk] for nk in neighbors]
            neighbor_str = "".join(
                sorted(neighbor_elems, key=lambda e: gemmi.Element(e).atomic_number)
            )
            descriptor = f"{elem}({neighbor_str})"
            if cra.atom.name in ("OD1", "OD2", "OE1", "OE2"):
                if cra.residue.name in ("ASP", "GLU"):
                    descriptor = f"{elem}({neighbor_str}, carboxyl)"
                elif cra.residue.name in ("ASN", "GLN"):
                    descriptor = f"{elem}({neighbor_str}, amide)"
            species.append(descriptor)
            n_matched += 1

        if verbose:
            print(f"[get_atom_species] {n_matched}/{len(species)} atoms typed")

        return atomic_numbers, positions, species, b_factors, used_library

    @staticmethod
    def get_atom_species(
        filepath: str,
        monomer_library_path: str | None = None,
        verbose: bool = True,
    ) -> list[str | None]:
        """
        Determine each atom's bonded-neighbor species descriptor.

        Thin wrapper over :meth:`_build_typed_model`, which documents the
        typing rules and the monomer-library behaviour.

        Returns
        -------
        species : list of str or None
            One entry per atom, in the same order as the model this was
            derived from. ``None`` where no neighbors could be determined.
        """
        _, _, species, _, _ = PDB._build_typed_model(
            filepath, monomer_library_path, verbose
        )
        return species

    @staticmethod
    def center_of_particle(coords: torch.Tensor) -> torch.Tensor:
        """
        Return a particle's geometric center.

        Parameters
        ----------
        coords : torch.Tensor
            Atom coordinates of molecule with N atoms, shape (N, 3).

        Returns
        -------
        center : torch.Tensor
            Geometric center of the molecule, shape (3,).
        """
        center = coords.mean(dim=0)
        return center

    @staticmethod
    def center_coordinates(coords: torch.Tensor) -> torch.Tensor:
        """
        Centers coordinates on its geometric center.

        Parameters
        ----------
        coords : tensor
            Atom coordinates of molecule with N atoms, shape (N,3)

        Returns
        -------
        centered_coordinates : tensor
            Centered coordinates, shape (N,3)
        """
        center = PDB.center_of_particle(coords)
        return coords - center

    @staticmethod
    def estimate_max_diameter(coordinates: torch.Tensor) -> float:
        """
        Estimate the maximum diameter of a structure using convex hull.

        Parameters
        ----------
        coordinates : torch.Tensor
            Atomic coordinates with shape (N, 3).

        Returns
        -------
        max_diameter : float
            Maximum pairwise distance between convex hull vertices.

        Notes
        -----
        Computes the convex hull of the coordinates and returns the maximum
        distance between any two hull vertices.
        """
        hull = ConvexHull(coordinates)
        hull_points = coordinates[hull.vertices]
        max_diameter = pdist(hull_points).max()
        return max_diameter
