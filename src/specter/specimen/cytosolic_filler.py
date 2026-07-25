"""
Reference list of generic, non-target cytosolic macromolecules, for filling
the "everything else" background of a CryoETSpecimenGenerator specimen
(crowding below/around whatever specific species you're annotating as
targets).

Source (full credit): Pei L, Xu M, Frazier Z, Alber F. "Simulating cryo
electron tomograms of crowded cell cytoplasm for assessment of automated
particle picking." BMC Bioinformatics. 2016;17(1):405.
doi:10.1186/s12859-016-1283-3. PMCID: PMC5050594.
https://pmc.ncbi.nlm.nih.gov/articles/PMC5050594/

The entries below (PDB code, name, molecular weight, minimum bounding
sphere radius, relative occurrence frequency) are transcribed from that
paper's Additional file 1 (Table S1), which is not rendered in the article's
HTML/PDF and had to be pulled from the journal's supplementary-material
store directly. `occurrence_freq` is the paper's own relative-abundance
weighting (their normalization of proteomic abundance data), reused here
only for the *ratio* between species -- see build_filler_protein_specs.

The paper's table has 21 rows; only 20 are listed here. Its 2AWB entry
(bacterial 70S ribosome) was obsoleted by the PDB on 2014-12-10 and merged
into 4V4Q -- confirmed live against RCSB (2026-07-24: 2AWB 404s, the other
20 all 200 OK). Dropped rather than repointed to 4V4Q, since it would have
just duplicated a ribosome-class target's size range anyway.

This is one reasonable, published reference set for "generic crowded
cytoplasm," not a claim about the true composition of any specific real
specimen -- there isn't one universal answer to what's "really" in the
cytoplasm (composition varies by organism/cell state, and most of it is
below any real dataset's identification threshold anyway). Swap in a
different/broader list freely; nothing downstream depends on this exact set.
"""

from __future__ import annotations

PEI2016_CROWDING_TABLE: list[dict] = [
    {
        "code": "1A1S",
        "name": "Ornithine carbamoyltransferase (P. furiosus)",
        "mw_kda": 35.10,
        "radius_nm": 3.9597,
        "occurrence_freq": 0.0604,
    },
    {
        "code": "1EQR",
        "name": "Aspartyl-tRNA synthetase (E. coli)",
        "mw_kda": 198.06,
        "radius_nm": 8.2916,
        "occurrence_freq": 0.0671,
    },
    {
        "code": "1GYT",
        "name": "Aminopeptidase A / PepA (E. coli)",
        "mw_kda": 661.79,
        "radius_nm": 12.2575,
        "occurrence_freq": 0.0094,
    },
    {
        "code": "1KYI",
        "name": "HslUV-NLVS complex (H. influenzae)",
        "mw_kda": 834.91,
        "radius_nm": 10.2705,
        "occurrence_freq": 0.0677,
    },
    {
        "code": "1VPX",
        "name": "Transaldolase (T. maritima)",
        "mw_kda": 516.09,
        "radius_nm": 9.0134,
        "occurrence_freq": 0.0469,
    },
    {
        "code": "1W6T",
        "name": "Octameric enolase (S. pneumoniae)",
        "mw_kda": 97.70,
        "radius_nm": 4.7670,
        "occurrence_freq": 0.0072,
    },
    {
        "code": "2BYU",
        "name": "Small heat-shock protein Acr1 (M. tuberculosis)",
        "mw_kda": 148.52,
        "radius_nm": 6.6332,
        "occurrence_freq": 0.0405,
    },
    {
        "code": "2GLS",
        "name": "Glutamine synthetase",
        "mw_kda": 623.83,
        "radius_nm": 8.3666,
        "occurrence_freq": 0.0709,
    },
    {
        "code": "2IDB",
        "name": "UbiD decarboxylase (E. coli)",
        "mw_kda": 172.94,
        "radius_nm": 6.6077,
        "occurrence_freq": 0.0715,
    },
    {
        "code": "3DY4",
        "name": "Yeast 20S proteasome + spirolactacystin",
        "mw_kda": 705.07,
        "radius_nm": 9.7921,
        "occurrence_freq": 0.0117,
    },
    {
        "code": "1BXR",
        "name": "Carbamoyl phosphate synthetase + AMPPNP",
        "mw_kda": 644.91,
        "radius_nm": 12.0576,
        "occurrence_freq": 0.0719,
    },
    {
        "code": "1F1B",
        "name": "Aspartate transcarbamoylase P268A (E. coli)",
        "mw_kda": 103.55,
        "radius_nm": 6.7334,
        "occurrence_freq": 0.0709,
    },
    {
        "code": "1KP8",
        "name": "GroEL-KMgATP14 chaperonin",
        "mw_kda": 810.24,
        "radius_nm": 9.8714,
        "occurrence_freq": 0.0360,
    },
    {
        "code": "1QO1",
        "name": "ATP synthase rotary motor (yeast mitochondria)",
        "mw_kda": 448.86,
        "radius_nm": 10.6771,
        "occurrence_freq": 0.0593,
    },
    {
        "code": "1VRG",
        "name": "Propionyl-CoA carboxylase, beta subunit (T. maritima)",
        "mw_kda": 352.30,
        "radius_nm": 6.8439,
        "occurrence_freq": 0.0105,
    },
    {
        "code": "1YG6",
        "name": "ATP-dependent Clp protease",
        "mw_kda": 303.87,
        "radius_nm": 6.8267,
        "occurrence_freq": 0.0313,
    },
    {
        "code": "2BO9",
        "name": "Carboxypeptidase A4 + latexin",
        "mw_kda": 123.01,
        "radius_nm": 6.6997,
        "occurrence_freq": 0.0678,
    },
    {
        "code": "2GHO",
        "name": "RNA polymerase (T. aquaticus)",
        "mw_kda": 333.29,
        "radius_nm": 7.6688,
        "occurrence_freq": 0.0587,
    },
    {
        "code": "2H12",
        "name": "Citrate synthase (A. aceti)",
        "mw_kda": 298.56,
        "radius_nm": 7.0397,
        "occurrence_freq": 0.0711,
    },
    {
        "code": "2REC",
        "name": "RecA hexamer",
        "mw_kda": 228.10,
        "radius_nm": 7.0887,
        "occurrence_freq": 0.0486,
    },
]


def build_filler_protein_specs(
    codes: list[str] | None = None,
    exclude_codes: list[str] | None = None,
    total_occupancy: float = 0.05,
    **overrides,
) -> list[dict]:
    """Build protein_specs entries for generic cytosolic filler, from
    PEI2016_CROWDING_TABLE (see module docstring for source/credit).

    Splits `total_occupancy` across the selected species in proportion to
    their relative occurrence_freq in the table (renormalized over just the
    selected subset), giving each a PMER_OCC. Only the *ratio* between
    species is taken from the source paper -- total_occupancy itself has no
    universally-correct value (see cryoet.py's discussion of this); treat
    the default as a starting point to tune empirically against your own
    target volume and resolution, not a physical constant.

    Parameters
    ----------
    codes : list of str, optional
        PDB codes to include (must exist in PEI2016_CROWDING_TABLE).
        Defaults to the full table.
    exclude_codes : list of str, optional
        PDB codes to drop from the (defaulted-to-full) table. Mutually
        exclusive with ``codes``.
    total_occupancy : float, optional
        Combined PMER_OCC budget across all selected species.
    **overrides
        Extra protein_specs keys applied to every generated entry, e.g.
        ``PMER_OVER_TOL=0.02``.

    Returns
    -------
    list[dict]
        One entry per selected species, each
        ``{"PDB_CODE": ..., "PMER_OCC": ..., **overrides}`` -- ready to
        concatenate onto your own protein_specs list.
    """
    if codes is not None and exclude_codes is not None:
        raise ValueError("pass only one of codes / exclude_codes, not both")

    table = PEI2016_CROWDING_TABLE
    if codes is not None:
        by_code = {e["code"]: e for e in table}
        missing = [c for c in codes if c not in by_code]
        if missing:
            raise ValueError(f"not in PEI2016_CROWDING_TABLE: {missing}")
        selected = [by_code[c] for c in codes]
    elif exclude_codes is not None:
        selected = [e for e in table if e["code"] not in exclude_codes]
    else:
        selected = table

    freq_sum = sum(e["occurrence_freq"] for e in selected)
    return [
        {
            "PDB_CODE": e["code"],
            "PMER_OCC": total_occupancy * e["occurrence_freq"] / freq_sum,
            **overrides,
        }
        for e in selected
    ]
