"""
Reference lists of generic, non-target cytosolic macromolecules, for
filling the "everything else" background of a specimen (crowding
below/around whatever specific species you're annotating as targets).

Two independent, additive tables, both adapted by the same function --
`build_filler_pool_specs` -- to `TomogramSpecimenGenerator`'s
(`specter build tomogram`) flat ``{"pdb_source"}`` filler_specs format.
`specter build tomogram`'s `filler_from_pei2016`/`filler_from_cryoetsim`
both route through this one function regardless of which table they pull
from:

- `PEI2016_CROWDING_TABLE`, weighted by the source paper's own
  relative-abundance data.
- `CRYOETSIM_PARTICLE_TABLE` (see its own docstring below).

## PEI2016_CROWDING_TABLE

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
only for the *ratio* between species -- see build_filler_pool_specs.

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


# =============================================================================
# CryoETSim reference table -- a second, independent filler source, additive
# to PEI2016_CROWDING_TABLE above (see build_filler_pool_specs's own
# docstring for how the two combine).
# =============================================================================

# Source (full credit): Stojanovska F, Sanchez RM, Jensen RK, Mahamid J,
# Kreshuk A, Zaugg JB. "CryoSiam: self-supervised representation learning
# for automated analysis of cryo-electron tomograms." bioRxiv
# 2025.11.11.687379 (2025). doi:10.1101/2025.11.11.687379
#
# The entries below (PDB code, category, molecular weight, name) are
# transcribed from that preprint's Supplementary Note 1, Supplementary
# Table 2 ("Macromolecular complexes included in the simulated tomograms
# of the CryoETSim dataset").
#
# The source table has ~140 rows across 7 "Selection type" categories. Only
# 4 are included here -- macromolecules, distractors,
# transcription_translation, nucleosomes (~129 entries) -- since these are
# the only ones with plain, directly RCSB-fetchable 4-character PDB codes
# (matching what `specter.pdb.PDB(pdb_source=...)` expects). The other
# three categories are deliberately excluded, not just forgotten:
#
# - cytoskeleton (``MT_6o2tx2``, ``actin cofilactin``, ``actin long``,
#   ``neuroMT``) -- custom-built structures with no RCSB entry and no size
#   given in the source table, not a simple code fetch.
# - DNA filaments (``dna_len_<N>_persis_<P>``, ~70 rows) -- synthetic
#   worm-like-chain-simulated filaments (see the preprint's Supplementary
#   Figure 1 / the PolymerCpp tool), not PDB structures at all.
# - membrane proteins -- mostly non-standard aliases for modified/renamed
#   structures (``GPROTEIN``, ``6LFM_GPROTEIN``, ``6N52_GLUM``,
#   ``4COF_GABAAR``, ``6HIS_5HT``, ``6WHV_GLUN1``), not plain RCSB codes;
#   only ``5SOA`` is a clean code and wasn't judged worth a single-entry
#   category on its own. A handful of macromolecules-category rows in the
#   source table (``6lfm_gprotienpdb``, ``6n52_GluMpdb``, ``6his_5HTpdb``)
#   are the same underlying structures under this same non-standard-naming
#   problem and are excluded for the same reason.
#
# Revisit the membrane proteins and cytoskeleton categories once there's a
# real use for them -- the per-category reasons above (non-standard
# aliases, no RCSB entry, not PDB structures at all) are data-format
# problems independent of any particular generator's own placement
# capabilities, so this isn't blocked on anything in specter itself.
#
# This is one reasonable, published reference set, not a claim about the
# true composition of any specific real specimen -- swap in a different/
# broader list freely; nothing downstream depends on this exact set.
CRYOETSIM_PARTICLE_TABLE: list[dict] = [
    # --- macromolecules ---
    {"code": "7ELY", "category": "macromolecules", "mw_kda": 2.90, "name": "peptide"},
    {
        "code": "7BLG",
        "category": "macromolecules",
        "mw_kda": 18.14,
        "name": "sugar binding protein",
    },
    {
        "code": "3QM1",
        "category": "macromolecules",
        "mw_kda": 31.31,
        "name": "serine cinnamoyl esterase",
    },
    {
        "code": "1S3X",
        "category": "macromolecules",
        "mw_kda": 42.75,
        "name": "human Hsp70 ATPase",
    },
    {
        "code": "6IOD",
        "category": "macromolecules",
        "mw_kda": 55.33,
        "name": "UdgX in complex with single-stranded DNA",
    },
    {
        "code": "7S7K",
        "category": "macromolecules",
        "mw_kda": 59.67,
        "name": "EphB2 extracellular domain",
    },
    {
        "code": "4WRM",
        "category": "macromolecules",
        "mw_kda": 74.48,
        "name": "human CSF-1:CSF-1R complex",
    },
    {
        "code": "3H84",
        "category": "macromolecules",
        "mw_kda": 79.04,
        "name": "GET3 chaperone",
    },
    {
        "code": "3GL1",
        "category": "macromolecules",
        "mw_kda": 84.61,
        "name": "ATPase domain of Ssb1 chaperone",
    },
    {
        "code": "4UIC",
        "category": "macromolecules",
        "mw_kda": 86.71,
        "name": "S-layer protein rSbsC, sugar binding protein",
    },
    {
        "code": "7E6G",
        "category": "macromolecules",
        "mw_kda": 122.34,
        "name": "diguanylate cyclase SiaD in complex with its activator SiaC",
    },
    {
        "code": "6JYO",
        "category": "macromolecules",
        "mw_kda": 135.63,
        "name": "GII.13/21 noroviruses recognize glycans with a terminal "
        "beta-galactose via an unconventional glycan binding site",
    },
    {
        "code": "7VTG",
        "category": "macromolecules",
        "mw_kda": 136.04,
        "name": "Pseudouridine kinase (PUKI) S30A mutant from Escherichia "
        "coli strain B",
    },
    {
        "code": "7B5S",
        "category": "macromolecules",
        "mw_kda": 166.62,
        "name": "Ubiquitin ligation to F-box protein substrates by SCF-RBR "
        "E3-E3 super-assembly: CUL1-RBX1-ARIH1 Ariadne. Transition State 1",
    },
    {
        "code": "7SHK",
        "category": "macromolecules",
        "mw_kda": 172.48,
        "name": "Xenopus laevis CRL2Lrr1",
    },
    {
        "code": "5LJO",
        "category": "macromolecules",
        "mw_kda": 180.07,
        "name": "E. coli BAM complex (BamABCDE), membrane protein",
    },
    {
        "code": "2CG9",
        "category": "macromolecules",
        "mw_kda": 188.73,
        "name": "Hsp90-Sba1 closed chaperone complex",
    },
    {
        "code": "7SGM",
        "category": "macromolecules",
        "mw_kda": 193.89,
        "name": "Fab variant containing a fluorescent noncanonical amino "
        "acid with blocked excited state proton transfer and in complex "
        "with its antigen",
    },
    {
        "code": "5A20",
        "category": "macromolecules",
        "mw_kda": 197.50,
        "name": "bacteriophage SPP1 head-to-tail interface filled with DNA "
        "and tape measure protein",
    },
    {
        "code": "6ZIU",
        "category": "macromolecules",
        "mw_kda": 197.83,
        "name": "bovine ATP synthase stator domain, state 3 hydrolase",
    },
    {
        "code": "6VGR",
        "category": "macromolecules",
        "mw_kda": 203.29,
        "name": "Human Dipeptidase 3 in Complex with Fab of SC-003 hydrolase",
    },
    {
        "code": "2R9R",
        "category": "macromolecules",
        "mw_kda": 206.88,
        "name": "membrane/transport protein",
    },
    {
        "code": "6LX3",
        "category": "macromolecules",
        "mw_kda": 208.31,
        "name": "human secretory immunoglobulin A",
    },
    {
        "code": "7WBT",
        "category": "macromolecules",
        "mw_kda": 209.00,
        "name": "bovine NLRP9",
    },
    {
        "code": "5CSA",
        "category": "macromolecules",
        "mw_kda": 209.23,
        "name": "BT-BCCP-AC1-AC5 of yeast acetyl-CoA carboxylase, ligase",
    },
    {
        "code": "1UL1",
        "category": "macromolecules",
        "mw_kda": 214.07,
        "name": "FEN1-PCNA complex, hydrolase/dna-binding protein",
    },
    {
        "code": "3ULV",
        "category": "macromolecules",
        "mw_kda": 226.08,
        "name": "human TLR3ecd with three Fabs",
    },
    {
        "code": "5JH9",
        "category": "macromolecules",
        "mw_kda": 230.78,
        "name": "prApe1 hydrolase",
    },
    {
        "code": "3D2F",
        "category": "macromolecules",
        "mw_kda": 236.10,
        "name": "Crystal structure of a complex of Sse1p and Hsp70, chaperone",
    },
    {
        "code": "7NIU",
        "category": "macromolecules",
        "mw_kda": 238.79,
        "name": "Nanodisc reconstituted human ABCB4 in complex with 4B1-Fab "
        "and QA2-Fab",
    },
    {
        "code": "1U6G",
        "category": "macromolecules",
        "mw_kda": 238.82,
        "name": "Cand1-Cul1-Roc1 Complex, ligase",
    },
    {
        "code": "7BLR",
        "category": "macromolecules",
        "mw_kda": 241.79,
        "name": "Vps5 (SNX-BAR) complex",
    },
    {
        "code": "2RHS",
        "category": "macromolecules",
        "mw_kda": 245.45,
        "name": "PheRS, ligase",
    },
    {
        "code": "2WW2",
        "category": "macromolecules",
        "mw_kda": 252.90,
        "name": "Family GH92 Inverting Mannosidase BT2199 from Bacteroides "
        "thetaiotaomicron VPI-5482",
    },
    {
        "code": "1N9G",
        "category": "macromolecules",
        "mw_kda": 255.99,
        "name": "thioester reductase, hydrolase",
    },
    {
        "code": "3CF3",
        "category": "macromolecules",
        "mw_kda": 270.87,
        "name": "Structure of P97/vcp in complex with ADP",
    },
    {
        "code": "7KFE",
        "category": "macromolecules",
        "mw_kda": 277.45,
        "name": "Bundibugyo virus GP (mucin deleted) bound to antibody Fab BDBV-329",
    },
    {
        "code": "6ZQJ",
        "category": "macromolecules",
        "mw_kda": 279.60,
        "name": "trimeric prME spike of Spondweni virus",
    },
    {
        "code": "1BXN",
        "category": "macromolecules",
        "mw_kda": 279.81,
        "name": "rubisco lyase",
    },
    {
        "code": "6KRK",
        "category": "macromolecules",
        "mw_kda": 288.88,
        "name": "Peroxiredoxin from Aeropyrum pernix K1",
    },
    {
        "code": "6LMT",
        "category": "macromolecules",
        "mw_kda": 295.89,
        "name": "Cryo-EM structure of the killifish CALHM1",
    },
    {
        "code": "1QVR",
        "category": "macromolecules",
        "mw_kda": 296.68,
        "name": "ClpB, chaperone",
    },
    {
        "code": "5H0S",
        "category": "macromolecules",
        "mw_kda": 297.39,
        "name": "VP1A and VP1B",
    },
    {
        "code": "7JSN",
        "category": "macromolecules",
        "mw_kda": 304.61,
        "name": "Visual Signaling Complex between Transducin and Phosphodiesterase 6",
    },
    {
        "code": "7K5X",
        "category": "macromolecules",
        "mw_kda": 311.14,
        "name": "chromatosome containing human linker histone H1.0",
    },
    {
        "code": "6WZT",
        "category": "macromolecules",
        "mw_kda": 323.80,
        "name": "influenza hemagglutinin A/Victoria/361/2011 in complex "
        "with cyno antibody 3B10",
    },
    {
        "code": "3MKQ",
        "category": "macromolecules",
        "mw_kda": 333.41,
        "name": "yeast alpha/betaprime-COP subcomplex of the COPI vesicular coat",
    },
    {
        "code": "6LXK",
        "category": "macromolecules",
        "mw_kda": 352.70,
        "name": "Z2B3 D102R Fab in complex with influenza virus "
        "neuraminidase from A/Serbia/NS-601/2014",
    },
    {
        "code": "6Z80",
        "category": "macromolecules",
        "mw_kda": 360.87,
        "name": "stimulatory human GTP cyclohydrolase I - GFRP complex",
    },
    {
        "code": "6CES",
        "category": "macromolecules",
        "mw_kda": 370.31,
        "name": "GATOR1-RAG",
    },
    {
        "code": "2XNX",
        "category": "macromolecules",
        "mw_kda": 371.16,
        "name": "BC1 fragment of streptococcal M1 protein in complex with "
        "human fibrinogen",
    },
    {
        "code": "6KSP",
        "category": "macromolecules",
        "mw_kda": 382.87,
        "name": "Rat GluD1 receptor(splayed conformation) in complex with "
        "7-CKA and Calcium ions",
    },
    {"code": "1SS8", "category": "macromolecules", "mw_kda": 386.04, "name": "GroEL"},
    {
        "code": "6VN1",
        "category": "macromolecules",
        "mw_kda": 392.96,
        "name": "glycoprotein B-Neutralizing Antibody Complex",
    },
    {
        "code": "6PIF",
        "category": "macromolecules",
        "mw_kda": 421.50,
        "name": "V. cholerae TniQ-Cascade complex, RNA bindng protein",
    },
    {
        "code": "2DFS",
        "category": "macromolecules",
        "mw_kda": 453.49,
        "name": "Myosin-V, transport protein",
    },
    {
        "code": "6CNJ",
        "category": "macromolecules",
        "mw_kda": 470.11,
        "name": "2alpha3beta stiochiometry of the human Alpha4Beta2 nicotinic receptor",
    },
    {
        "code": "6X5Z",
        "category": "macromolecules",
        "mw_kda": 485.51,
        "name": "Bovine Cardiac Myosin in Complex with Chicken Skeletal "
        "Actin and Human Cardiac Tropomyosin in the Rigor State",
    },
    {
        "code": "6AHU",
        "category": "macromolecules",
        "mw_kda": 492.83,
        "name": "Ribonuclease P with mature tRNA",
    },
    {
        "code": "7KJ2",
        "category": "macromolecules",
        "mw_kda": 495.88,
        "name": "SARS-CoV-2 Spike Glycoprotein with one ACE2 Bound",
    },
    {
        "code": "6U8Q",
        "category": "macromolecules",
        "mw_kda": 511.64,
        "name": "HIV-1 cleaved synaptic complex (CSC) intasome",
    },
    {
        "code": "6KLH",
        "category": "macromolecules",
        "mw_kda": 513.21,
        "name": "Dimeric structure of Machupo virus polymerase bound to vRNA promoter",
    },
    {
        "code": "3LUE",
        "category": "macromolecules",
        "mw_kda": 541.23,
        "name": "alpha-actinin CH1 bound to F-actin",
    },
    {
        "code": "2VZ9",
        "category": "macromolecules",
        "mw_kda": 548.04,
        "name": "Mammalian Fatty Acid Synthase in complex with NADP",
    },
    {
        "code": "7B7U",
        "category": "macromolecules",
        "mw_kda": 553.88,
        "name": "mammalian RNA polymerase II in complex with human RPAP2",
    },
    {
        "code": "5O32",
        "category": "macromolecules",
        "mw_kda": 569.31,
        "name": "complement complex, immune system",
    },
    {
        "code": "6M04",
        "category": "macromolecules",
        "mw_kda": 595.87,
        "name": "human homo-hexameric LRRC8D channel",
    },
    {
        "code": "6TGC",
        "category": "macromolecules",
        "mw_kda": 602.54,
        "name": "ternary DOCK2-ELMO1-RAC1 complex",
    },
    {
        "code": "7SFW",
        "category": "macromolecules",
        "mw_kda": 606.58,
        "name": "Venezuelan Equine Encephalitis virus (VEEV) TC-83 strain "
        "VLP in complex with Fab hVEEV-63",
    },
    {
        "code": "6BQ1",
        "category": "macromolecules",
        "mw_kda": 607.18,
        "name": "Human PI4KIIIa lipid kinase complex",
    },
    {
        "code": "6SCJ",
        "category": "macromolecules",
        "mw_kda": 617.95,
        "name": "human thyroglobulin",
    },
    {
        "code": "7NYZ",
        "category": "macromolecules",
        "mw_kda": 618.62,
        "name": "MukBEF-MatP-DNA monomer",
    },
    {
        "code": "7R04",
        "category": "macromolecules",
        "mw_kda": 635.36,
        "name": "Neurofibromin in open conformation",
    },
    {
        "code": "7NHS",
        "category": "macromolecules",
        "mw_kda": 644.57,
        "name": "Wzc K540M C8",
    },
    {
        "code": "7E8H",
        "category": "macromolecules",
        "mw_kda": 653.34,
        "name": "Kv4.2-DPP6S-KChIP1 complex",
    },
    {
        "code": "6TAV",
        "category": "macromolecules",
        "mw_kda": 666.20,
        "name": "endopeptidase-induced alpha2-macroglobulin",
    },
    {
        "code": "6XF8",
        "category": "macromolecules",
        "mw_kda": 689.88,
        "name": "DLP 5 fold",
    },
    {
        "code": "7AMV",
        "category": "macromolecules",
        "mw_kda": 698.63,
        "name": "poxvirus transcription pre-initiation complex, vRNAP",
    },
    {
        "code": "7DD9",
        "category": "macromolecules",
        "mw_kda": 715.19,
        "name": "Ams1 and Nbr1 complex, hydrolase, S. pombe",
    },
    {
        "code": "6Z3A",
        "category": "macromolecules",
        "mw_kda": 721.57,
        "name": "Mec1-Ddc2 (wild-type) in complex with AMP-PNP",
    },
    {
        "code": "6TA5",
        "category": "macromolecules",
        "mw_kda": 733.11,
        "name": "OprM-MexA complex from the MexAB-OprM Pseudomonas "
        "aeruginosa whole assembly reconstituted in nanodiscs",
    },
    {
        "code": "7Q21",
        "category": "macromolecules",
        "mw_kda": 735.45,
        "name": "III2-IV2 respiratory supercomplex from Corynebacterium glutamicum",
    },
    {
        "code": "6LXV",
        "category": "macromolecules",
        "mw_kda": 751.32,
        "name": "phosphoketolase from Bifidobacterium longum",
    },
    {
        "code": "7KDV",
        "category": "macromolecules",
        "mw_kda": 759.89,
        "name": "Murine core lysosomal multienzyme complex (LMC) composed "
        "of acid beta-galactosidase (GLB1) and protective protein "
        "cathepsin A (PPCA, CTSA). hydrolase",
    },
    {
        "code": "7LSY",
        "category": "macromolecules",
        "mw_kda": 762.36,
        "name": "NHEJ Short-range synaptic complex",
    },
    {
        "code": "5FMG",
        "category": "macromolecules",
        "mw_kda": 764.27,
        "name": "Proteasome",
    },
    {
        "code": "7EGQ",
        "category": "macromolecules",
        "mw_kda": 801.37,
        "name": "Coupling of N7-methyltransferase and 3'-5' exoribonuclease "
        "with polymerase reveals mechanisms for capping and proofreading",
    },
    {
        "code": "1G3I",
        "category": "macromolecules",
        "mw_kda": 826.23,
        "name": "protease-chaperone complex",
    },
    {
        "code": "6W6M",
        "category": "macromolecules",
        "mw_kda": 874.43,
        "name": "V. cholerae Type IV competence pilus secretin PilQ",
    },
    {
        "code": "6TPS",
        "category": "macromolecules",
        "mw_kda": 880.41,
        "name": "early intermediate RNA Polymerase I Pre-initiation complex - eiPIC",
    },
    {
        "code": "7O01",
        "category": "macromolecules",
        "mw_kda": 918.44,
        "name": "Dimeric Photosystem I of a temperature sensitive mutant "
        "Chlamydomonas reinhardtii",
    },
    {
        "code": "7ETM",
        "category": "macromolecules",
        "mw_kda": 943.62,
        "name": "C6 portal vertex in the enveloped virion capsid",
    },
    {
        "code": "6MRC",
        "category": "macromolecules",
        "mw_kda": 944.43,
        "name": "ADP-bound human mitochondrial Hsp60-Hsp10 football complex, chaperone",
    },
    {
        "code": "6VZ8",
        "category": "macromolecules",
        "mw_kda": 954.61,
        "name": "arabidopsis thaliana acetohydroxyacid synthase complex "
        "with valine bound",
    },
    {"code": "6KS8", "category": "macromolecules", "mw_kda": 968.94, "name": "TRiC"},
    {
        "code": "6EMK",
        "category": "macromolecules",
        "mw_kda": 1041.14,
        "name": "Saccharomyces cerevisiae Target of Rapamycin Complex 2",
    },
    {
        "code": "4XK8",
        "category": "macromolecules",
        "mw_kda": 1046.38,
        "name": "photosystem I-LHCI super-complex",
    },
    {
        "code": "7EEP",
        "category": "macromolecules",
        "mw_kda": 1048.62,
        "name": "Cyanophage Pam1 portal-adaptor complex",
    },
    {
        "code": "7BKC",
        "category": "macromolecules",
        "mw_kda": 1070.10,
        "name": "Formate dehydrogenase - heterodisulfide reductase - "
        "formylmethanofuran dehydrogenase complex",
    },
    {
        "code": "6GY6",
        "category": "macromolecules",
        "mw_kda": 1117.80,
        "name": "XaxAB pore complex from Xenorhabdus nematophila",
    },
    {
        "code": "6Z6O",
        "category": "macromolecules",
        "mw_kda": 1144.83,
        "name": "HDAC-TC",
    },
    {
        "code": "7T3U",
        "category": "macromolecules",
        "mw_kda": 1200.59,
        "name": "IP3 ATP, and Ca2+ bound type 3 IP3 receptor in the inactive state",
    },
    {
        "code": "7MEI",
        "category": "macromolecules",
        "mw_kda": 1213.88,
        "name": "Composite structure of EC+EC, transcription",
    },
    {
        "code": "7QJ0",
        "category": "macromolecules",
        "mw_kda": 1224.45,
        "name": "recombinant human gamma-Tubulin Ring Complex 6-spoked "
        "assembly intermediate",
    },
    {
        "code": "5G04",
        "category": "macromolecules",
        "mw_kda": 1250.92,
        "name": "human APC-Cdc20-Hsl1 complex",
    },
    {
        "code": "7EGE",
        "category": "macromolecules",
        "mw_kda": 1281.07,
        "name": "TFIID in canonical conformation",
    },
    {
        "code": "7WOO",
        "category": "macromolecules",
        "mw_kda": 1286.20,
        "name": "inner ring protomer of the Saccharomyces cerevisiae "
        "nuclear pore complex",
    },
    {
        "code": "7SN7",
        "category": "macromolecules",
        "mw_kda": 1291.21,
        "name": "enteropathogenic E. coli O127:H6 flagellar filament",
    },
    {
        "code": "6GYM",
        "category": "macromolecules",
        "mw_kda": 1297.09,
        "name": "Structure of a yeast closed complex with distorted DNA - "
        "RNA polymerase II (Pol II) pre-initiation complex (PIC)",
    },
    {
        "code": "4CR2",
        "category": "macromolecules",
        "mw_kda": 1309.28,
        "name": "26S proteasome",
    },
    {
        "code": "7EGD",
        "category": "macromolecules",
        "mw_kda": 1379.49,
        "name": "SCP promoter-bound TFIID-TFIIA in initial TBP-loading state",
    },
    {
        "code": "6YT5",
        "category": "macromolecules",
        "mw_kda": 1410.42,
        "name": "T7 bacteriophage DNA translocation gp15-gp16 core complex "
        "intermediate assembly",
    },
    {
        "code": "6DUZ",
        "category": "macromolecules",
        "mw_kda": 1746.11,
        "name": "periplasmic domains of PrgH and PrgK from the assembled "
        "Salmonella type III secretion injectisome needle complex, "
        "membrane protein",
    },
    {
        "code": "6UP6",
        "category": "macromolecules",
        "mw_kda": 1797.10,
        "name": "Endophilin B1 helical scaffold",
    },
    {
        "code": "5OOL",
        "category": "macromolecules",
        "mw_kda": 1797.38,
        "name": "human mitochondrial ribosome with unfolded interfacial rRNA",
    },
    {
        "code": "6F8L",
        "category": "macromolecules",
        "mw_kda": 1818.38,
        "name": "Thermus thermophilus PilF ATPase",
    },
    {
        "code": "7EY7",
        "category": "macromolecules",
        "mw_kda": 2040.07,
        "name": "bacteriophage T7 tail complex",
    },
    {
        "code": "6ID1",
        "category": "macromolecules",
        "mw_kda": 2147.52,
        "name": "human intron lariat spliceosome after Prp43 loaded",
    },
    {
        "code": "6IGC",
        "category": "macromolecules",
        "mw_kda": 2365.70,
        "name": "HPV58/33/52 chimeric L1 pentamer",
    },
    {
        "code": "6X9Q",
        "category": "macromolecules",
        "mw_kda": 2796.28,
        "name": "transcription-translation complex B3 (TTC-B3), ribosome, "
        "rna polymerase",
    },
    {
        "code": "5MRC",
        "category": "macromolecules",
        "mw_kda": 3325.59,
        "name": "yeast mitochondrial ribosome",
    },
    {
        "code": "4UJD",
        "category": "macromolecules",
        "mw_kda": 4044.60,
        "name": "mammalian 80S ribosome",
    },
    # --- distractors (smaller cytosolic complexes) ---
    {
        "code": "1EXR",
        "category": "distractors",
        "mw_kda": 16.89,
        "name": "CA+2 bound calmodulin",
    },
    {
        "code": "1QTX",
        "category": "distractors",
        "mw_kda": 19.10,
        "name": "Calmodulin RS20 peptide complex",
    },
    {
        "code": "2Q0U",
        "category": "distractors",
        "mw_kda": 43.72,
        "name": "Pectenotoxin-2 and Latrunculin B Bound to Actin",
    },
    {
        "code": "12GS",
        "category": "distractors",
        "mw_kda": 48.01,
        "name": "Glutathione S-transferase complexed with S-nonyl-glutathione",
    },
    {
        "code": "1SJJ",
        "category": "distractors",
        "mw_kda": 199.75,
        "name": "Chicken Gizzard Smooth Muscle alpha-Actinin",
    },
    # --- transcription/translation ---
    {
        "code": "1MWS",
        "category": "transcription_translation",
        "mw_kda": 149.62,
        "name": "nitrocefin acyl-Penicillin binding protein 2a from "
        "methicillin resistant Staphylococcus aureus strain",
    },
    {
        "code": "6N60",
        "category": "transcription_translation",
        "mw_kda": 458.37,
        "name": "RNA polymerase sigma70-holoenzyme bound to upstream fork promoter DNA",
    },
    {
        "code": "6CA0",
        "category": "transcription_translation",
        "mw_kda": 491.43,
        "name": "E. coli RNAP sigma70 open complex",
    },
    {
        "code": "7OOC",
        "category": "transcription_translation",
        "mw_kda": 810.14,
        "name": "30S subunit of ribosomes in chloramphenicol-treated cells, Mycoplasma",
    },
    {
        "code": "1I94",
        "category": "transcription_translation",
        "mw_kda": 841.86,
        "name": "small ribosomal subunit with tetracycline, edeine and IF3",
    },
    {
        "code": "4YG2",
        "category": "transcription_translation",
        "mw_kda": 920.17,
        "name": "RNA polymerase sigma70 holoenzyme",
    },
    {
        "code": "6AWB",
        "category": "transcription_translation",
        "mw_kda": 1141.70,
        "name": "30S ribosomal subunit and RNA polymerase complex in non-rotated state",
    },
    {
        "code": "7PAT",
        "category": "transcription_translation",
        "mw_kda": 1411.99,
        "name": "free 50S in untreated Mycoplasma pneumoniae cells",
    },
    {
        "code": "7P6Z",
        "category": "transcription_translation",
        "mw_kda": 2250.56,
        "name": "70S ribosome in untreated cells, mycoplasma",
    },
    {
        "code": "4V4R",
        "category": "transcription_translation",
        "mw_kda": 2277.81,
        "name": "whole ribosomal complex",
    },
    {
        "code": "6ZTN",
        "category": "transcription_translation",
        "mw_kda": 2669.95,
        "name": "E. coli 70S-RNAP expressome complex in NusG-coupled state",
    },
    {
        "code": "4V5D",
        "category": "transcription_translation",
        "mw_kda": 4516.21,
        "name": "70S ribosome in complex with mRNA, paromomycin, acylated "
        "A- and P-site tRNAs, and E-site tRNA",
    },
    {
        "code": "4V5C",
        "category": "transcription_translation",
        "mw_kda": 4519.39,
        "name": "70S ribosome in complex with mRNA, paromomycin, acylated "
        "A-site tRNA, deacylated P-site tRNA, and E-site tRNA",
    },
    {
        "code": "4V5G",
        "category": "transcription_translation",
        "mw_kda": 4668.79,
        "name": "70S ribosome bound to EF-Tu and tRNA",
    },
    {
        "code": "4V5F",
        "category": "transcription_translation",
        "mw_kda": 4772.30,
        "name": "ribosome with elongation factor G trapped in the "
        "post-translocational state",
    },
    # --- nucleosomes ---
    {
        "code": "5F99",
        "category": "nucleosomes",
        "mw_kda": 199.29,
        "name": "MMTV-A Nucleosome Core Particle",
    },
    {
        "code": "7PEY",
        "category": "nucleosomes",
        "mw_kda": 219.68,
        "name": "Nucleosome 3 of the 4x177 nucleosome array containing H1",
    },
    {
        "code": "7PEW",
        "category": "nucleosomes",
        "mw_kda": 222.77,
        "name": "Nucleosome 1 of the 4x177 nucleosome array containing H1",
    },
    {
        "code": "7PEX",
        "category": "nucleosomes",
        "mw_kda": 245.19,
        "name": "Nucleosome 2 of the 4x177 nucleosome array containing H1",
    },
    {
        "code": "5OXV",
        "category": "nucleosomes",
        "mw_kda": 411.66,
        "name": "4_601_157 tetranucleosome",
    },
    {
        "code": "1ZBB",
        "category": "nucleosomes",
        "mw_kda": 431.87,
        "name": "4_601_167 Tetranucleosome",
    },
    {
        "code": "6M3V",
        "category": "nucleosomes",
        "mw_kda": 439.12,
        "name": "355 bp di-nucleosome harboring cohesive DNA termini",
    },
    {
        "code": "6LA8",
        "category": "nucleosomes",
        "mw_kda": 457.42,
        "name": "349 bp di-nucleosome harboring cohesive DNA termini "
        "assembled with linker histone H1.0",
    },
    {
        "code": "6L4A",
        "category": "nucleosomes",
        "mw_kda": 635.98,
        "name": "H3-H3-H3 tri-nucleosome with the 22 base-pair linker DNA",
    },
    {
        "code": "6L49",
        "category": "nucleosomes",
        "mw_kda": 637.16,
        "name": "H3-CA-H3 tri-nucleosome with the 22 base-pair linker DNA",
    },
    {
        "code": "7PEV",
        "category": "nucleosomes",
        "mw_kda": 661.93,
        "name": "4x177 nucleosome array containing H1",
    },
    {
        "code": "7PEU",
        "category": "nucleosomes",
        "mw_kda": 685.97,
        "name": "4x177 nucleosome array containing H1",
    },
    {
        "code": "7PF0",
        "category": "nucleosomes",
        "mw_kda": 720.12,
        "name": "Trinucleosome of the 4x177 nucleosome array containing H1",
    },
    {
        "code": "7VA4",
        "category": "nucleosomes",
        "mw_kda": 745.81,
        "name": "Telomeric tetranucleosome in open state",
    },
    {
        "code": "7V9K",
        "category": "nucleosomes",
        "mw_kda": 750.15,
        "name": "Telomeric tetranucleosome",
    },
    {
        "code": "7PFT",
        "category": "nucleosomes",
        "mw_kda": 772.82,
        "name": "Trinucleosome of the 4x207 nucleosome array containing H1",
    },
    {
        "code": "5OY7",
        "category": "nucleosomes",
        "mw_kda": 827.37,
        "name": "4_601_157 tetranucleosome",
    },
    {
        "code": "7PFA",
        "category": "nucleosomes",
        "mw_kda": 872.74,
        "name": "Trinucleosome of the 4x197 nucleosome array containing H1",
    },
    {
        "code": "7PET",
        "category": "nucleosomes",
        "mw_kda": 933.73,
        "name": "4x177 nucleosome array containing H1",
    },
]


def build_filler_pool_specs(
    table: list[dict],
    max_mw_kda: float | None = None,
    min_mw_kda: float | None = None,
    categories: list[str] | None = None,
    codes: list[str] | None = None,
    exclude_codes: list[str] | None = None,
) -> list[dict]:
    """
    Filter a filler reference table down to a mass range and/or category,
    and adapt it to `TomogramSpecimenGenerator`'s (`specter build
    tomogram`) flat ``{"pdb_source": ...}`` filler_specs shape -- one
    entry per selected species, weighted implicitly equally (via
    `TomogramProteinSpec.ratio`'s own default) unless a caller overrides
    that afterward.

    Works on any ``list[dict]`` with a ``"code"``/``"mw_kda"`` key per
    entry -- both `CRYOETSIM_PARTICLE_TABLE` and `PEI2016_CROWDING_TABLE`
    (this module) qualify, so the same helper adapts either (or both,
    called twice and concatenated) into one filler_specs list.
    `categories` only has an effect on tables that carry a ``"category"``
    key (`CRYOETSIM_PARTICLE_TABLE`); entries without one are kept
    regardless of this filter.

    Parameters
    ----------
    table : list of dict
        e.g. `CRYOETSIM_PARTICLE_TABLE` or `PEI2016_CROWDING_TABLE`.
    max_mw_kda : float, optional
        Only include entries at or below this mass, kDa.
    min_mw_kda : float, optional
        Only include entries at or above this mass, kDa.
    categories : list of str, optional
        Only include entries whose ``"category"`` is in this list
        (ignored for entries with no ``"category"`` key at all).
    codes : list of str, optional
        Only include these specific codes (must exist in `table`).
        Mutually exclusive with `exclude_codes`.
    exclude_codes : list of str, optional
        Drop these codes from `table`. Mutually exclusive with `codes`.

    Returns
    -------
    list[dict]
        One ``{"pdb_source": code}`` entry per selected species, ready to
        concatenate onto your own `filler_specs`/`[[filler]]` list.
    """
    if codes is not None and exclude_codes is not None:
        raise ValueError("pass only one of codes / exclude_codes, not both")

    selected = table
    if codes is not None:
        by_code = {e["code"]: e for e in selected}
        missing = [c for c in codes if c not in by_code]
        if missing:
            raise ValueError(f"not in table: {missing}")
        selected = [by_code[c] for c in codes]
    elif exclude_codes is not None:
        selected = [e for e in selected if e["code"] not in exclude_codes]

    if categories is not None:
        selected = [
            e for e in selected if "category" not in e or e["category"] in categories
        ]
    if max_mw_kda is not None:
        selected = [e for e in selected if e["mw_kda"] <= max_mw_kda]
    if min_mw_kda is not None:
        selected = [e for e in selected if e["mw_kda"] >= min_mw_kda]

    return [{"pdb_source": e["code"]} for e in selected]
