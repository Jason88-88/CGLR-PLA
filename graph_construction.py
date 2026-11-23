"""CGLR graph construction for protein-ligand complexes.

Chemistry-Guided Local Rewiring (CGLR) builds one unified graph per complex:

* pocket atoms (PDB, heavy atoms only, water removed, altlocs resolved) and
  ligand atoms (SDF, explicit H kept) as nodes with 108-D atom features;
* covalent pocket / ligand bonds as edges;
* ligand-pocket interaction edges chosen per ligand heavy atom:
    1. collect pocket heavy atoms within INTERACTION_CUTOFF (5.0 A);
    2. compute five chemical flags for every candidate pair;
    3. the chemically compatible candidates (priority > 0), ranked by
       (priority desc, distance), give the reference set of at most
       INTERACTION_K atoms;
    4. k_i = min(INTERACTION_K, N_compatible);
    5. the k_i spatially nearest atoms of the full candidate set, ranked by
       (distance, pocket index), are connected;
    6. edges are added in both directions;
    7. each unique contact is stored for contact attention.
* 24-D edge features: 16-D Gaussian RBF distance + 5 chemical flags +
  3-D edge type (pocket bond / ligand bond / interaction).

Graphs are written as ``pd_graphs_<split>_part<N>.pt`` (lists of PyG ``Data``)
with ``pd_graphs_<split>_meta.csv``, ``pd_graphs_<split>_failed.csv`` and
``PDBbind_<split>_success_index.parquet``.

Usage:
    python graph_construction.py --data-root /path/to/structures \
        --output-dir /path/to/graphs \
        --split train /path/to/train_index.parquet \
        --split val /path/to/val_index.parquet \
        --split CASF_2016_core /path/to/casf2016_index.parquet

Each index parquet needs ``PDB_code`` and ``log_Kd_Ki`` columns; structures
are read from ``<data-root>/<split>/<PDB>/<PDB>_pocket.pdb`` and
``<PDB>_ligand.sdf``.
"""

import argparse
import gc
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy.spatial import cKDTree
from torch_geometric.data import Data


RDLogger.DisableLog("rdApp.*")


# Structure preprocessing
REMOVE_POCKET_H = True
REMOVE_WATER = True
RESOLVE_ALTLOC = True
REMOVE_LIGAND_H = False

# CGLR interaction edges
INTERACTION_CUTOFF = 5.0
INTERACTION_K = 6

# Chemical interaction rules
HBOND_CUTOFF = 3.8
HYDROPHOBIC_CUTOFF = 4.5
SALT_BRIDGE_CUTOFF = 5.0
AROMATIC_CUTOFF = 5.0
VDW_TOLERANCE = 0.5
EXCLUDE_H_IN_INTERACTION_EDGES = True

INTERACTION_FLAG_NAMES = [
    "hbond",
    "hydrophobic",
    "salt_bridge",
    "aromatic_candidate",
    "vdw_contact",
]

# Distance RBF (16 centers on [0, 5] A, sigma 0.5 A)
RBF_NUM_CENTERS = 16
RBF_MU_MIN = 0.0
RBF_MU_MAX = 5.0
RBF_SIGMA = 0.5

# Size limits
MAX_POCKET_ATOMS = 6000
MAX_LIGAND_ATOMS = 500
MAX_TOTAL_DIRECTED_EDGES = 800000

# Legacy triplet fields are stored empty (shape [0] / [0, 52]) so that newly
# built graphs keep the schema of the existing graph files. They are not used
# by the model.
LEGACY_TRIPLET_ATTR_DIM = 52

# Runtime defaults (overridable on the command line)
BATCH_SIZE = 100
NUM_WORKERS = 4
SAMPLE_TIMEOUT_SEC = 300
MP_START_METHOD = "spawn"



ALLOW_ATOMS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "I", "B",
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
]

CHIRAL_TYPES = [
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_OTHER,
]

HYBRID_TYPES = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]

WATER_RESNAMES = {"HOH", "WAT", "DOD", "SOL"}

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",
}


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def normalize_element(elem):
    elem = str(elem).strip()
    if not elem:
        return ""

    elem_upper = elem.replace(" ", "").upper()

    two_char_map = {
        "CL": "Cl",
        "BR": "Br",
        "SI": "Si",
        "NA": "Na",
        "MG": "Mg",
        "AL": "Al",
        "CA": "Ca",
        "SC": "Sc",
        "TI": "Ti",
        "CR": "Cr",
        "MN": "Mn",
        "FE": "Fe",
        "CO": "Co",
        "NI": "Ni",
        "CU": "Cu",
        "ZN": "Zn",
        "GA": "Ga",
        "GE": "Ge",
        "AS": "As",
        "SE": "Se",
        "KR": "Kr",
        "RB": "Rb",
        "SR": "Sr",
        "ZR": "Zr",
        "NB": "Nb",
        "MO": "Mo",
        "TC": "Tc",
        "RU": "Ru",
        "RH": "Rh",
    }

    if len(elem_upper) >= 2 and elem_upper[:2] in two_char_map:
        return two_char_map[elem_upper[:2]]

    return elem_upper[0]


def infer_element_from_atom_name(atom_name, record, resname):
    name = atom_name.strip().upper()
    if not name:
        return ""

    name = re.sub(r"^[0-9]+", "", name)


    if record == "ATOM" and resname in STANDARD_AA:
        if name.startswith("H"):
            return "H"
        if name.startswith("C"):
            return "C"
        if name.startswith("N"):
            return "N"
        if name.startswith("O"):
            return "O"
        if name.startswith("S"):
            return "S"

    for two in [
        "CL", "BR", "SI", "NA", "MG", "AL", "CA", "SC", "TI",
        "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE",
        "AS", "SE", "KR", "RB", "SR", "ZR", "NB", "MO", "TC",
        "RU", "RH",
    ]:
        if name.startswith(two):
            return two

    return name[0] if name else ""


def parse_pdb_atom_line(line):
    raw = line.rstrip("\n")
    padded = raw.ljust(80)

    record = padded[0:6].strip()
    atom_name = padded[12:16]
    altloc = padded[16].strip()
    resname = padded[17:20].strip().upper()
    chain = padded[21].strip()
    resseq = padded[22:26].strip()
    icode = padded[26].strip()

    x = safe_float(padded[30:38].strip())
    y = safe_float(padded[38:46].strip())
    z = safe_float(padded[46:54].strip())

    occupancy = safe_float(padded[54:60].strip(), default=1.0)
    bfactor = safe_float(padded[60:66].strip(), default=0.0)

    elem_col = padded[76:78].strip()
    element = normalize_element(elem_col)

    if not element:
        element = normalize_element(
            infer_element_from_atom_name(atom_name, record, resname)
        )

    return {
        "raw": raw,
        "record": record,
        "atom_name": atom_name.strip(),
        "atom_name_raw": atom_name,
        "altloc": altloc,
        "resname": resname,
        "chain": chain,
        "resseq": resseq,
        "icode": icode,
        "x": x,
        "y": y,
        "z": z,
        "occupancy": occupancy,
        "bfactor": bfactor,
        "element": element,
    }


def choose_altloc_atoms(atoms):
    grouped = defaultdict(list)

    for atom in atoms:
        key = (
            atom["record"],
            atom["chain"],
            atom["resseq"],
            atom["icode"],
            atom["resname"],
            atom["atom_name"],
        )
        grouped[key].append(atom)

    selected = []

    for _, group in grouped.items():
        if len(group) == 1:
            selected.append(group[0])
            continue

        def rank(a):
            occ = a["occupancy"] if a["occupancy"] is not None else 0.0
            alt = a["altloc"]

            if alt == "":
                alt_rank = 3
            elif alt == "A":
                alt_rank = 2
            elif alt == "B":
                alt_rank = 1
            else:
                alt_rank = 0

            return occ, alt_rank

        selected.append(sorted(group, key=rank, reverse=True)[0])

    return selected


def format_pdb_line(atom, serial):


    record = atom["record"]
    atom_name = atom["atom_name_raw"]
    resname = atom["resname"]
    chain = atom["chain"] if atom["chain"] else " "
    resseq = atom["resseq"]
    icode = atom["icode"] if atom["icode"] else " "

    x, y, z = atom["x"], atom["y"], atom["z"]
    occ = atom["occupancy"] if atom["occupancy"] is not None else 1.0
    bfac = atom["bfactor"] if atom["bfactor"] is not None else 0.0
    element = atom["element"]

    if len(atom_name) < 4:

        atom_name_fmt = f"{atom_name:>4s}"
    else:
        atom_name_fmt = atom_name[:4]

    elem_fmt = f"{element:>2s}"[:2]

    line = (
        f"{record:<6s}"
        f"{serial:5d} "
        f"{atom_name_fmt}"
        f" "
        f"{resname:>3s} "
        f"{chain:1s}"
        f"{resseq:>4s}"
        f"{icode:1s}"
        f"   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occ:6.2f}{bfac:6.2f}"
        f"          "
        f"{elem_fmt:>2s}"
    )

    return line.rstrip() + "\n"


def preprocess_pocket_pdb_to_block(pocket_path):


    pocket_path = Path(pocket_path)

    if not pocket_path.exists():
        return None, "missing_pocket_file"

    atoms = []
    raw_atom_count = 0
    removed_water = 0
    removed_h = 0
    removed_bad_coord = 0

    try:
        with open(pocket_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue

                raw_atom_count += 1
                atom = parse_pdb_atom_line(line)

                if REMOVE_WATER and atom["resname"] in WATER_RESNAMES:
                    removed_water += 1
                    continue

                if atom["x"] is None or atom["y"] is None or atom["z"] is None:
                    removed_bad_coord += 1
                    continue

                if not (
                    math.isfinite(atom["x"])
                    and math.isfinite(atom["y"])
                    and math.isfinite(atom["z"])
                ):
                    removed_bad_coord += 1
                    continue

                if REMOVE_POCKET_H and atom["element"] == "H":
                    removed_h += 1
                    continue

                atoms.append(atom)

        if len(atoms) == 0:
            return None, (
                f"empty_after_preprocess: raw={raw_atom_count}, "
                f"water_removed={removed_water}, h_removed={removed_h}, "
                f"bad_coord_removed={removed_bad_coord}"
            )

        if RESOLVE_ALTLOC:
            before = len(atoms)
            atoms = choose_altloc_atoms(atoms)
            alt_removed = before - len(atoms)
        else:
            alt_removed = 0

        if len(atoms) > MAX_POCKET_ATOMS:
            return None, f"too_many_pocket_atoms_after_preprocess: {len(atoms)}"

        lines = []
        for i, atom in enumerate(atoms, start=1):
            lines.append(format_pdb_line(atom, i))
        lines.append("END\n")

        pdb_block = "".join(lines)

        preprocess_info = {
            "raw_atom_count": raw_atom_count,
            "preprocessed_atom_count": len(atoms),
            "removed_water": removed_water,
            "removed_h": removed_h,
            "removed_bad_coord": removed_bad_coord,
            "removed_altloc": alt_removed,
        }

        return pdb_block, preprocess_info

    except Exception as e:
        return None, f"preprocess_exception: {repr(e)}"


def safe_call(default, func):
    try:
        return func()
    except Exception:
        return default


def get_atom_features(atom, force_no_ring=False):


    symbol = safe_call("", atom.GetSymbol)


    atom_type_enc = [int(symbol == x) for x in ALLOW_ATOMS]

    implicit_val = safe_call(-999, atom.GetImplicitValence)
    implicit_val_enc = [int(implicit_val == x) for x in range(11)]

    chiral_tag = safe_call(None, atom.GetChiralTag)
    chiral_enc = [int(chiral_tag == x) for x in CHIRAL_TYPES]
    chiral_enc += [int(chiral_tag not in CHIRAL_TYPES)]

    degree = safe_call(-999, atom.GetDegree)
    degree_enc = [int(degree == x) for x in range(11)]

    formal_charge = safe_call(-999, atom.GetFormalCharge)
    formal_charge_enc = [int(formal_charge == x) for x in range(-5, 7)]

    total_h = safe_call(-999, atom.GetTotalNumHs)
    total_h_enc = [int(total_h == x) for x in range(11)]

    radical = safe_call(-999, atom.GetNumRadicalElectrons)
    radical_enc = [int(radical == x) for x in range(6)]

    hybrid_type = safe_call(None, atom.GetHybridization)
    hybrid_enc = [int(hybrid_type == x) for x in HYBRID_TYPES]
    hybrid_enc += [int(hybrid_type not in HYBRID_TYPES)]

    is_aromatic = safe_call(False, atom.GetIsAromatic)
    is_aromatic_enc = [int(is_aromatic)]

    if force_no_ring:
        is_in_ring = False
    else:
        is_in_ring = safe_call(False, atom.IsInRing)

    is_in_ring_enc = [int(is_in_ring)]

    features = (
        atom_type_enc
        + implicit_val_enc
        + chiral_enc
        + degree_enc
        + formal_charge_enc
        + total_h_enc
        + radical_enc
        + hybrid_enc
        + is_aromatic_enc
        + is_in_ring_enc
    )

    if len(features) != 108:
        raise ValueError(f"atom feature dimension error: {len(features)}")

    return features


def rbf_expansion(distances):
    centers = np.linspace(RBF_MU_MIN, RBF_MU_MAX, RBF_NUM_CENTERS, dtype=np.float32)
    distances = distances.astype(np.float32)
    distances = np.expand_dims(distances, axis=-1)
    rbf = np.exp(-((distances - centers) ** 2) / (2 * (RBF_SIGMA ** 2)))
    return rbf.astype(np.float32)

def make_empty_triplet_arrays():
    """Empty legacy triplet arrays (kept for graph-schema compatibility)."""
    triplet_attr_dim = LEGACY_TRIPLET_ATTR_DIM

    return (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0, triplet_attr_dim), dtype=np.float32),
    )

def get_coords(mol, name):
    if mol.GetNumConformers() == 0:
        return None, f"{name}_no_conformer"

    try:
        coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)

        if coords.ndim != 2 or coords.shape[1] != 3:
            return None, f"{name}_bad_coord_shape"

        if not np.isfinite(coords).all():
            return None, f"{name}_non_finite_coords"

        return coords, None

    except Exception as e:
        return None, f"{name}_coord_exception: {repr(e)}"


AROMATIC_RES_RING_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HID": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIE": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIP": {"CG", "ND1", "CD2", "CE1", "NE2"},
}

NEGATIVE_RES_ATOMS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

POSITIVE_RES_ATOMS = {
    "LYS": {"NZ"},
    "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"},
    "HID": {"ND1", "NE2"},
    "HIE": {"ND1", "NE2"},
    "HIP": {"ND1", "NE2"},
}


def get_pdb_res_info(atom):


    try:
        info = atom.GetPDBResidueInfo()
    except Exception:
        info = None

    if info is None:
        return "", ""

    try:
        resname = info.GetResidueName().strip().upper()
    except Exception:
        resname = ""

    try:
        atom_name = info.GetName().strip().upper()
    except Exception:
        atom_name = ""

    return resname, atom_name


def atom_symbol(atom):
    return safe_call("", atom.GetSymbol)


def atom_formal_charge(atom):
    return int(safe_call(0, atom.GetFormalCharge))


def protein_residue_charge_sign(atom):


    resname, atom_name = get_pdb_res_info(atom)

    if resname in NEGATIVE_RES_ATOMS and atom_name in NEGATIVE_RES_ATOMS[resname]:
        return -1

    if resname in POSITIVE_RES_ATOMS and atom_name in POSITIVE_RES_ATOMS[resname]:
        return 1

    fc = atom_formal_charge(atom)
    if fc > 0:
        return 1
    if fc < 0:
        return -1

    return 0


def ligand_charge_sign(atom):


    fc = atom_formal_charge(atom)

    if fc > 0:
        return 1
    if fc < 0:
        return -1

    return 0


def is_potential_donor(atom, is_pocket=False):


    sym = atom_symbol(atom)

    if sym not in {"N", "O", "S"}:
        return False

    if is_pocket:
        charge = protein_residue_charge_sign(atom)
    else:
        charge = ligand_charge_sign(atom)


    if charge < 0:
        return False

    return True


def is_potential_acceptor(atom, is_pocket=False):


    sym = atom_symbol(atom)

    if sym not in {"N", "O", "S"}:
        return False

    if is_pocket:
        charge = protein_residue_charge_sign(atom)
    else:
        charge = ligand_charge_sign(atom)


    if charge > 0:
        return False

    return True


def is_hydrophobic_atom(atom, is_pocket=False):


    sym = atom_symbol(atom)

    if sym != "C":
        return False

    if is_pocket:
        charge = protein_residue_charge_sign(atom)
    else:
        charge = ligand_charge_sign(atom)

    if charge != 0:
        return False

    return True


def is_aromatic_atom(atom, is_pocket=False):


    if not is_pocket:
        return bool(safe_call(False, atom.GetIsAromatic))

    resname, atom_name = get_pdb_res_info(atom)

    if resname in AROMATIC_RES_RING_ATOMS:
        return atom_name in AROMATIC_RES_RING_ATOMS[resname]

    return False


PTABLE = Chem.GetPeriodicTable()


def get_vdw_radius(atom):


    sym = atom_symbol(atom)
    atomic_num = int(safe_call(0, atom.GetAtomicNum))

    try:
        r = float(PTABLE.GetRvdw(atomic_num))
        if math.isfinite(r) and r > 0:
            return r
    except Exception:
        pass

    fallback = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "F": 1.47,
        "P": 1.80,
        "S": 1.80,
        "Cl": 1.75,
        "Br": 1.85,
        "I": 1.98,
    }

    return fallback.get(sym, 1.70)


def is_vdw_contact(lig_atom, pocket_atom, distance):


    lig_sym = atom_symbol(lig_atom)
    poc_sym = atom_symbol(pocket_atom)

    if EXCLUDE_H_IN_INTERACTION_EDGES and (lig_sym == "H" or poc_sym == "H"):
        return False

    r_lig = get_vdw_radius(lig_atom)
    r_poc = get_vdw_radius(pocket_atom)

    return float(distance) <= (r_lig + r_poc + VDW_TOLERANCE)


def classify_ligand_pocket_pair(lig_atom, pocket_atom, distance):


    flags = np.zeros(len(INTERACTION_FLAG_NAMES), dtype=np.float32)

    if EXCLUDE_H_IN_INTERACTION_EDGES:
        if atom_symbol(lig_atom) == "H" or atom_symbol(pocket_atom) == "H":
            return flags


    lig_donor = is_potential_donor(lig_atom, is_pocket=False)
    lig_acceptor = is_potential_acceptor(lig_atom, is_pocket=False)
    poc_donor = is_potential_donor(pocket_atom, is_pocket=True)
    poc_acceptor = is_potential_acceptor(pocket_atom, is_pocket=True)

    if distance <= HBOND_CUTOFF:
        if (lig_donor and poc_acceptor) or (poc_donor and lig_acceptor):
            flags[0] = 1.0


    if distance <= HYDROPHOBIC_CUTOFF:
        if is_hydrophobic_atom(lig_atom, is_pocket=False) and is_hydrophobic_atom(
            pocket_atom,
            is_pocket=True,
        ):
            flags[1] = 1.0


    lig_charge = ligand_charge_sign(lig_atom)
    poc_charge = protein_residue_charge_sign(pocket_atom)

    if distance <= SALT_BRIDGE_CUTOFF:
        if lig_charge * poc_charge < 0:
            flags[2] = 1.0


    if distance <= AROMATIC_CUTOFF:
        if is_aromatic_atom(lig_atom, is_pocket=False) and is_aromatic_atom(
            pocket_atom,
            is_pocket=True,
        ):
            flags[3] = 1.0


    strong = flags[:4].sum() > 0
    if (not strong) and is_vdw_contact(lig_atom, pocket_atom, distance):
        flags[4] = 1.0

    return flags

def interaction_priority(flags):


    hbond = flags[0] > 0
    hydrophobic = flags[1] > 0
    salt_bridge = flags[2] > 0
    aromatic_candidate = flags[3] > 0
    vdw_contact = flags[4] > 0

    if salt_bridge:
        return 5
    if hbond:
        return 4
    if aromatic_candidate:
        return 3
    if hydrophobic:
        return 2
    if vdw_contact:
        return 1

    return 0


def worker_build_graph(args):


    pocket_path, ligand_path, affinity, pdb_code = args

    try:


        pocket_block, info = preprocess_pocket_pdb_to_block(pocket_path)
        if pocket_block is None:
            return None, pdb_code, str(info)


        pocket_mol = Chem.MolFromPDBBlock(
            pocket_block,
            sanitize=False,
            removeHs=False,
            proximityBonding=True,
        )

        if pocket_mol is None:
            return None, pdb_code, "rdkit_parse_pocket_failed"

        try:
            pocket_mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass

        n_pocket = pocket_mol.GetNumAtoms()

        if n_pocket == 0:
            return None, pdb_code, "zero_pocket_atoms"

        if n_pocket > MAX_POCKET_ATOMS:
            return None, pdb_code, f"too_many_pocket_atoms: {n_pocket}"


        if not os.path.exists(ligand_path):
            return None, pdb_code, "missing_ligand_file"

        ligand_mol = Chem.MolFromMolFile(
            str(ligand_path),
            sanitize=False,
            removeHs=False,
            strictParsing=False,
        )

        if ligand_mol is None:
            return None, pdb_code, "rdkit_parse_ligand_failed"

        if REMOVE_LIGAND_H:
            try:
                ligand_mol = Chem.RemoveHs(ligand_mol, sanitize=False)
            except Exception:
                pass

        try:
            ligand_mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass


        try:
            Chem.GetSymmSSSR(ligand_mol)
        except Exception:
            pass

        n_ligand = ligand_mol.GetNumAtoms()

        if n_ligand == 0:
            return None, pdb_code, "zero_ligand_atoms"

        if n_ligand > MAX_LIGAND_ATOMS:
            return None, pdb_code, f"too_many_ligand_atoms: {n_ligand}"


        pocket_coords, err = get_coords(pocket_mol, "pocket")
        if err is not None:
            return None, pdb_code, err

        ligand_coords, err = get_coords(ligand_mol, "ligand")
        if err is not None:
            return None, pdb_code, err


        pocket_features = [
            get_atom_features(atom, force_no_ring=True)
            for atom in pocket_mol.GetAtoms()
        ]

        ligand_features = [
            get_atom_features(atom, force_no_ring=False)
            for atom in ligand_mol.GetAtoms()
        ]

        x_np = np.asarray(pocket_features + ligand_features, dtype=np.float32)


        src_nodes = []
        dst_nodes = []
        edge_types = []
        edge_distances = []
        edge_flags = []
        contact_pocket_nodes = []
        contact_ligand_nodes = []
        contact_forward_edge_ids = []
        contact_flags = []

        zero_flags = np.zeros(len(INTERACTION_FLAG_NAMES), dtype=np.float32)

        def add_directed_edge(src, dst, edge_type, distance, flags=None):
            src_nodes.append(int(src))
            dst_nodes.append(int(dst))
            edge_types.append(int(edge_type))
            edge_distances.append(float(distance))

            if flags is None:
                edge_flags.append(zero_flags.copy())
            else:
                edge_flags.append(np.asarray(flags, dtype=np.float32))


        for bond in pocket_mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()

            if u >= n_pocket or v >= n_pocket:
                continue

            d = float(np.linalg.norm(pocket_coords[u] - pocket_coords[v]))

            add_directed_edge(u, v, 0, d)
            add_directed_edge(v, u, 0, d)


        for bond in ligand_mol.GetBonds():
            u0 = bond.GetBeginAtomIdx()
            v0 = bond.GetEndAtomIdx()

            if u0 >= n_ligand or v0 >= n_ligand:
                continue

            u = u0 + n_pocket
            v = v0 + n_pocket

            d = float(np.linalg.norm(ligand_coords[u0] - ligand_coords[v0]))

            add_directed_edge(u, v, 1, d)
            add_directed_edge(v, u, 1, d)


        pocket_tree = cKDTree(pocket_coords)

        directed_interaction_edges = 0
        undirected_interaction_edges = 0

        hbond_edge_count = 0
        hydrophobic_edge_count = 0
        salt_bridge_edge_count = 0
        aromatic_edge_count = 0
        vdw_edge_count = 0
        unclassified_local_edge_count = 0

        reference_chemical_contact_count = 0
        nearest_reference_overlap_count = 0
        reference_union_count = 0

        for l_idx in range(n_ligand):
            lig_coord = ligand_coords[l_idx]

            pocket_candidate_ids = pocket_tree.query_ball_point(
                lig_coord,
                r=INTERACTION_CUTOFF,
            )

            lig_atom = ligand_mol.GetAtomWithIdx(l_idx)
            all_candidates = []

            for p_idx_raw in pocket_candidate_ids:
                p_idx = int(p_idx_raw)
                pocket_atom = pocket_mol.GetAtomWithIdx(p_idx)
                d = float(np.linalg.norm(pocket_coords[p_idx] - lig_coord))

                flags = classify_ligand_pocket_pair(
                    lig_atom=lig_atom,
                    pocket_atom=pocket_atom,
                    distance=d,
                )
                priority = interaction_priority(flags)

                all_candidates.append({
                    "p_idx": p_idx,
                    "l_idx": int(l_idx),
                    "distance": d,
                    "flags": np.asarray(flags, dtype=np.float32),
                    "priority": int(priority),
                })

            if not all_candidates:
                continue


            chemical_candidates = [
                item for item in all_candidates
                if item["priority"] > 0
            ]
            chemical_candidates.sort(
                key=lambda item: (
                    -item["priority"],
                    item["distance"],
                )
            )
            reference_selected = chemical_candidates[:INTERACTION_K]
            matched_k = len(reference_selected)

            if matched_k == 0:
                continue


            nearest_candidates = sorted(
                all_candidates,
                key=lambda item: (
                    item["distance"],
                    item["p_idx"],
                ),
            )
            selected = nearest_candidates[:matched_k]

            if len(selected) != matched_k:
                return None, pdb_code, (
                    f"nearest_selection_count_mismatch:"
                    f" ligand={l_idx}, expected={matched_k}, got={len(selected)}"
                )

            reference_pairs = {
                (item["p_idx"], item["l_idx"])
                for item in reference_selected
            }
            selected_pairs = {
                (item["p_idx"], item["l_idx"])
                for item in selected
            }

            reference_chemical_contact_count += len(reference_pairs)
            nearest_reference_overlap_count += len(
                reference_pairs & selected_pairs
            )
            reference_union_count += len(
                reference_pairs | selected_pairs
            )

            for item in selected:
                p_idx = item["p_idx"]
                l_idx = item["l_idx"]
                d = item["distance"]
                flags = item["flags"]

                u = p_idx
                v = l_idx + n_pocket


                forward_edge_id = len(src_nodes)

                add_directed_edge(u, v, 2, d, flags)
                add_directed_edge(v, u, 2, d, flags)

                contact_pocket_nodes.append(int(u))
                contact_ligand_nodes.append(int(v))
                contact_forward_edge_ids.append(int(forward_edge_id))
                contact_flags.append(np.asarray(flags, dtype=np.float32))

                directed_interaction_edges += 2
                undirected_interaction_edges += 1

                hbond_edge_count += int(flags[0] > 0)
                hydrophobic_edge_count += int(flags[1] > 0)
                salt_bridge_edge_count += int(flags[2] > 0)
                aromatic_edge_count += int(flags[3] > 0)
                vdw_edge_count += int(flags[4] > 0)
                unclassified_local_edge_count += int(flags.sum() == 0)


        if undirected_interaction_edges != reference_chemical_contact_count:
            return None, pdb_code, (
                "matched_contact_count_mismatch:"
                f" nearest={undirected_interaction_edges},"
                f" chemical_reference={reference_chemical_contact_count}"
            )

        if len(src_nodes) == 0:
            return None, pdb_code, "zero_edges"

        if len(src_nodes) > MAX_TOTAL_DIRECTED_EDGES:
            return None, pdb_code, f"too_many_total_edges: {len(src_nodes)}"


        edge_index_np = np.vstack([
            np.asarray(src_nodes, dtype=np.int64),
            np.asarray(dst_nodes, dtype=np.int64),
        ])

        edge_distances_np = np.asarray(edge_distances, dtype=np.float32)
        edge_types_np = np.asarray(edge_types, dtype=np.int64)
        edge_flags_np = np.asarray(edge_flags, dtype=np.float32)

        rbf_feats = rbf_expansion(edge_distances_np)

        edge_type_onehot = np.zeros((len(edge_types_np), 3), dtype=np.float32)
        edge_type_onehot[np.arange(len(edge_types_np)), edge_types_np] = 1.0


        edge_attr_np = np.concatenate(
            [
                rbf_feats,
                edge_flags_np,
                edge_type_onehot,
            ],
            axis=1,
        ).astype(np.float32)


        pos_np = np.concatenate(
            [
                pocket_coords,
                ligand_coords,
            ],
            axis=0,
        ).astype(np.float32)


        contact_pocket_np = np.asarray(contact_pocket_nodes, dtype=np.int64)
        contact_ligand_np = np.asarray(contact_ligand_nodes, dtype=np.int64)
        contact_forward_edge_ids_np = np.asarray(contact_forward_edge_ids, dtype=np.int64)

        if len(contact_flags) > 0:
            contact_flags_np = np.asarray(contact_flags, dtype=np.float32)
        else:
            contact_flags_np = np.zeros(
                (0, len(INTERACTION_FLAG_NAMES)),
                dtype=np.float32,
            )

        if len(contact_forward_edge_ids_np) > 0:
            contact_edge_attr_np = edge_attr_np[contact_forward_edge_ids_np].astype(np.float32)
        else:
            contact_edge_attr_np = np.zeros(
                (0, edge_attr_np.shape[1]),
                dtype=np.float32,
            )


        (
            triplet_edge_id_np,
            triplet_outer_np,
            triplet_center_np,
            triplet_other_np,
            triplet_side_np,
            triplet_attr_np,
        ) = make_empty_triplet_arrays()
        return {
            "x_np": x_np.astype(np.float32),
            "pos_np": pos_np.astype(np.float32),
            "edge_index_np": edge_index_np.astype(np.int64),
            "edge_attr_np": edge_attr_np.astype(np.float32),

            "y": float(affinity),

            "pdb_code": str(pdb_code),
            "n_pocket": int(n_pocket),
            "n_drug": int(n_ligand),
            "num_nodes_total": int(n_pocket + n_ligand),
            "num_edges_total": int(edge_index_np.shape[1]),

            "num_pocket_bond_edges": int(sum(1 for t in edge_types if t == 0)),
            "num_ligand_bond_edges": int(sum(1 for t in edge_types if t == 1)),
            "num_interaction_edges": int(directed_interaction_edges),

            "num_interaction_undirected_edges": int(undirected_interaction_edges),
            "num_hbond_edges": int(hbond_edge_count),
            "num_hydrophobic_edges": int(hydrophobic_edge_count),
            "num_salt_bridge_edges": int(salt_bridge_edge_count),
            "num_ionic_edges": int(salt_bridge_edge_count),
            "num_aromatic_edges": int(aromatic_edge_count),
            "num_vdw_edges": int(vdw_edge_count),
            "num_proximity_edges": 0,
            "num_unclassified_local_edges": int(unclassified_local_edge_count),

            "num_reference_chemical_contacts": int(
                reference_chemical_contact_count
            ),
            "num_nearest_reference_overlap": int(
                nearest_reference_overlap_count
            ),
            "pair_jaccard_with_reference_chemical": float(
                nearest_reference_overlap_count / reference_union_count
            ) if reference_union_count > 0 else 1.0,
            "graph_variant": "nearest_chemical_matched",

            "interaction_k": int(INTERACTION_K),
            "interaction_cutoff": float(INTERACTION_CUTOFF),
            "edge_attr_dim": int(edge_attr_np.shape[1]),

            "raw_pocket_atom_count": int(info.get("raw_atom_count", -1)) if isinstance(info, dict) else -1,
            "preprocessed_pocket_atom_count": int(info.get("preprocessed_atom_count", -1)) if isinstance(info, dict) else -1,
            "removed_water": int(info.get("removed_water", 0)) if isinstance(info, dict) else 0,
            "removed_h": int(info.get("removed_h", 0)) if isinstance(info, dict) else 0,
            "removed_altloc": int(info.get("removed_altloc", 0)) if isinstance(info, dict) else 0,
            "contact_pocket_np": contact_pocket_np.astype(np.int64),
            "contact_ligand_np": contact_ligand_np.astype(np.int64),
            "contact_edge_attr_np": contact_edge_attr_np.astype(np.float32),
            "contact_flags_np": contact_flags_np.astype(np.float32),

            "triplet_edge_id_np": triplet_edge_id_np.astype(np.int64),
            "triplet_outer_np": triplet_outer_np.astype(np.int64),
            "triplet_center_np": triplet_center_np.astype(np.int64),
            "triplet_other_np": triplet_other_np.astype(np.int64),
            "triplet_side_np": triplet_side_np.astype(np.int64),
            "triplet_attr_np": triplet_attr_np.astype(np.float32),

            "num_contacts": int(contact_pocket_np.shape[0]),
            "num_triplets": int(triplet_edge_id_np.shape[0]),
            "triplet_attr_dim": int(triplet_attr_np.shape[1]),
        }, pdb_code, None

    except Exception as e:
        return None, pdb_code, f"worker_exception: {repr(e)}"


def worker_loop(worker_id, task_queue, result_queue):
    while True:
        try:
            task = task_queue.get(timeout=1.0)
        except queue.Empty:
            break

        pdb_code = str(task[3])

        result_queue.put({
            "event": "start",
            "worker_id": worker_id,
            "pdb_code": pdb_code,
            "time": time.time(),
        })

        graph_data, p_code, err = worker_build_graph(task)

        result_queue.put({
            "event": "result",
            "worker_id": worker_id,
            "pdb_code": p_code,
            "graph": graph_data,
            "err": err,
            "time": time.time(),
        })

    result_queue.put({
        "event": "done",
        "worker_id": worker_id,
        "time": time.time(),
    })


def start_worker(ctx, worker_id, task_queue, result_queue):
    p = ctx.Process(
        target=worker_loop,
        args=(worker_id, task_queue, result_queue),
    )
    p.daemon = True
    p.start()
    return p


def payload_to_pyg_data(payload):


    x = torch.from_numpy(payload["x_np"]).float()
    pos = torch.from_numpy(payload["pos_np"]).float()
    edge_index = torch.from_numpy(payload["edge_index_np"]).long()
    edge_attr = torch.from_numpy(payload["edge_attr_np"]).float()
    contact_pocket = torch.from_numpy(payload["contact_pocket_np"]).long()
    contact_ligand = torch.from_numpy(payload["contact_ligand_np"]).long()
    contact_edge_attr = torch.from_numpy(payload["contact_edge_attr_np"]).float()
    contact_flags = torch.from_numpy(payload["contact_flags_np"]).float()

    triplet_edge_id = torch.from_numpy(payload["triplet_edge_id_np"]).long()
    triplet_outer = torch.from_numpy(payload["triplet_outer_np"]).long()
    triplet_center = torch.from_numpy(payload["triplet_center_np"]).long()
    triplet_other = torch.from_numpy(payload["triplet_other_np"]).long()
    triplet_side = torch.from_numpy(payload["triplet_side_np"]).long()
    triplet_attr = torch.from_numpy(payload["triplet_attr_np"]).float()
    y = torch.tensor([float(payload["y"])], dtype=torch.float32)

    graph_data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,

        contact_pocket=contact_pocket,
        contact_ligand=contact_ligand,
        contact_edge_attr=contact_edge_attr,
        contact_flags=contact_flags,

        triplet_edge_id=triplet_edge_id,
        triplet_outer=triplet_outer,
        triplet_center=triplet_center,
        triplet_other=triplet_other,
        triplet_side=triplet_side,
        triplet_attr=triplet_attr,
    )

    graph_data.pdb_code = str(payload["pdb_code"])
    graph_data.n_pocket = int(payload["n_pocket"])
    graph_data.n_drug = int(payload["n_drug"])
    graph_data.num_nodes_total = int(payload["num_nodes_total"])
    graph_data.num_edges_total = int(payload["num_edges_total"])
    graph_data.num_pocket_bond_edges = int(payload["num_pocket_bond_edges"])
    graph_data.num_ligand_bond_edges = int(payload["num_ligand_bond_edges"])
    graph_data.num_interaction_edges = int(payload["num_interaction_edges"])

    graph_data.num_interaction_undirected_edges = int(
        payload.get("num_interaction_undirected_edges", -1)
    )
    graph_data.num_hbond_edges = int(payload.get("num_hbond_edges", 0))
    graph_data.num_hydrophobic_edges = int(payload.get("num_hydrophobic_edges", 0))
    graph_data.num_salt_bridge_edges = int(payload.get("num_salt_bridge_edges", payload.get("num_ionic_edges", 0)))
    graph_data.num_ionic_edges = int(payload.get("num_ionic_edges", graph_data.num_salt_bridge_edges))
    graph_data.num_aromatic_edges = int(payload.get("num_aromatic_edges", 0))
    graph_data.num_vdw_edges = int(payload.get("num_vdw_edges", 0))
    graph_data.num_proximity_edges = int(payload.get("num_proximity_edges", 0))
    graph_data.num_unclassified_local_edges = int(
        payload.get("num_unclassified_local_edges", 0)
    )
    graph_data.num_reference_chemical_contacts = int(
        payload.get(
            "num_reference_chemical_contacts",
            payload.get("num_contacts", 0),
        )
    )
    graph_data.num_nearest_reference_overlap = int(
        payload.get("num_nearest_reference_overlap", 0)
    )
    graph_data.pair_jaccard_with_reference_chemical = float(
        payload.get("pair_jaccard_with_reference_chemical", float("nan"))
    )
    graph_data.graph_variant = str(
        payload.get("graph_variant", "nearest_chemical_matched")
    )
    graph_data.interaction_k = int(payload.get("interaction_k", -1))
    graph_data.interaction_cutoff = float(payload.get("interaction_cutoff", -1.0))
    graph_data.edge_attr_dim = int(payload.get("edge_attr_dim", -1))
    graph_data.num_contacts = int(payload.get("num_contacts", 0))
    graph_data.num_triplets = int(payload.get("num_triplets", 0))
    graph_data.triplet_attr_dim = int(payload.get("triplet_attr_dim", -1))
    graph_data.raw_pocket_atom_count = int(payload.get("raw_pocket_atom_count", -1))
    graph_data.preprocessed_pocket_atom_count = int(
        payload.get("preprocessed_pocket_atom_count", -1)
    )
    graph_data.removed_water = int(payload.get("removed_water", 0))
    graph_data.removed_h = int(payload.get("removed_h", 0))
    graph_data.removed_altloc = int(payload.get("removed_altloc", 0))

    return graph_data

def flush_graph_batch(graphs, task_name, batch_counter, output_dir):
    if not graphs:
        return 0, []

    batch_path = Path(output_dir) / f"pd_graphs_{task_name}_part{batch_counter}.pt"

    total_nodes = sum(int(g.x.size(0)) for g in graphs)
    total_edges = sum(int(g.edge_index.size(1)) for g in graphs)

    print(
        f"\nSaving part {batch_counter}: "
        f"graphs={len(graphs)}, nodes={total_nodes}, edges={total_edges}"
    )
    print(f"   -> {batch_path}")

    torch.save(graphs, batch_path)

    meta_rows = []
    for local_idx, g in enumerate(graphs):
        meta_rows.append({
            "task_name": task_name,
            "part": batch_counter,
            "index_in_part": local_idx,
            "pt_file": batch_path.name,
            "PDB_code": str(g.pdb_code),
            "log_Kd_Ki": float(g.y.item()),
            "n_pocket": int(g.n_pocket),
            "n_drug": int(g.n_drug),
            "num_nodes_total": int(g.num_nodes_total),
            "num_edges_total": int(g.num_edges_total),
            "num_pocket_bond_edges": int(g.num_pocket_bond_edges),
            "num_ligand_bond_edges": int(g.num_ligand_bond_edges),
            "num_interaction_edges": int(g.num_interaction_edges),
                        "num_interaction_undirected_edges": int(
                getattr(g, "num_interaction_undirected_edges", -1)
            ),
            "num_contacts": int(getattr(g, "num_contacts", 0)),
            "num_triplets": int(getattr(g, "num_triplets", 0)),
            "triplet_attr_dim": int(getattr(g, "triplet_attr_dim", -1)),
            "num_hbond_edges": int(getattr(g, "num_hbond_edges", 0)),
            "num_hydrophobic_edges": int(getattr(g, "num_hydrophobic_edges", 0)),
            "num_salt_bridge_edges": int(getattr(g, "num_salt_bridge_edges", getattr(g, "num_ionic_edges", 0))),
            "num_ionic_edges": int(getattr(g, "num_ionic_edges", 0)),
            "num_aromatic_edges": int(getattr(g, "num_aromatic_edges", 0)),
            "num_vdw_edges": int(getattr(g, "num_vdw_edges", 0)),
            "num_proximity_edges": int(getattr(g, "num_proximity_edges", 0)),
            "num_unclassified_local_edges": int(
                getattr(g, "num_unclassified_local_edges", 0)
            ),
            "num_reference_chemical_contacts": int(
                getattr(g, "num_reference_chemical_contacts", -1)
            ),
            "num_nearest_reference_overlap": int(
                getattr(g, "num_nearest_reference_overlap", 0)
            ),
            "pair_jaccard_with_reference_chemical": float(
                getattr(g, "pair_jaccard_with_reference_chemical", float("nan"))
            ),
            "graph_variant": str(
                getattr(g, "graph_variant", "nearest_chemical_matched")
            ),
            "interaction_k": int(getattr(g, "interaction_k", -1)),
            "interaction_cutoff": float(getattr(g, "interaction_cutoff", -1.0)),
            "edge_attr_dim": int(getattr(g, "edge_attr_dim", -1)),
            "raw_pocket_atom_count": int(getattr(g, "raw_pocket_atom_count", -1)),
            "preprocessed_pocket_atom_count": int(
                getattr(g, "preprocessed_pocket_atom_count", -1)
            ),
            "removed_water": int(getattr(g, "removed_water", 0)),
            "removed_h": int(getattr(g, "removed_h", 0)),
            "removed_altloc": int(getattr(g, "removed_altloc", 0)),
        })

    return len(graphs), meta_rows


def build_tasks_for_dataset(task_name, parquet_path, data_root):
    df = pd.read_parquet(parquet_path)

    if "PDB_code" not in df.columns:
        raise ValueError(f"{parquet_path} missing PDB_code column")

    if "log_Kd_Ki" not in df.columns:
        raise ValueError(f"{parquet_path} missing log_Kd_Ki column")

    df["PDB_code"] = df["PDB_code"].astype(str).str.strip()

    # Structures of split <task_name> are stored under <data_root>/<task_name>.
    folder_sub = task_name

    tasks = []
    for _, row in df.iterrows():
        pdb_code = str(row["PDB_code"]).strip()
        affinity = float(row["log_Kd_Ki"])

        sub_dir = Path(data_root) / folder_sub / pdb_code

        pocket_file = sub_dir / f"{pdb_code}_pocket.pdb"
        ligand_file = sub_dir / f"{pdb_code}_ligand.sdf"

        tasks.append((str(pocket_file), str(ligand_file), affinity, pdb_code))

    return df, tasks


def _print_progress(processed_count, total_tasks, n_success, n_failed, start_time):
    elapsed = time.time() - start_time
    sys.stdout.write(
        f"\rProgress: {processed_count}/{total_tasks} | "
        f"success: {n_success} | "
        f"failed: {n_failed} | "
        f"elapsed: {elapsed / 60:.1f} min"
    )
    sys.stdout.flush()


def process_dataset(
    task_name,
    parquet_path,
    ctx,
    data_root,
    output_dir,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    sample_timeout_sec=SAMPLE_TIMEOUT_SEC,
):
    output_dir = Path(output_dir)

    print("\n" + "=" * 80)
    print(f"Split: {task_name}")
    print("=" * 80)

    if not os.path.exists(parquet_path):
        print(f"Index parquet not found, skipping: {parquet_path}")
        return

    df_used, tasks = build_tasks_for_dataset(task_name, parquet_path, data_root)

    total_tasks = len(tasks)
    print(f"Samples to process: {total_tasks}")

    task_queue = ctx.Queue()
    result_queue = ctx.Queue(maxsize=num_workers * 2)

    for task in tasks:
        task_queue.put(task)

    workers = {}
    worker_states = {}

    for wid in range(num_workers):
        workers[wid] = start_worker(ctx, wid, task_queue, result_queue)
        worker_states[wid] = {
            "pdb_code": None,
            "start_time": None,
            "alive": True,
        }

    current_batch_graphs = []
    batch_counter = 0
    total_saved_count = 0
    processed_count = 0

    success_codes = set()
    failed_records = []
    meta_rows_all = []

    start_time = time.time()

    try:
        while processed_count < total_tasks:

            try:
                msg = result_queue.get(timeout=1.0)
            except queue.Empty:
                msg = None

            if msg is not None:
                event = msg.get("event")
                wid = msg.get("worker_id")

                if event == "start":
                    worker_states[wid]["pdb_code"] = str(msg["pdb_code"])
                    worker_states[wid]["start_time"] = float(msg["time"])
                    worker_states[wid]["alive"] = True

                elif event == "result":
                    graph_data = msg.get("graph")
                    pdb_code = str(msg.get("pdb_code"))
                    err = msg.get("err")

                    worker_states[wid]["pdb_code"] = None
                    worker_states[wid]["start_time"] = None

                    processed_count += 1

                    if graph_data is not None:
                        try:
                            graph_data = payload_to_pyg_data(graph_data)
                            current_batch_graphs.append(graph_data)
                            success_codes.add(pdb_code)
                        except Exception as e:
                            failed_records.append({
                                "dataset": task_name,
                                "PDB_code": pdb_code,
                                "error": f"payload_to_pyg_data_failed: {repr(e)}",
                            })
                    else:
                        failed_records.append({
                            "dataset": task_name,
                            "PDB_code": pdb_code,
                            "error": str(err),
                        })

                    if len(current_batch_graphs) >= batch_size:
                        saved_num, meta_rows = flush_graph_batch(
                            current_batch_graphs,
                            task_name,
                            batch_counter,
                            output_dir,
                        )

                        total_saved_count += saved_num
                        meta_rows_all.extend(meta_rows)

                        current_batch_graphs.clear()
                        gc.collect()
                        batch_counter += 1

                    if processed_count % 100 == 0 or processed_count == total_tasks:
                        _print_progress(
                            processed_count,
                            total_tasks,
                            total_saved_count + len(current_batch_graphs),
                            len(failed_records),
                            start_time,
                        )

                elif event == "done":
                    worker_states[wid]["alive"] = False
                    worker_states[wid]["pdb_code"] = None
                    worker_states[wid]["start_time"] = None

            now = time.time()

            for wid in range(num_workers):
                state = worker_states[wid]
                pdb_code = state.get("pdb_code")
                st = state.get("start_time")

                if pdb_code is None or st is None:
                    continue

                if now - st > sample_timeout_sec:
                    print(
                        f"\nSample timeout, skipped: "
                        f"worker={wid}, PDB={pdb_code}, "
                        f"time={now - st:.1f}s"
                    )

                    proc = workers.get(wid)
                    if proc is not None and proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=5)

                    failed_records.append({
                        "dataset": task_name,
                        "PDB_code": pdb_code,
                        "error": f"timeout>{sample_timeout_sec}s",
                    })

                    processed_count += 1

                    workers[wid] = start_worker(ctx, wid, task_queue, result_queue)
                    worker_states[wid] = {
                        "pdb_code": None,
                        "start_time": None,
                        "alive": True,
                    }

                    _print_progress(
                        processed_count,
                        total_tasks,
                        total_saved_count + len(current_batch_graphs),
                        len(failed_records),
                        start_time,
                    )

    except KeyboardInterrupt:
        print("\nInterrupted, saving completed graphs...")

    except Exception as e:
        print("\nMain process exception:")
        print(repr(e))
        traceback.print_exc()

    finally:
        for wid, proc in workers.items():
            if proc is not None and proc.is_alive():
                proc.terminate()
            if proc is not None:
                proc.join(timeout=5)

    if current_batch_graphs:
        saved_num, meta_rows = flush_graph_batch(
            current_batch_graphs,
            task_name,
            batch_counter,
            output_dir,
        )

        total_saved_count += saved_num
        meta_rows_all.extend(meta_rows)

        current_batch_graphs.clear()
        gc.collect()

    meta_path = output_dir / f"pd_graphs_{task_name}_meta.csv"
    if meta_rows_all:
        pd.DataFrame(meta_rows_all).to_csv(meta_path, index=False)
        print(f"\nMetadata saved: {meta_path}")
    else:
        print("\nNo graph was built; metadata not saved.")

    failed_path = output_dir / f"pd_graphs_{task_name}_failed.csv"
    if failed_records:
        pd.DataFrame(failed_records).to_csv(failed_path, index=False)
        print(f"Failures saved: {failed_path}")
    else:
        print("No failed samples.")

    df_success = df_used[
        df_used["PDB_code"].astype(str).str.strip().isin(success_codes)
    ].copy()

    success_parquet_path = output_dir / f"PDBbind_{task_name}_success_index.parquet"
    df_success.to_parquet(success_parquet_path, index=False)

    elapsed = time.time() - start_time

    print(
        f"\n{task_name} done | "
        f"graphs: {total_saved_count} | "
        f"failed/skipped: {len(failed_records)} | "
        f"elapsed: {elapsed / 60:.2f} min"
    )
    print(f"Success index: {success_parquet_path} ({len(df_success)} rows)")


def existing_outputs(output_dir, task_name):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    patterns = [
        f"pd_graphs_{task_name}_part*.pt",
        f"pd_graphs_{task_name}_meta.csv",
        f"pd_graphs_{task_name}_failed.csv",
        f"PDBbind_{task_name}_success_index.parquet",
    ]
    return sorted(str(p) for pattern in patterns for p in output_dir.glob(pattern))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build CGLR complex graphs.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root holding <split>/<PDB>/<PDB>_pocket.pdb and <PDB>_ligand.sdf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", nargs=2, action="append", required=True,
                        metavar=("NAME", "INDEX_PARQUET"),
                        help="Split name and index parquet (PDB_code, log_Kd_Ki); repeatable")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--part-size", type=int, default=BATCH_SIZE,
                        help="Graphs per pd_graphs_<split>_part<N>.pt file")
    parser.add_argument("--timeout", type=int, default=SAMPLE_TIMEOUT_SEC,
                        help="Per-sample timeout in seconds")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an output dir that already holds files for a split")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.overwrite:
        clashes = [p for name, _ in args.split for p in existing_outputs(args.output_dir, name)]
        if clashes:
            sys.exit(
                "Refusing to overwrite existing graph files (use --overwrite):\n  "
                + "\n  ".join(clashes[:20])
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("====== CGLR graph construction ======")
    print(f"data_root = {args.data_root}")
    print(f"output_dir = {args.output_dir}")
    print(f"num_workers = {args.num_workers}")
    print(f"sample_timeout_sec = {args.timeout}")
    print(f"part_size = {args.part_size}")
    print(f"REMOVE_POCKET_H = {REMOVE_POCKET_H}")
    print(f"REMOVE_WATER = {REMOVE_WATER}")
    print(f"RESOLVE_ALTLOC = {RESOLVE_ALTLOC}")
    print(f"REMOVE_LIGAND_H = {REMOVE_LIGAND_H}")
    print(f"MP_START_METHOD = {MP_START_METHOD}")
    print(f"INTERACTION_CUTOFF = {INTERACTION_CUTOFF}")
    print(f"INTERACTION_K = {INTERACTION_K}")
    print(f"VDW_TOLERANCE = {VDW_TOLERANCE}")
    print(f"EXCLUDE_H_IN_INTERACTION_EDGES = {EXCLUDE_H_IN_INTERACTION_EDGES}")
    print(f"INTERACTION_FLAG_NAMES = {INTERACTION_FLAG_NAMES}")

    try:
        ctx = mp.get_context(MP_START_METHOD)
    except ValueError:
        print(f"Start method {MP_START_METHOD} unavailable, using spawn")
        ctx = mp.get_context("spawn")

    for task_name, parquet_path in args.split:
        process_dataset(
            task_name,
            parquet_path,
            ctx,
            data_root=args.data_root,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            batch_size=args.part_size,
            sample_timeout_sec=args.timeout,
        )

    print("\n====== Graph construction finished ======")


if __name__ == "__main__":
    main()
