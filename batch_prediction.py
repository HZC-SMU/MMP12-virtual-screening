import argparse
import os
from typing import List, Optional

import pandas as pd
from Bio import SeqIO
import torch
from rdkit import Chem
from tqdm import tqdm

from featurizer import featurizer


class Tester(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins = [], [], []
                atom, adj, protein = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                packed = pack(atoms, adjs, proteins, self.device)
                predicted_scores = self.model(packed)
        return predicted_scores


def pack(atoms, adjs, proteins, device):
    # Cannot convert directly to tensor because each sample has different dimensions
    atoms_len = 0
    proteins_len = 0
    n_samples = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0] + 1)
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    atoms_len += 1
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((n_samples, atoms_len, 34), device=device)
    for idx, atom in enumerate(atoms):
        a_len = atom.shape[0]
        atoms_new[idx, 1 : a_len + 1, :] = torch.FloatTensor(atom)
    adjs_new = torch.zeros((n_samples, atoms_len, atoms_len), device=device)
    for idx, adj in enumerate(adjs):
        adjs_new[idx, 0, :] = 1
        adjs_new[idx, :, 0] = 1
        a_len = adj.shape[0]
        adj_tensor = torch.FloatTensor(adj) + torch.eye(a_len)
        adjs_new[idx, 1 : a_len + 1, 1 : a_len + 1] = adj_tensor
    proteins_new = torch.zeros((n_samples, proteins_len), dtype=torch.int64, device=device)
    for idx, protein in enumerate(proteins):
        a_len = protein.shape[0]
        proteins_new[idx, :a_len] = torch.LongTensor(protein)
    return atoms_new, adjs_new, proteins_new, atom_num, protein_num


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TCMSP_CANDIDATE_PATHS = [
    "/data/mydata/Mols/tcmsp.csv"  # built-in TCMSP path
]


def resolve_compound_path(compound_input: str) -> str:
    if compound_input.lower() != "tcmsp":
        return compound_input
    for candidate in TCMSP_CANDIDATE_PATHS:
        if os.path.exists(candidate):
            print(f"Using built-in TCMSP database: {candidate}")
            return candidate
    raise FileNotFoundError("Could not find the built-in tcmsp.csv database; please check whether the file exists.")


def canonicalise_column(columns: List[str], target: Optional[str]) -> Optional[str]:
    if target is None:
        return None
    column_map = {col.lower(): col for col in columns}
    lowered = target.lower()
    if lowered in column_map:
        return column_map[lowered]
    if target in columns:
        return target
    return None


def prompt_for_column(columns: List[str], column_type: str) -> str:
    print(f"Available {column_type} columns:")
    for idx, col in enumerate(columns):
        print(f"[{idx}] {col}")
    while True:
        choice = input(f"Select the {column_type} column (enter column name or index): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(columns):
                return columns[idx]
        if choice in columns:
            return choice
        lowered_choice = choice.lower()
        matches = [col for col in columns if col.lower() == lowered_choice]
        if matches:
            return matches[0]
        print("Invalid input, please try again.")


def auto_select_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        col = canonicalise_column(columns, candidate)
        if col:
            return col
    return None


def load_molecules_from_csv(path: str, smiles_col: Optional[str], id_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The CSV file is empty; cannot run predictions.")
    columns = list(df.columns)

    smiles_col = canonicalise_column(columns, smiles_col)
    if smiles_col is None:
        smiles_col = auto_select_column(columns, ["smiles", "canonical_smiles", "SMILES"])
    if smiles_col is None:
        smiles_col = prompt_for_column(columns, "SMILES")

    id_col = canonicalise_column(columns, id_col)
    if id_col is None:
        id_col = auto_select_column(columns, ["id", "name", "compound_id", "cas number", "cas_number"])
    if id_col is None:
        id_col = prompt_for_column(columns, "ID")

    subset = df[[id_col, smiles_col]].rename(columns={id_col: "id", smiles_col: "smiles"})
    subset = subset.dropna(subset=["smiles"])
    subset = subset[subset["smiles"].astype(str).str.strip() != ""]
    subset = subset.drop_duplicates(subset=["id", "smiles"], keep="first")
    if subset.empty:
        raise ValueError("No valid molecule records were found in the CSV.")
    return subset.reset_index(drop=True)


def collect_sdf_properties(molecules: List[Chem.Mol]) -> List[str]:
    props = set()
    for mol in molecules:
        if mol is None:
            continue
        props.update(mol.GetPropNames())
    return sorted(props)


def prompt_for_sdf_property(props: List[str]) -> str:
    if not props:
        raise ValueError("No usable molecule properties were found in the SDF file.")
    print("Available ID properties:")
    for idx, prop in enumerate(props):
        print(f"[{idx}] {prop}")
    while True:
        choice = input("Select the ID property (enter property name or index): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(props):
                return props[idx]
        if choice in props:
            return choice
        lowered_choice = choice.lower()
        matches = [prop for prop in props if prop.lower() == lowered_choice]
        if matches:
            return matches[0]
        print("Invalid input, please try again.")


def load_molecules_from_sdf(path: str, id_property: Optional[str]) -> pd.DataFrame:
    supplier = Chem.SDMolSupplier(path)
    molecules = [mol for mol in supplier if mol is not None]
    if not molecules:
        raise ValueError("No valid molecules could be parsed from the SDF file.")

    props = collect_sdf_properties(molecules)
    if id_property:
        matches = [prop for prop in props if prop.lower() == id_property.lower()]
        id_property = matches[0] if matches else id_property
    if not id_property or id_property not in props:
        id_property = prompt_for_sdf_property(props)

    records = []
    missing_id = 0
    for mol in molecules:
        if not mol.HasProp(id_property):
            missing_id += 1
            continue
        mol_id = mol.GetProp(id_property)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        records.append({"id": mol_id, "smiles": smiles})

    if not records:
        raise ValueError("No molecules containing the selected ID property were found in the SDF file.")

    df = pd.DataFrame(records)
    df = df.dropna(subset=["smiles"])
    df = df[df["smiles"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["id", "smiles"], keep="first")
    df = df.reset_index(drop=True)

    if missing_id:
        print(f"Warning: {missing_id} molecules are missing property {id_property}; skipped.")
    return df


def load_molecules(compound_input: str, smiles_col: Optional[str], id_col: Optional[str], id_property: Optional[str]) -> pd.DataFrame:
    resolved_path = resolve_compound_path(compound_input)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Compound file not found: {resolved_path}")

    ext = os.path.splitext(resolved_path)[1].lower()
    if ext == ".csv":
        return load_molecules_from_csv(resolved_path, smiles_col, id_col)
    if ext == ".sdf":
        return load_molecules_from_sdf(resolved_path, id_property)
    raise ValueError("Only CSV or SDF molecule files are supported.")


def load_protein_sequence(sequence: Optional[str], fasta_path: Optional[str]) -> str:
    if sequence:
        return sequence.strip()
    if not fasta_path:
        raise ValueError("A protein sequence or FASTA file path must be provided.")
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]
    if not sequences:
        raise ValueError("No protein sequence could be read from the FASTA file.")
    if len(sequences) > 1:
        print(f"Warning: the FASTA file contains {len(sequences)} sequences; using the first one by default.")
    return sequences[0]


def compound_predict(seq: str, saved_path: str, molecules: pd.DataFrame, ckpt: str) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU...")
    else:
        device = torch.device("cpu")
        print("Running on CPU...")

    print(f"Loading model: {ckpt}")
    model = torch.load(ckpt, map_location=device)
    model.to(device)
    tester = Tester(model, device)

    results = []
    failed_records = []

    for _, row in tqdm(molecules.iterrows(), total=len(molecules), desc="Prediction progress"):
        mol_id = row["id"]
        smiles = row["smiles"]
        try:
            compounds, adjacencies, proteins = featurizer(smiles, seq)
            test_set = list(zip(compounds, adjacencies, proteins))
            score = float(tester.test(test_set))
            results.append({"id": mol_id, "smiles": smiles, "score": score})
        except Exception as exc:  # Catch invalid molecules and keep processing the rest
            failed_records.append((mol_id, str(exc)))

    if failed_records:
        print("The following molecules failed:")
        for mol_id, reason in failed_records:
            print(f"- {mol_id}: {reason}")

    if not results:
        raise RuntimeError("All molecule predictions failed.")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="score", ascending=False)
    result_df.to_csv(saved_path, index=False)
    print(f"Results saved to: {saved_path}")
    print("Top 10 molecules:")
    print(result_df.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch prediction of compound-protein binding scores")
    parser.add_argument("--compound-input", default=os.path.join("data", "xy01", "mols_compound_info.csv"), help="Path to the molecule input file; supports CSV, SDF, or the keyword tcmsp")
    parser.add_argument("--tcmsp", action="store_true", help="Use the built-in TCMSP molecule database for prediction")
    parser.add_argument("--smiles-column", help="SMILES column name when the input is a CSV.")
    parser.add_argument("--id-column", help="ID column name when the input is a CSV.")
    parser.add_argument("--id-property", help="Property to use as the ID when the input is an SDF.")
    parser.add_argument("--protein-seq", help="Provide the protein sequence directly.")
    parser.add_argument("--protein-fasta", default=os.path.join("data", "xy01", "protein.fasta"), help="Path to the protein FASTA file.")
    parser.add_argument("--output", default=os.path.join("data", "xy01", "compound_prediction_results.csv"), help="Output path for prediction results.")
    parser.add_argument("--ckpt", default=os.path.join("ckpt", "Virtual Screening.pt"), help="Path to the model checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        compound_input_arg = "tcmsp" if args.tcmsp else args.compound_input
        molecules = load_molecules(
            compound_input=compound_input_arg,
            smiles_col=args.smiles_column,
            id_col=args.id_column,
            id_property=args.id_property,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to load molecules: {exc}")

    try:
        protein_sequence = load_protein_sequence(args.protein_seq, args.protein_fasta)
    except Exception as exc:
        raise SystemExit(f"Failed to load protein sequence: {exc}")

    print(f"A total of {len(molecules)} molecules will be used for prediction.")
    print(f"Protein sequence length: {len(protein_sequence)}")

    compound_predict(seq=protein_sequence, saved_path=args.output, molecules=molecules, ckpt=args.ckpt)


if __name__ == "__main__":
    main()
