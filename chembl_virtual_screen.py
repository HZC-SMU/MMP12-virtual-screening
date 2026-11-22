#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from Bio import SeqIO
import gzip
import torch
import os
import tempfile
import shutil
from tqdm import tqdm
import numpy as np
from featurizer import featurizer, seq_cat
from tape import TAPETokenizer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import argparse


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
                data = pack(atoms, adjs, proteins, self.device)
                predicted_scores = self.model(data)
        return predicted_scores


def pack(atoms, adjs, proteins, device):
    atoms = torch.FloatTensor(atoms)
    adjs = torch.FloatTensor(adjs)
    proteins = torch.FloatTensor(proteins)
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
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
    atoms_new = torch.zeros((N, atoms_len, 34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, 1:a_len + 1, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        adjs_new[i, 0, :] = 1
        adjs_new[i, :, 0] = 1
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, 1:a_len + 1, 1:a_len + 1] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len), dtype=torch.int64, device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len] = protein
        i += 1
    return (atoms_new, adjs_new, proteins_new, atom_num, protein_num)


class ProteinFeaturizer:
    def __init__(self):
        self.tokenizer = TAPETokenizer(vocab='iupac')
        self.cached_protein_features = {}
    
    def featurize_protein(self, sequence):
        if sequence not in self.cached_protein_features:
            sequence_tokens = seq_cat(sequence, self.tokenizer)
            with torch.no_grad():
                protein_embedding = torch.tensor([sequence_tokens], dtype=torch.int64)
            self.cached_protein_features[sequence] = protein_embedding.squeeze(0).numpy()
        return self.cached_protein_features[sequence]


def read_sdf_molecules(sdf_path, max_molecules=None):
    molecules = []
    is_compressed = sdf_path.endswith('.gz')
    
    try:
        if is_compressed:
            
            print("Extracting the SDF file (this may take a while)...")
            with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp_file:
                with gzip.open(sdf_path, 'rb') as gz_file:
                    shutil.copyfileobj(gz_file, tmp_file)
                temp_sdf_path = tmp_file.name
            
            suppl = Chem.SDMolSupplier(temp_sdf_path)
        else:
            suppl = Chem.SDMolSupplier(sdf_path)
        
        count = 0
        processed = 0
        
        print("Reading molecules...")
        for mol in suppl:
            processed += 1
            if processed % 10000 == 0:
                print(f"Processed {processed} molecules, successfully read {count}")
            
            if mol is not None:
                try:
                    mol_id = mol.GetProp('_Name') if mol.HasProp('_Name') else f'CHEMBL_{count}'
                    smiles = Chem.MolToSmiles(mol)
                    
                    if smiles and len(smiles) > 0:
                        molecules.append({
                            'id': mol_id,
                            'smiles': smiles
                        })
                        count += 1
                        
                        if max_molecules and count >= max_molecules:
                            break
                            
                except Exception as e:
                    continue
        
        if is_compressed:
            os.unlink(temp_sdf_path)
        
        print(f"Processed {processed} molecules in total, {count} valid molecules read")
        
    except Exception as e:
        print(f"Error reading the SDF file: {str(e)}")
        return []
    
    return molecules


def mol_features_optimized(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        from featurizer import atom_features, adjacent_matrix, num_atom_feat
        atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
        for atom in mol.GetAtoms():
            atom_feat[atom.GetIdx(), :] = atom_features(atom)
        adj_matrix = adjacent_matrix(mol)
        return atom_feat, adj_matrix
    except Exception:
        return None, None


def get_processed_molecules(output_path):
    """Get the set of already-processed molecule IDs."""
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            if not df_existing.empty and 'id' in df_existing.columns:
                processed_ids = set(df_existing['id'].astype(str))
                print(f"Found {len(processed_ids)} processed molecules")
        except Exception as e:
            print(f"Error reading the existing results file: {e}")
    return processed_ids


def save_single_result(output_path, mol_id, smiles, score):
    """Save a single prediction result."""
    result_row = pd.DataFrame([{
        'id': mol_id,
        'smiles': smiles, 
        'score': score
    }])
    
    # Write the header if the file does not exist
    if not os.path.exists(output_path):
        result_row.to_csv(output_path, mode='w', index=False)
    else:
        result_row.to_csv(output_path, mode='a', index=False, header=False)


def chembl_virtual_screen(protein_fasta_path, sdf_path, output_path, 
                         ckpt_path='ckpt/Virtual Screening.pt',
                         batch_size=100, max_molecules=None, resume=True):
    
    print(f"Starting ChEMBL virtual screening...")
    print(f"Protein file: {protein_fasta_path}")
    print(f"Molecule file: {sdf_path}")
    print(f"Output file: {output_path}")
    print(f"Resume from checkpoint: {'enabled' if resume else 'disabled'}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Running on GPU...')
    else:
        device = torch.device('cpu')
        print('Running on CPU!!!')
    
    model = torch.load(ckpt_path)
    model.to(device)
    tester = Tester(model, device)
    
    protein_featurizer = ProteinFeaturizer()
    
    print("Reading protein sequence...")
    target_sequence = None
    for record in SeqIO.parse(protein_fasta_path, 'fasta'):
        target_sequence = str(record.seq)
        break
    
    if not target_sequence:
        raise ValueError("Could not read a protein sequence from the FASTA file")
    
    print(f"Protein sequence length: {len(target_sequence)}")
    print("Pre-featurizing the protein sequence...")
    cached_protein_feature = protein_featurizer.featurize_protein(target_sequence)
    print(f"Protein feature shape: {cached_protein_feature.shape}")
    
    # Check for already-processed molecules
    processed_molecules = set()
    if resume:
        processed_molecules = get_processed_molecules(output_path)
    
    print("Reading the molecule file...")
    molecules = read_sdf_molecules(sdf_path, max_molecules)
    total_molecules = len(molecules)
    print(f"Successfully read {total_molecules} molecules")
    
    # Filter out already-processed molecules
    if processed_molecules:
        molecules = [mol for mol in molecules if str(mol['id']) not in processed_molecules]
        print(f"Skipping processed molecules; {len(molecules)} remaining")
    
    if not molecules:
        print("All molecules have already been processed!")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    successful_predictions = 0
    failed_predictions = 0
    error_log_path = output_path.replace('.csv', '_errors.log')
    
    print("Starting predictions...")
    progress_bar = tqdm(molecules, desc="Prediction progress", unit="mol")
    
    with open(error_log_path, 'a', encoding='utf-8') as error_log:
        error_log.write(f"\n=== New screening session started: {pd.Timestamp.now()} ===\n")
        
        for i, mol_info in enumerate(progress_bar):
            mol_id = str(mol_info['id'])
            smiles = mol_info['smiles']
            
            try:
                atom_feature, adj_matrix = mol_features_optimized(smiles)
                
                if atom_feature is None or adj_matrix is None:
                    error_msg = f"Featurization failed: {mol_id} - {smiles}"
                    error_log.write(f"{pd.Timestamp.now()}: {error_msg}\n")
                    error_log.flush()
                    failed_predictions += 1
                    continue
                
                test_data = [(atom_feature, adj_matrix, cached_protein_feature)]
                score = float(tester.test(test_data))
                
                # Save the result immediately
                save_single_result(output_path, mol_id, smiles, score)
                successful_predictions += 1
                
                # Update the progress bar
                progress_bar.set_postfix({
                    'success': successful_predictions,
                    'failed': failed_predictions,
                    'current_score': f'{score:.4f}',
                    'total_progress': f'{successful_predictions + len(processed_molecules)}/{total_molecules}'
                })
                
            except Exception as e:
                error_msg = f"Prediction failed: {mol_id} - {smiles} - error: {str(e)}"
                error_log.write(f"{pd.Timestamp.now()}: {error_msg}\n")
                error_log.flush()
                failed_predictions += 1
                continue
    
    print(f"\nPrediction finished!")
    print(f"This session - success: {successful_predictions}, failed: {failed_predictions}")
    print(f"Total processed: {successful_predictions + len(processed_molecules)}/{total_molecules}")
    
    # Finalize the results file
    if os.path.exists(output_path):
        print("Finalizing and sorting the results file...")
        try:
            df_final = pd.read_csv(output_path)
            if not df_final.empty:
                # Deduplicate and sort
                df_final = df_final.drop_duplicates(subset=['id'], keep='last')
                df_final = df_final.sort_values(by='score', ascending=False)
                df_final.to_csv(output_path, index=False)
                
                print(f"Final results saved to: {output_path}")
                print(f"{len(df_final)} valid predictions in total")
                print(f"Top 10 highest-scoring compounds:")
                print(df_final.head(10)[['id', 'score']].to_string(index=False))
                
                return df_final
        except Exception as e:
            print(f"Error finalizing the results file: {e}")
    
    print("No valid results were generated!")
    return None


def main():
    parser = argparse.ArgumentParser(description='ChEMBL virtual screening tool - supports resuming from a checkpoint')
    parser.add_argument('--protein', default='data/diguazhiwang_xy/TAF4B.fasta',
                       help='Path to the protein FASTA file')
    parser.add_argument('--molecules', default='/data/mydata/Mols/chembl_35.sdf.gz',
                       help='Path to the molecule SDF file')
    parser.add_argument('--output', default='data/chembl_screening_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--checkpoint', default='ckpt/Virtual Screening.pt',
                       help='Model checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size (used for progress display)')
    parser.add_argument('--max_molecules', type=int, default=None,
                       help='Maximum number of molecules to process (for testing)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Disable checkpoint resuming and start from scratch')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.protein):
        print(f"Error: protein file does not exist - {args.protein}")
        return
    
    if not os.path.exists(args.molecules):
        print(f"Error: molecule file does not exist - {args.molecules}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: model checkpoint file does not exist - {args.checkpoint}")
        return
    
    # Check whether the existing results should be cleared
    if args.no_resume and os.path.exists(args.output):
        confirm = input(f"This will delete the existing results file {args.output}. Confirm? (y/N): ")
        if confirm.lower() in ['y', 'yes']:
            os.remove(args.output)
            error_log = args.output.replace('.csv', '_errors.log')
            if os.path.exists(error_log):
                os.remove(error_log)
            print("Existing results file cleared")
        else:
            print("Operation cancelled")
            return
    
    chembl_virtual_screen(
        protein_fasta_path=args.protein,
        sdf_path=args.molecules,
        output_path=args.output,
        ckpt_path=args.checkpoint,
        batch_size=args.batch_size,
        max_molecules=args.max_molecules,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()