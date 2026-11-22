"""
Step 1: Evaluate the model in the transformerCPI environment and save predictions
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
import json
import pickle
from finetune import CPIDataset, collate_fn, load_data_from_csv
from model import pack

def evaluate_model(model, data_loader, device):
    """Evaluate model performance."""
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for compounds, adjacencies, proteins, labels in data_loader:
            data = pack(compounds, adjacencies, proteins, device)
            outputs = model.forward(*data)

            # Get predicted probabilities and predicted classes
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)

def main():
    """Main entry point."""
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Using GPU...')
    else:
        device = torch.device('cpu')
        print('Using CPU...')

    # Load data
    print("\nLoading data...")
    data_path = 'data/dataset.csv'
    smiles_list, sequence_list, labels = load_data_from_csv(
        data_path, 'smiles', 'sequence', 'label'
    )

    # Split the data using the same split as training
    train_smiles, val_smiles, train_seq, val_seq, train_labels, val_labels = train_test_split(
        smiles_list, sequence_list, labels,
        test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train set: {len(train_smiles)}, Validation set: {len(val_smiles)}")

    # Create datasets
    train_dataset = CPIDataset(train_smiles, train_seq, train_labels)
    val_dataset = CPIDataset(val_smiles, val_seq, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )

    # Load the best model
    print("\nLoading best model...")
    model_path = 'finetune_ckpt/best_model_full.pt'
    model = torch.load(model_path, map_location=device)
    model.to(device)

    # Evaluate on the training set
    print("Evaluating on training set...")
    train_labels_np, train_probs, train_preds = evaluate_model(model, train_loader, device)

    # Evaluate on the validation set
    print("Evaluating on validation set...")
    val_labels_np, val_probs, val_preds = evaluate_model(model, val_loader, device)

    # Save prediction results
    results = {
        'train_labels': train_labels_np,
        'train_probs': train_probs,
        'train_preds': train_preds,
        'val_labels': val_labels_np,
        'val_probs': val_probs,
        'val_preds': val_preds
    }

    os.makedirs('image', exist_ok=True)
    save_path = 'image/prediction_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nPrediction results saved to: {save_path}")
    print(f"Train set size: {len(train_labels_np)}")
    print(f"Validation set size: {len(val_labels_np)}")
    print("\nNext step: Run plot_results.py in gnn environment to generate plots")

if __name__ == "__main__":
    main()
