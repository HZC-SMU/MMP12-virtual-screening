"""
Model evaluation script: generates ROC comparison plots and a training-results table
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, roc_auc_score, accuracy_score,
                            precision_score, recall_score, f1_score,
                            confusion_matrix, precision_recall_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
import json
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

def calculate_metrics(labels, probs, preds):
    """Compute evaluation metrics."""
    # Basic metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    # AUC and AUPR
    try:
        roc_auc = roc_auc_score(labels, probs)
    except:
        roc_auc = 0.5

    try:
        precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
        aupr = auc(recall_vals, precision_vals)
    except:
        aupr = 0.5

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1-Score': f1,
        'AUC': roc_auc,
        'AUPR': aupr,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }

def plot_roc_comparison(train_labels, train_probs, val_labels, val_probs, save_path):
    """Plot ROC curves for training and validation sets, compared with a random classifier."""
    plt.figure(figsize=(10, 8))

    # Compute the training-set ROC curve
    fpr_train, tpr_train, _ = roc_curve(train_labels, train_probs)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Compute the validation-set ROC curve
    fpr_val, tpr_val, _ = roc_curve(val_labels, val_probs)
    roc_auc_val = auc(fpr_val, tpr_val)

    # Plot the ROC curves
    plt.plot(fpr_train, tpr_train, 'b-', linewidth=2.5,
             label=f'Train (AUC = {roc_auc_train:.4f})')
    plt.plot(fpr_val, tpr_val, 'r-', linewidth=2.5,
             label=f'Validation (AUC = {roc_auc_val:.4f})')

    # Plot the random-classifier baseline
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2,
             label='Random Classifier (AUC = 0.5000)')

    # Figure settings
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    plt.title('ROC Curve Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add the diagonal fill region
    plt.fill_between(fpr_train, tpr_train, alpha=0.2, color='blue')
    plt.fill_between(fpr_val, tpr_val, alpha=0.2, color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {save_path}")
    plt.close()

def create_results_table(train_metrics, val_metrics, save_path):
    """Create a training-results comparison table."""
    # Prepare the data
    results = {
        'Metric': list(train_metrics.keys()),
        'Training Set': list(train_metrics.values()),
        'Validation Set': list(val_metrics.values())
    }

    df = pd.DataFrame(results)

    # Format values (4 decimal places)
    for col in ['Training Set', 'Validation Set']:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)

    # Save as CSV
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Results table saved to: {save_path}")

    # Print the table
    print("\n" + "="*70)
    print("Model Performance Summary")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")

    return df

def main():
    """Main entry point."""
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Using GPU...')
    else:
        device = torch.device('cpu')
        print('Using CPU...')

    # Load the training history to get training parameters
    history_path = 'finetune_ckpt/training_history.json'
    with open(history_path, 'r') as f:
        history = json.load(f)

    # Load data (adjust to the actual training configuration)
    print("\nLoading data...")
    data_path = 'data/dataset.csv'  # adjust as needed
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
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)

    # Evaluate on the training set
    print("Evaluating on training set...")
    train_labels_np, train_probs, train_preds = evaluate_model(model, train_loader, device)
    train_metrics = calculate_metrics(train_labels_np, train_probs, train_preds)

    # Evaluate on the validation set
    print("Evaluating on validation set...")
    val_labels_np, val_probs, val_preds = evaluate_model(model, val_loader, device)
    val_metrics = calculate_metrics(val_labels_np, val_probs, val_preds)

    # Create the image folder
    os.makedirs('image', exist_ok=True)

    # Plot the ROC comparison
    print("\nGenerating ROC curve comparison...")
    plot_roc_comparison(
        train_labels_np, train_probs,
        val_labels_np, val_probs,
        'image/roc_comparison.png'
    )

    # Create the results table
    print("\nGenerating results table...")
    create_results_table(
        train_metrics, val_metrics,
        'image/model_performance_results.csv'
    )

    print("\nEvaluation completed!")
    print(f"Best model AUC (from training): {max(history['val_auc']):.4f}")
    print(f"Current evaluation AUC (validation): {val_metrics['AUC']:.4f}")

if __name__ == "__main__":
    main()
