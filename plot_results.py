"""
Step 2: Plot ROC curves and generate the results table in the gnn environment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, roc_auc_score, accuracy_score,
                            precision_score, recall_score, f1_score,
                            confusion_matrix, precision_recall_curve)
import pickle
import os

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
    # Load prediction results
    print("Loading prediction results...")
    with open('image/prediction_results.pkl', 'rb') as f:
        results = pickle.load(f)

    train_labels = results['train_labels']
    train_probs = results['train_probs']
    train_preds = results['train_preds']
    val_labels = results['val_labels']
    val_probs = results['val_probs']
    val_preds = results['val_preds']

    print(f"Train set: {len(train_labels)} samples")
    print(f"Validation set: {len(val_labels)} samples")

    # Compute metrics
    print("\nCalculating metrics...")
    train_metrics = calculate_metrics(train_labels, train_probs, train_preds)
    val_metrics = calculate_metrics(val_labels, val_probs, val_preds)

    # Plot the ROC comparison
    print("\nGenerating ROC curve comparison...")
    plot_roc_comparison(
        train_labels, train_probs,
        val_labels, val_probs,
        'image/roc_comparison.png'
    )

    # Create the results table
    print("\nGenerating results table...")
    create_results_table(
        train_metrics, val_metrics,
        'image/model_performance_results.csv'
    )

    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main()
