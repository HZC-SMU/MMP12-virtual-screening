"""
Dataset statistics and molecular property-activity correlation analysis
Run in the gnn environment; uses RDKit to compute molecular properties
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def calculate_molecular_properties(smiles):
    """Compute molecular properties."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        properties = {
            'MW': Descriptors.MolWt(mol),  # molecular weight
            'LogP': Crippen.MolLogP(mol),  # hydrophobicity
            'HBD': Lipinski.NumHDonors(mol),  # hydrogen bond donors
            'HBA': Lipinski.NumHAcceptors(mol),  # hydrogen bond acceptors
            'TPSA': Descriptors.TPSA(mol),  # polar surface area
            'RotBonds': Lipinski.NumRotatableBonds(mol),  # rotatable bonds
            'Rings': Lipinski.RingCount(mol),  # ring count
            'AromaticRings': Lipinski.NumAromaticRings(mol)  # aromatic ring count
        }
        return properties
    except:
        return None

def load_and_process_data(csv_path):
    """Load and process the data."""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Compute pIC50 (converted from nM)
    df['pIC50'] = -np.log10(df['value_nM'] * 1e-9)

    # Compute molecular properties.
    print("Calculating molecular properties...")
    properties_list = []
    valid_indices = []

    for idx, smiles in enumerate(df['smiles']):
        if idx % 500 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        props = calculate_molecular_properties(smiles)
        if props is not None:
            properties_list.append(props)
            valid_indices.append(idx)

    # Create a DataFrame of properties
    props_df = pd.DataFrame(properties_list)
    df_processed = df.iloc[valid_indices].reset_index(drop=True)
    df_final = pd.concat([df_processed, props_df], axis=1)

    print(f"Valid samples after property calculation: {len(df_final)}")
    return df_final

def plot_dataset_statistics(df, save_dir='image'):
    """Plot dataset statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Statistics', fontsize=16, fontweight='bold')

    # 1. Activity label distribution
    label_counts = df['label'].value_counts()
    axes[0, 0].bar(['Inactive (0)', 'Active (1)'],
                   [label_counts[0], label_counts[1]],
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Label Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, (label, count) in enumerate([(0, label_counts[0]), (1, label_counts[1])]):
        axes[0, 0].text(i, count + 20, str(count), ha='center', fontsize=12, fontweight='bold')

    # 2. pIC50 distribution
    axes[0, 1].hist(df['pIC50'], bins=30, color='#95E1D3', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df['pIC50'].median(), color='red', linestyle='--',
                       linewidth=2, label=f'Median: {df["pIC50"].median():.2f}')
    axes[0, 1].set_xlabel('pIC50', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('pIC50 Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. pIC50 distribution grouped by label
    active = df[df['label'] == 1]['pIC50']
    inactive = df[df['label'] == 0]['pIC50']

    bp = axes[0, 2].boxplot([active, inactive],
                            labels=['Active', 'Inactive'],
                            patch_artist=True,
                            showmeans=True)
    for patch, color in zip(bp['boxes'], ['#4ECDC4', '#FF6B6B']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 2].set_ylabel('pIC50', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('pIC50 by Activity', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # 4. Molecular weight distribution
    axes[1, 0].hist(df['MW'], bins=30, color='#F38181', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['MW'].median(), color='blue', linestyle='--',
                       linewidth=2, label=f'Median: {df["MW"].median():.1f}')
    axes[1, 0].set_xlabel('Molecular Weight (Da)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Molecular Weight Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. LogP distribution
    axes[1, 1].hist(df['LogP'], bins=30, color='#AA96DA', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(df['LogP'].median(), color='red', linestyle='--',
                       linewidth=2, label=f'Median: {df["LogP"].median():.2f}')
    axes[1, 1].set_xlabel('LogP', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('LogP Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. TPSA distribution
    axes[1, 2].hist(df['TPSA'], bins=30, color='#FCBAD3', alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(df['TPSA'].median(), color='blue', linestyle='--',
                       linewidth=2, label=f'Median: {df["TPSA"].median():.1f}')
    axes[1, 2].set_xlabel('TPSA (Ų)', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 2].set_title('TPSA Distribution', fontsize=14, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/dataset_statistics.png', dpi=300, bbox_inches='tight')
    print(f"Dataset statistics plot saved to: {save_dir}/dataset_statistics.png")
    plt.close()

def plot_property_activity_correlation(df, save_dir='image'):
    """Plot correlations between molecular properties and activity."""
    properties = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Molecular Properties vs Activity (pIC50)', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, prop in enumerate(properties):
        # scatter plot
        active = df[df['label'] == 1]
        inactive = df[df['label'] == 0]

        axes[idx].scatter(inactive[prop], inactive['pIC50'],
                         alpha=0.5, s=30, color='#FF6B6B', label='Inactive', edgecolors='black', linewidth=0.5)
        axes[idx].scatter(active[prop], active['pIC50'],
                         alpha=0.5, s=30, color='#4ECDC4', label='Active', edgecolors='black', linewidth=0.5)

        # compute correlation
        pearson_r, pearson_p = pearsonr(df[prop], df['pIC50'])
        spearman_r, spearman_p = spearmanr(df[prop], df['pIC50'])

        # add trend line
        z = np.polyfit(df[prop], df['pIC50'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[prop].min(), df[prop].max(), 100)
        axes[idx].plot(x_line, p(x_line), "k--", alpha=0.8, linewidth=2)

        axes[idx].set_xlabel(prop, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('pIC50', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{prop} vs pIC50\nPearson r={pearson_r:.3f} (p={pearson_p:.3e})',
                           fontsize=12, fontweight='bold')
        axes[idx].legend(loc='best', fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/property_activity_correlation.png', dpi=300, bbox_inches='tight')
    print(f"Property-activity correlation plot saved to: {save_dir}/property_activity_correlation.png")
    plt.close()

def create_correlation_heatmap(df, save_dir='image'):
    """Create a property correlation heatmap."""
    properties = ['pIC50', 'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'Rings', 'AromaticRings']
    corr_df = df[properties].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Molecular Properties', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to: {save_dir}/correlation_heatmap.png")
    plt.close()

def create_summary_table(df, save_dir='image'):
    """Create a dataset summary table."""
    properties = ['pIC50', 'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'Rings', 'AromaticRings']

    summary_stats = []
    for prop in properties:
        stats = {
            'Property': prop,
            'Mean': df[prop].mean(),
            'Std': df[prop].std(),
            'Min': df[prop].min(),
            'Q25': df[prop].quantile(0.25),
            'Median': df[prop].median(),
            'Q75': df[prop].quantile(0.75),
            'Max': df[prop].max()
        }

        # compute correlation with pIC50
        if prop != 'pIC50':
            pearson_r, pearson_p = pearsonr(df[prop], df['pIC50'])
            stats['Pearson_r'] = pearson_r
            stats['Pearson_p'] = pearson_p

        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    # format
    for col in ['Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max']:
        summary_df[col] = summary_df[col].apply(lambda x: f'{x:.2f}')

    if 'Pearson_r' in summary_df.columns:
        summary_df['Pearson_r'] = summary_df['Pearson_r'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        summary_df['Pearson_p'] = summary_df['Pearson_p'].apply(lambda x: f'{x:.4e}' if pd.notna(x) else 'N/A')

    # save
    summary_df.to_csv(f'{save_dir}/dataset_summary.csv', index=False, encoding='utf-8-sig')
    print(f"Dataset summary table saved to: {save_dir}/dataset_summary.csv")

    # print
    print("\n" + "="*100)
    print("Dataset Summary Statistics")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100 + "\n")

def main():
    """Main entry point."""
    # Load and process the data
    df = load_and_process_data('data/dataset.csv')

    # Plot statistics
    print("\nGenerating dataset statistics plots...")
    plot_dataset_statistics(df)

    # Plot property-activity correlations
    print("\nGenerating property-activity correlation plots...")
    plot_property_activity_correlation(df)

    # Create the correlation heatmap
    print("\nGenerating correlation heatmap...")
    create_correlation_heatmap(df)

    # Create the summary table
    print("\nGenerating summary table...")
    create_summary_table(df)

    print("\nAll analyses completed!")
    print(f"\nBasic dataset info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Active samples: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    print(f"  Inactive samples: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()
