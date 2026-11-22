import argparse
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import MACCSkeys
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs


def compute_scaffold_smiles(smiles: str, include_chirality: bool = False) -> str:
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
        return scaffold
    except Exception:
        # Fall back: try via Mol object
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
            return smi
        except Exception:
            return ""


def aggregate_scaffolds(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Compute Murcko scaffolds
    df = df.copy()
    df["scaffold"] = [compute_scaffold_smiles(smi) for smi in df["smiles"].astype(str)]
    df = df[df["scaffold"].astype(str).str.len() > 0]
    if df.empty:
        raise RuntimeError("Could not generate any Murcko scaffolds from the input molecules.")

    # Group by scaffold and aggregate scores
    agg = df.groupby("scaffold").agg(
        n_compounds=("id", "count"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        max_score=("score", "max"),
    )
    agg = agg.reset_index()
    agg = agg.sort_values(["mean_score", "max_score", "n_compounds"], ascending=[False, False, False])
    return agg, df


def _bv_to_array(bv) -> np.ndarray:
    arr = np.zeros((bv.GetNumBits(),), dtype=float)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _get_fp_bitvect(mol: Chem.Mol, fp_type: str, n_bits: int, radius: int):
    t = fp_type.lower()
    if t == "morgan":
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    if t == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)  # 167 bits
    if t == "rdk":
        return Chem.RDKFingerprint(mol, fpSize=n_bits, minPath=1, maxPath=7)
    # default fallback
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def get_fps_numpy(smiles_list: List[str], n_bits: int = 1024, radius: int = 2, fp_type: str = "morgan") -> Tuple[np.ndarray, List[int]]:
    fps = []
    keep = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        bv = _get_fp_bitvect(mol, fp_type=fp_type, n_bits=n_bits, radius=radius)
        arr = _bv_to_array(bv)
        fps.append(arr)
        keep.append(i)
    if not fps:
        return np.empty((0, 0)), []
    # ensure consistent shape
    max_bits = max(a.shape[0] for a in fps)
    fps = [a if a.shape[0] == max_bits else np.pad(a, (0, max_bits - a.shape[0])) for a in fps]
    return np.vstack(fps), keep


def get_fp_bitvecs(smiles_list: List[str], n_bits: int = 1024, radius: int = 2, fp_type: str = "morgan"):
    bitvecs = []
    keep = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        bv = _get_fp_bitvect(mol, fp_type=fp_type, n_bits=n_bits, radius=radius)
        bitvecs.append(bv)
        keep.append(i)
    return bitvecs, keep


def butina_cluster_labels(bitvecs: List, cutoff: float) -> List[int]:
    n = len(bitvecs)
    if n == 0:
        return []
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bitvecs[i], bitvecs[:i])
        dists.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, 1.0 - float(cutoff), isDistData=True)
    labels = [-1] * n
    for cid, cl in enumerate(clusters):
        for idx in cl:
            labels[idx] = cid
    return labels


def pca_fit_transform(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]
    X2 = Xc @ comps.T
    mean = X.mean(axis=0)
    return X2, mean, comps


def pca_transform(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    Xc = X - mean
    return Xc @ comps.T


def assign_class_ids(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy().reset_index(drop=True)
    agg["class_id"] = [f"S{i+1}" for i in range(len(agg))]
    return agg


def prepare_top_scaffolds(agg_with_ids: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    top = agg_with_ids.head(top_n).copy().reset_index(drop=True)
    return top


def plot_top_scaffold_bars(top: pd.DataFrame, out_dir: str) -> str:
    if top.empty:
        return ""
    fig, ax = plt.subplots(figsize=(max(12, len(top) * 0.4), 6))
    bars = ax.bar(range(len(top)), top["mean_score"].values, color="#4C72B0")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["class_id"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Mean Score")
    ax.set_title("Top Scaffolds (Mean Score)")
    ax.grid(True, axis="y", alpha=0.3)
    # Set y-axis range to emphasize differences
    ax.set_ylim(0.95, 1.003)
    # Add value labels above bars
    for i, v in enumerate(top["mean_score"].values):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    out_path = os.path.join(out_dir, "scaffolds_top30_bar.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def draw_scaffold_grid_with_ids(top: pd.DataFrame, out_dir: str, mols_per_row: int = 6) -> str:
    if top.empty:
        return ""
    mols, legends = [], []
    for _, row in top.iterrows():
        mol = Chem.MolFromSmiles(row["scaffold"]) 
        if mol is None:
            continue
        mols.append(mol)
        legends.append(row["class_id"])  # only show class id under image
    if not mols:
        return ""
    try:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(220, 220), legends=legends, useSVG=False)
        out_path = os.path.join(out_dir, "scaffolds_top30_grid.png")
        img.save(out_path)
        return out_path
    except Exception:
        svg = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(220, 220), legends=legends, useSVG=True)
        out_path = os.path.join(out_dir, "scaffolds_top30_grid.svg")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg.data if hasattr(svg, "data") else str(svg))
        return out_path


def plot_molecule_chemspace(df: pd.DataFrame, out_dir: str, cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize, n_bits: int, radius: int, fp_type: str) -> Tuple[str, np.ndarray, np.ndarray]:
    X, keep = get_fps_numpy(df["smiles"].astype(str).tolist(), n_bits=n_bits, radius=radius, fp_type=fp_type)
    if X.size == 0:
        return "", np.array([]), np.array([])
    coords, mean, comps = pca_fit_transform(X)
    df_keep = df.iloc[keep].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=df_keep["score"].values, cmap=cmap, norm=norm, s=30, edgecolors="k", linewidths=0.3, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Score")
    ax.set_xlabel("PC1 (Morgan FP)")
    ax.set_ylabel("PC2 (Morgan FP)")
    ax.set_title("Molecules Chemical Space")
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, "chemspace_molecules.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path, mean, comps


def plot_scaffold_chemspace(
    agg: pd.DataFrame,
    out_dir: str,
    mean: np.ndarray,
    comps: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    n_bits: int,
    radius: int,
    min_mean: float = 0.5,
    top_k: int = 300,
    fp_type: str = "morgan",
    size_scale: float = 3.0,
    colorbar_mode: str = "local",
) -> Tuple[str, str]:
    # keep top_k scaffolds by current ranking, then filter by mean score threshold
    agg_t = agg.head(top_k).copy()
    agg_f = agg_t[agg_t["mean_score"] >= float(min_mean)].copy()
    if agg_f.empty:
        return ""
    Xs, keep = get_fps_numpy(agg_f["scaffold"].astype(str).tolist(), n_bits=n_bits, radius=radius, fp_type=fp_type)
    if Xs.size == 0:
        return ""
    coords = pca_transform(Xs, mean, comps)
    agg_keep = agg_f.iloc[keep].copy()
    # emphasize size differences with a multiplicative scale factor
    counts = agg_keep["n_compounds"].values.astype(float)
    cmin, cmax = counts.min(), counts.max()
    if cmax > cmin:
        norm_counts = (counts - cmin) / (cmax - cmin)
    else:
        norm_counts = np.ones_like(counts) * 0.5
    sizes = (60 + 340 * (norm_counts ** 1.5)) * float(size_scale)
    # color scaling: local (class min-max) or global (same as molecules)
    if colorbar_mode == "local":
        color_norm = mpl.colors.Normalize(vmin=float(agg_keep["mean_score"].min()), vmax=float(agg_keep["mean_score"].max()))
        cbar_label = "Score (class min-max)"
    else:
        color_norm = norm
        cbar_label = "Score (same scale)"

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=agg_keep["mean_score"].values,
        cmap=cmap,
        norm=color_norm,
        s=sizes,
        edgecolors="k",
        linewidths=0.4,
        alpha=0.9,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(cbar_label)
    ax.set_xlabel("PC1 (Morgan FP)")
    ax.set_ylabel("PC2 (Morgan FP)")
    ax.set_title("Scaffolds Chemical Space (size = count)")
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, "chemspace_scaffolds.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    # export csv source for the scatter
    scatter_df = agg_keep.copy()
    scatter_df["PC1"] = coords[:, 0]
    scatter_df["PC2"] = coords[:, 1]
    scatter_df["size"] = sizes
    csv_path = os.path.join(out_dir, "chemspace_scaffolds.csv")
    # keep key columns at front if present
    cols = [c for c in ["class_id", "scaffold", "mean_score", "median_score", "max_score", "n_compounds", "PC1", "PC2", "size"] if c in scatter_df.columns]
    scatter_df[cols].to_csv(csv_path, index=False)
    return out_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Scaffold clustering and chemical-space analysis (compact version)")
    parser.add_argument("--input", required=True, help="Prediction-results CSV; must contain columns: id,smiles,score")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to the input file's directory")
    parser.add_argument("--bar-top", type=int, default=30, help="Number of top scaffolds shown in the bar chart (default 30)")
    parser.add_argument("--fp-bits", type=int, default=2048, help="Morgan fingerprint length in bits")
    parser.add_argument("--fp-radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--scaffold-min-mean", type=float, default=0.5, help="Minimum mean score for including a scaffold in the scatter plot (default 0.5)")
    parser.add_argument("--scaffold-top-k", type=int, default=300, help="Only plot the top-K scaffolds in the scatter plot (default 300)")
    parser.add_argument("--size-scale", type=float, default=3.0, help="Point-size scale factor for the scaffold scatter plot (default 3.0)")
    parser.add_argument("--fp-type", choices=["morgan", "maccs", "rdk"], default="morgan", help="Fingerprint type: Morgan, MACCS, or RDKit")
    parser.add_argument("--sim-threshold", type=float, default=None, help="Similarity clustering threshold (Tanimoto); molecules are clustered by this threshold when set")
    parser.add_argument("--scaffold-colorbar", choices=["global", "local"], default="local", help="Colorbar mode for the scaffold scatter plot: global=shared color scale; local=per-class min/max score")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(input_path))
    os.makedirs(out_dir, exist_ok=True)

    # Load results
    df = pd.read_csv(input_path)
    cols_lower = {c.lower(): c for c in df.columns}
    try:
        df = df[[cols_lower["id"], cols_lower["smiles"], cols_lower["score"]]].rename(
            columns={cols_lower["id"]: "id", cols_lower["smiles"]: "smiles", cols_lower["score"]: "score"}
        )
    except Exception:
        raise SystemExit("The input CSV must contain columns: id, smiles, score")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["smiles", "score"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("No valid records available for analysis.")

    # 1) Scaffold aggregation
    agg, detailed = aggregate_scaffolds(df)
    agg_path = os.path.join(out_dir, "scaffolds_agg.csv")
    # assign class ids to ALL scaffolds (ordered by ranking)
    agg_with_ids = assign_class_ids(agg)
    agg_with_ids.to_csv(agg_path, index=False)
    # Prepare top scaffolds with class ids
    top_df = prepare_top_scaffolds(agg_with_ids, top_n=args.bar_top)
    # Save bar plot source CSV
    bar_csv_path = os.path.join(out_dir, "scaffolds_top30_bar.csv")
    top_export_cols = [c for c in ["class_id", "scaffold", "mean_score", "median_score", "max_score", "n_compounds"] if c in top_df.columns]
    top_df[top_export_cols].to_csv(bar_csv_path, index=False)
    # 1a) Bar chart (ymin=0.95, no images)
    bar_path = plot_top_scaffold_bars(top_df, out_dir)
    # 1b) Separate scaffold grid with class ids under images
    grid_path = draw_scaffold_grid_with_ids(top_df, out_dir)

    # Prepare shared colormap/norm using global molecule scores
    vmin, vmax = float(df["score"].min()), float(df["score"].max())
    cmap = mpl.colormaps.get("viridis") if hasattr(mpl, "colormaps") else plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # 2) Molecules chem space (colored by score, with colorbar)
    mol_space_path, mean_vec, comps = plot_molecule_chemspace(df, out_dir, cmap=cmap, norm=norm, n_bits=args.fp_bits, radius=args.fp_radius, fp_type=args.fp_type)

    # 3) Scaffolds chem space (size = count, color = mean score, same colorbar scale)
    scaffold_space_path, scaffold_space_csv = plot_scaffold_chemspace(
        agg_with_ids,
        out_dir,
        mean_vec,
        comps,
        cmap=cmap,
        norm=norm,
        n_bits=args.fp_bits,
        radius=args.fp_radius,
        min_mean=args.scaffold_min_mean,
        top_k=args.scaffold_top_k,
        fp_type=args.fp_type,
        size_scale=args.size_scale,
        colorbar_mode=args.scaffold_colorbar,
    )

    # 4) Full molecule-to-scaffold mapping with class id
    mapping = detailed.merge(agg_with_ids[["scaffold", "class_id"]], on="scaffold", how="left")
    mapping_out = mapping[["class_id", "id", "smiles", "scaffold", "score"]].rename(
        columns={"id": "molecule_id", "scaffold": "scaffold_smiles"}
    )
    mapping_path = os.path.join(out_dir, "scaffold_molecule_mapping.csv")
    mapping_out.to_csv(mapping_path, index=False)

    # 5) Optional: similarity-based clustering with user-defined threshold
    sim_map_path = ""
    if args.sim_threshold is not None:
        bitvecs, keep = get_fp_bitvecs(df["smiles"].astype(str).tolist(), n_bits=args.fp_bits, radius=args.fp_radius, fp_type=args.fp_type)
        if bitvecs:
            labels = butina_cluster_labels(bitvecs, cutoff=float(args.sim_threshold))
            df_keep = df.iloc[keep].copy()
            df_keep["sim_class_id"] = [f"C{l+1}" if l >= 0 else "C-1" for l in labels]
            sim_map = df_keep[["sim_class_id", cols_lower["id"], "smiles", "score"]].rename(columns={cols_lower["id"]: "molecule_id"})
            sim_map_path = os.path.join(out_dir, "similarity_clusters.csv")
            sim_map.to_csv(sim_map_path, index=False)

    print(f"Scaffold statistics CSV: {agg_path}")
    if bar_path:
        print(f"Top-scaffold bar chart: {bar_path}")
        print(f"Top-scaffold bar chart data: {bar_csv_path}")
    if mol_space_path:
        print(f"Molecule chemical-space scatter plot: {mol_space_path}")
    if scaffold_space_path:
        print(f"Scaffold chemical-space scatter plot: {scaffold_space_path}")
        print(f"Scaffold chemical-space scatter data: {scaffold_space_csv}")
    if grid_path:
        print(f"Scaffold image grid: {grid_path}")
    print(f"Molecule-to-scaffold mapping CSV: {mapping_path}")
    if sim_map_path:
        print(f"Molecule similarity clustering CSV: {sim_map_path}")


if __name__ == "__main__":
    main()
