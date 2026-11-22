# MMP12 Virtual Screening Model (Fine-tuned from TransformerCPI2.0)

This project fine-tunes the TransformerCPI2.0 pretrained model for human MMP12
(macrophage metalloelastase, UniProt P39900) and uses it for high-throughput
virtual screening of large compound libraries, scaffold clustering, and
molecular docking validation.

## Pipeline

1. Data collection: IC50 data for MMP12 inhibitors were collected from three
   public bioactivity databases (PubChem, ChEMBL, BindingDB) and converted to nM.
2. Labeling: compounds were binarized at a 1000 nM activity threshold
   (0 = inactive, 1 = active).
3. Cleaning: SMILES were canonicalized, desalted, and deduplicated with RDKit;
   duplicate entries were merged by geometric mean.
4. Fine-tuning: the protein encoder was frozen and only the molecular module,
   decoder, and classification head were trained.
5. Virtual screening: compounds were scored by binding likelihood and ranked in
   descending order.
6. Scaffold clustering: Murcko scaffolds, Morgan fingerprints, PCA chemical-space
   visualization, and Butina clustering.
7. Docking: AutoDock Vina was used to validate top-ranked candidates.

## Repository contents

- `model.py` / `featurizer.py` / `predict.py`: model architecture and inference
- `finetune.py`: fine-tuning script
- `batch_prediction.py` / `chembl_virtual_screen.py`: batch prediction and
  virtual screening
- `evaluate_model.py` / `evaluate_and_save.py`: performance evaluation
- `analyze_dataset.py`: dataset statistics and property analysis
- `scaffold_cluster.py`: scaffold clustering analysis
- `plot_results.py` / `plot_training_history.py`: result visualization
- `data/`: training data, screening results, and docking results
- `FINETUNE_README.md` / `finetune_parameters.md`: fine-tuning guide and parameters
- `LICENSE`: GPL v2

## Environment

- python 3.8.8
- pytorch 1.9
- tape-proteins 0.5
- rdkit 2021.03.5
- numpy 1.19.5
- scikit-learn 0.24.1

## Fine-tuning (MMP12)

```bash
python finetune.py \
    --data_path path/to/Supplementary_Data_1.csv \
    --pretrained_model "ckpt/Virtual Screening.pt" \
    --freeze_encoder \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --early_stop_patience 10 \
    --scheduler_patience 3 \
    --gpu 0 \
    --seed 42 \
    --save_dir finetune_ckpt
```

## Inference example

```python
import torch
from featurizer import featurizer
from predict import Tester

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('finetune_ckpt/best_model_full.pt', map_location=device)
model.to(device)

sequence = "MKFLLILLLQATASGALPLNSSTSLEKNNVLFGERYLEKFYGLEINKLPVTKMKYSGNLM"
smiles = "CS(=O)(C1=NN=C(S1)CN2C3CCC2C=C(C4=CC=CC=C4)C3)=O"

compounds, adjacencies, proteins = featurizer(smiles, sequence)
tester = Tester(model, device)
score = float(tester.test(list(zip(compounds, adjacencies, proteins))))
print(f"Binding score: {score:.4f}")
```

## Data

The curated MMP-12 bioactivity dataset (2,649 samples; 1,532 active and
1,117 inactive), the virtual-screening predictions and the docking results are
provided as Supplementary Data 1-3 of the accompanying article, so they are not
duplicated in this repository.

- `data/chembl.csv`, `data/bindingDB.csv`: raw source records used to build the
  curated dataset
- `data/mmp12.fasta`: MMP12 protein sequence (UniProt P39900)

To run fine-tuning, export Supplementary Data 1 to a CSV file containing the
`smiles`, `sequence` and `label` columns, and pass its path to `--data_path`.

## Model weights

The fine-tuned MMP12 inference model is available at `models/best_model_full.pt`
(stored with Git LFS). It is the model described in this repository, fine-tuned
on the 2649-sample MMP12-only training set. Clone with Git LFS enabled to
download the weights:

Zenodo DOI: [10.5281/zenodo.24438954](https://doi.org/10.5281/zenodo.24438954)

```bash
git lfs install
git clone https://github.com/HZC-SMU/MMP12-virtual-screening.git
```

## License and citation

This project builds on TransformerCPI2.0, which is licensed under GPL v2;
derivative code is likewise GPL v2. Please cite TransformerCPI2.0, Nature
Communications (2023), doi:10.1038/s41467-023-39856-w.
