# TransformerCPI2.0 Fine-tuning Guide

## Quick start

### 1. Prepare the data

Create a CSV file with the following columns:

- `smiles`: SMILES string of the compound
- `sequence`: protein sequence
- `label`: label (0 = non-binding, 1 = binding)

Example data format:

```csv
smiles,sequence,label
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,MKKFFDSRREQ...,1
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,MGAAASIQ...,0
```

### 2. Basic fine-tuning command

```bash
python finetune.py \
    --data_path data/your_data.csv \
    --pretrained_model "ckpt/Virtual Screening.pt" \
    --freeze_encoder \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --save_dir finetune_ckpt/my_experiment
```

### 3. Arguments

#### Data arguments

- `--data_path`: **required** path to the training-data CSV
- `--smiles_col`: SMILES column name (default: smiles)
- `--seq_col`: protein-sequence column name (default: sequence)
- `--label_col`: label column name (default: label)
- `--val_split`: validation-set fraction (default: 0.2)

#### Model arguments

- `--pretrained_model`: pretrained model path (default: ckpt/Virtual Screening.pt)
- `--freeze_encoder`: whether to freeze the encoder (**recommended for small datasets**)

#### Training arguments

- `--epochs`: number of epochs (default: 50)
- `--batch_size`: batch size (default: 32)
- `--lr`: learning rate (default: 1e-4)
  - Frozen encoder: 5e-5 to 1e-4
  - Full fine-tuning: 1e-5 to 5e-5
- `--weight_decay`: weight decay (default: 1e-4)
- `--grad_clip`: gradient clipping (default: 1.0)
- `--class_weights`: class weights, format: `--class_weights negative_weight positive_weight`

#### Early stopping and scheduler

- `--early_stop_patience`: early-stopping patience (default: 10)
- `--scheduler_patience`: scheduler patience (default: 3)

#### Other

- `--save_dir`: output directory (default: finetune_ckpt)
- `--save_interval`: checkpoint save interval (default: 5 epochs)
- `--gpu`: GPU index (default: 0)
- `--seed`: random seed (default: 42)

## Use cases

### Case 1: small dataset (<5000 samples)

**Recommended configuration:**

```bash
python finetune.py \
    --data_path data/small_dataset.csv \
    --pretrained_model "ckpt/Virtual Screening.pt" \
    --freeze_encoder \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --early_stop_patience 10
```

**Why?**

- Freezing the encoder prevents overfitting
- A smaller learning rate keeps training stable
- Early stopping prevents overtraining

### Case 2: large dataset (>10000 samples)

**Recommended configuration:**

```bash
python finetune.py \
    --data_path data/large_dataset.csv \
    --pretrained_model "ckpt/Virtual Screening.pt" \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-5 \
    --early_stop_patience 8
```

**Why?**

- With enough data, full fine-tuning is preferable
- A larger batch size improves training efficiency
- A smaller learning rate keeps fine-tuning stable

### Case 3: class imbalance

Assume a 1:10 positive-to-negative ratio:

```bash
python finetune.py \
    --data_path data/imbalanced_data.csv \
    --pretrained_model "ckpt/Virtual Screening.pt" \
    --freeze_encoder \
    --class_weights 1.0 10.0 \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5
```

**Why?**

- `--class_weights 1.0 10.0` gives the positive class 10x weight
- This balances the contribution of both classes during learning

## Output files

After training, `save_dir` will contain:

```
finetune_ckpt/
├── best_model.pt              # best checkpoint (includes optimizer state)
├── best_model_full.pt         # best model (weights only, for inference)
├── checkpoint_epoch_5.pt      # periodic checkpoints
├── checkpoint_epoch_10.pt
├── ...
└── training_history.json      # training history (loss, acc, auc, ...)
```

## Inference with the fine-tuned model

```python
import torch
from featurizer import featurizer
from predict import Tester

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('finetune_ckpt/best_model_full.pt', map_location=device)
model.to(device)

sequence = "MPHSSLHPSIPCPRGHGAQKAALVLLSACLVTLWGLGEPPEHTLRYLVLHLA"
smiles = "CS(=O)(C1=NN=C(S1)CN2C3CCC2C=C(C4=CC=CC=C4)C3)=O"

compounds, adjacencies, proteins = featurizer(smiles, sequence)
tester = Tester(model, device)
test_set = list(zip(compounds, adjacencies, proteins))
score = float(tester.test(test_set))

print(f"Binding score: {score:.4f}")
```

## Monitoring training

During training, the script prints:

- loss for each batch
- train/validation metrics for each epoch
- learning-rate changes
- best-model save messages

Example output:

```
Epoch 10/50
------------------------------
  Batch 10/100, Loss: 0.4521
  Batch 20/100, Loss: 0.4312
  ...
Train - Loss: 0.4123, Acc: 0.8234
Val - Loss: 0.3876, Acc: 0.8456, AUC: 0.9012, AUPR: 0.8765
Learning rate: 0.000050

*** New best model! AUC: 0.9012 ***
```

## FAQ

### Q1: out of memory?
**A:** reduce `--batch_size`, or use `--freeze_encoder` to lower GPU memory.

### Q2: training too slow?
**A:** increase `--batch_size`, reduce `--num_workers`, or use a stronger GPU.

### Q3: overfitting?
**A:**
- use `--freeze_encoder`
- reduce `--lr`
- increase `--weight_decay`
- reduce `--epochs` or rely on early stopping

### Q4: underfitting?
**A:**
- do not use `--freeze_encoder`
- increase `--lr`
- increase `--epochs`
- increase model capacity (modify model.py)

### Q5: how to tune hyperparameters?
**A:** recommended order:
1. fix the other parameters and tune the learning rate (1e-5, 5e-5, 1e-4)
2. tune the batch size (16, 32, 64)
3. tune weight decay (1e-5, 1e-4, 1e-3)
4. decide whether to freeze the encoder

## Best practices

1. Data quality first: make sure SMILES and sequence formats are correct
2. Start small, then scale up
3. Watch AUC and AUPR, not just accuracy
4. Repeat with different random seeds and average the results
5. Record the full configuration for every experiment
6. Back up checkpoints regularly

## Advanced features (not yet implemented)

1. Multi-GPU training with DistributedDataParallel
2. Mixed-precision training with torch.cuda.amp
3. K-fold cross-validation
4. Independent test-set evaluation
5. TensorBoard visualization
6. Resume from checkpoint
7. Hyperparameter search (e.g. Optuna)

## Support

If you run into problems, check:
1. whether the data format is correct
2. whether the pretrained model path exists
3. whether the CUDA version is compatible
4. whether all dependencies are installed
