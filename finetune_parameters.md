# Fine-tuning method and parameters (MMP12)

## 1. Transfer-learning strategy

- Freeze encoder: use `--freeze_encoder` to freeze the pretrained protein encoder
  and only fine-tune the decoder and classification head.
- Rationale: the dataset has 2649 samples (medium size). Freezing the encoder
  prevents overfitting and preserves pretrained knowledge.

## 2. Optimizer and learning rate

- Optimizer: AdamW (Adam with decoupled weight decay)
- Learning rate: 5e-5, a small value suited to fine-tuning
- Weight decay: 1e-4 (L2 regularization against overfitting)
- Scheduler: ReduceLROnPlateau; halves the learning rate when validation AUC
  does not improve for 3 epochs

## 3. Regularization and anti-overfitting

- Gradient clipping: 1.0 (prevents exploding gradients)
- Early stopping: patience = 10 (stops when validation AUC does not improve for
  10 epochs)
- Validation split: 20%, used to monitor generalization

## 4. Training configuration

- Maximum epochs: 50
- Batch size: 32 (balances memory use and training stability)
- Loss: CrossEntropyLoss (binary classification)
- Split: stratified sampling keeps the positive/negative ratio consistent
  across train and validation sets

## 5. Evaluation metrics

- AUC-ROC: primary metric, used for scheduling and early stopping
- AUPR: area under the precision-recall curve
- Accuracy
- Training and validation loss

## Parameter summary

| Category | Parameter | Value | Description |
| --- | --- | --- | --- |
| Data | data_path | data/dataset.csv | Training-data path |
| | smiles_col | smiles | SMILES column name |
| | seq_col | sequence | Protein-sequence column name |
| | label_col | label | Label column name |
| | val_split | 0.2 | 20% validation split |
| Model | pretrained_model | ckpt/Virtual Screening.pt | Pretrained model |
| | freeze_encoder | True | Freeze the encoder |
| Optimization | lr | 5e-5 | Learning rate |
| | weight_decay | 1e-4 | Weight decay |
| | epochs | 50 | Maximum epochs |
| | batch_size | 32 | Batch size |
| | grad_clip | 1.0 | Gradient-clipping threshold |
| Early stopping | early_stop_patience | 10 | Early-stopping patience |
| | scheduler_patience | 3 | Scheduler patience |
| Other | gpu | 0 | GPU index |
| | seed | 42 | Random seed |
| | save_dir | finetune_ckpt | Model output directory |

## Why these parameters

1. Freeze encoder: 2649 samples is not very large; full fine-tuning overfits easily.
2. Small learning rate (5e-5): avoids damaging pretrained weights.
3. batch_size = 32: conservative for an RTX 4090 (24 GB); can be increased to 64.
4. early_stop_patience = 10: enough time to find the optimum without overtraining.
