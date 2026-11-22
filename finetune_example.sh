#!/bin/bash
# TransformerCPI2.0 fine-tuning example script

# ============================================
# Example 1: small dataset (freeze the encoder)
# ============================================
echo "Example 1: small dataset, freezing the pretrained encoder"
python finetune.py \
    --data_path data/your_training_data.csv \
    --smiles_col smiles \
    --seq_col sequence \
    --label_col label \
    --pretrained_model "ckpt/Virtual Screening.pt" \
    --freeze_encoder \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --early_stop_patience 10 \
    --scheduler_patience 3 \
    --val_split 0.2 \
    --save_dir finetune_ckpt/experiment_1 \
    --gpu 0 \
    --seed 42

# ============================================
# Example 2: large dataset, full fine-tuning
# ============================================
echo "Example 2: large dataset, full-parameter fine-tuning"
python finetune.py \
    --data_path data/your_large_training_data.csv \
    --smiles_col smiles \
    --seq_col sequence \
    --label_col label \
    --pretrained_model "ckpt/Virtual Screening.pt" \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-5 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --early_stop_patience 8 \
    --scheduler_patience 2 \
    --val_split 0.15 \
    --save_dir finetune_ckpt/experiment_2 \
    --gpu 0 \
    --seed 42

# ============================================
# Example 3: class-imbalanced dataset (class weights)
# ============================================
echo "Example 3: handling class imbalance with class weights"
# Assuming a 1:10 positive-to-negative ratio, give the positive class 10x weight
python finetune.py \
    --data_path data/imbalanced_data.csv \
    --smiles_col smiles \
    --seq_col sequence \
    --label_col label \
    --pretrained_model "ckpt/Virtual Screening.pt" \
    --freeze_encoder \
    --class_weights 1.0 10.0 \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --save_dir finetune_ckpt/experiment_3 \
    --gpu 0

# ============================================
# Example 4: multi-GPU training (requires DDP support in code)
# ============================================
# Note: this version does not support multi-GPU; DistributedDataParallel must be added

# ============================================
# Example 5: resume training from a checkpoint
# ============================================
# Note: resuming from a checkpoint is not yet supported; it can be added in the code
