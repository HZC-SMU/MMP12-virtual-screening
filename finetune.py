# -*- coding: utf-8 -*-
"""
TransformerCPI2.0 Fine-tuning Script
Fine-tune from a pretrained model, with full training loop, validation, and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from featurizer import featurizer
from model import pack
import warnings
warnings.filterwarnings('ignore')


class CPIDataset(Dataset):
    """Drug-protein interaction dataset."""
    def __init__(self, smiles_list, sequence_list, labels):
        """
        Args:
            smiles_list: list of SMILES strings
            sequence_list: list of protein sequences
            labels: list of labels (0 or 1 indicating binding)
        """
        self.smiles_list = smiles_list
        self.sequence_list = sequence_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        sequence = self.sequence_list[idx]
        label = self.labels[idx]

        try:
            compounds, adjacencies, proteins = featurizer(smiles, sequence)
            return (compounds[0], adjacencies[0], proteins[0], label)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return the next sample
            return self.__getitem__((idx + 1) % len(self))


def collate_fn(batch):
    """Custom batch collate function."""
    compounds, adjacencies, proteins, labels = zip(*batch)
    labels = torch.FloatTensor(labels)
    return list(compounds), list(adjacencies), list(proteins), labels


class Trainer:
    """Trainer class encapsulating the full training loop."""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # Set up the optimizer
        if config['freeze_encoder']:
            # Optimize only the decoder and classification head
            trainable_params = []
            for name, param in model.named_parameters():
                if 'encoder.pretrain' not in name:  # freeze the pretrained protein encoder
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
            print(f"Encoder frozen; trainable parameters: {sum(p.numel() for p in trainable_params)}")
        else:
            trainable_params = model.parameters()
            print(f"Full fine-tuning; trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # Optimizer and learning-rate scheduler
        self.optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=config['scheduler_patience'], verbose=True
        )

        # Loss function (supports class weights)
        if config.get('class_weights') is not None:
            class_weights = torch.FloatTensor(config['class_weights']).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Early stopping
        self.best_auc = 0
        self.patience_counter = 0
        self.early_stop = False

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_aupr': []
        }

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (compounds, adjacencies, proteins, labels) in enumerate(train_loader):
            labels = labels.long().to(self.device)

            # Forward pass
            data = pack(compounds, adjacencies, proteins, self.device)
            outputs = self.model.forward(*data)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % self.config['log_interval'] == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)

        return avg_loss, acc

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for compounds, adjacencies, proteins, labels in val_loader:
                labels_long = labels.long().to(self.device)

                # Forward pass
                data = pack(compounds, adjacencies, proteins, self.device)
                outputs = self.model.forward(*data)

                # Compute loss
                loss = self.criterion(outputs, labels_long)
                total_loss += loss.item()

                # Get predictions
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        avg_loss = total_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)

        # Compute AUC and AUPR
        try:
            auc_score = roc_auc_score(all_labels, all_probs)
            precision, recall, _ = precision_recall_curve(all_labels, all_probs)
            aupr = auc(recall, precision)
        except:
            auc_score = 0.5
            aupr = 0.5

        return avg_loss, acc, auc_score, aupr

    def fit(self, train_loader, val_loader, num_epochs):
        """Full training loop."""
        print(f"\n{'='*50}")
        print(f"Starting training - total epochs: {num_epochs}")
        print(f"{'='*50}\n")

        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 30)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_auc, val_aupr = self.validate(val_loader)

            # Learning-rate scheduling
            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['val_aupr'].append(val_aupr)

            # Print results
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, AUPR: {val_aupr:.4f}")
            print(f"Learning rate: {current_lr:.6f}\n")

            # Save the best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"*** New best model! AUC: {val_auc:.4f} ***\n")
            else:
                self.patience_counter += 1

            # Periodically save a checkpoint
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early-stopping check
            if self.patience_counter >= self.config['early_stop_patience']:
                print(f"Early stopping triggered! Validation AUC did not improve for {self.config['early_stop_patience']} epochs")
                self.early_stop = True
                break

        print(f"\n{'='*50}")
        print(f"Training finished! Best validation AUC: {self.best_auc:.4f}")
        print(f"{'='*50}\n")

        return self.history

    def save_checkpoint(self, epoch, is_best=False):
        """Save a model checkpoint."""
        os.makedirs(self.config['save_dir'], exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_auc': self.best_auc,
            'history': self.history,
            'config': self.config
        }

        if is_best:
            path = os.path.join(self.config['save_dir'], 'best_model.pt')
            torch.save(checkpoint, path)
            # Also save a weights-only version for easier deployment
            torch.save(self.model, os.path.join(self.config['save_dir'], 'best_model_full.pt'))
        else:
            path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, path)

        print(f"Checkpoint saved: {path}")


def load_data_from_csv(csv_path, smiles_col='smiles', seq_col='sequence', label_col='label'):
    """
    Load data from a CSV file

    Args:
        csv_path: Path to the CSV file
        smiles_col: SMILES column name
        seq_col: protein sequence column name
        label_col: label column name

    Returns:
        smiles_list, sequence_list, labels
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Positive samples: {sum(df[label_col] == 1)}, negative samples: {sum(df[label_col] == 0)}")

    return df[smiles_col].tolist(), df[seq_col].tolist(), df[label_col].tolist()


def main(args):
    """Main entry point."""

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f'Using GPU: {args.gpu}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # Load data
    print("\nLoading data...")
    smiles_list, sequence_list, labels = load_data_from_csv(
        args.data_path, args.smiles_col, args.seq_col, args.label_col
    )

    # Split into train and validation sets
    train_smiles, val_smiles, train_seq, val_seq, train_labels, val_labels = train_test_split(
        smiles_list, sequence_list, labels,
        test_size=args.val_split, random_state=args.seed, stratify=labels
    )

    print(f"Train set: {len(train_smiles)}, validation set: {len(val_smiles)}")

    # Create datasets and data loaders
    train_dataset = CPIDataset(train_smiles, train_seq, train_labels)
    val_dataset = CPIDataset(val_smiles, val_seq, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    # Load the pretrained model
    print(f"\nLoading pretrained model from {args.pretrained_model}...")
    model = torch.load(args.pretrained_model, map_location=device)
    model.to(device)

    # Configuration
    config = {
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'freeze_encoder': args.freeze_encoder,
        'scheduler_patience': args.scheduler_patience,
        'early_stop_patience': args.early_stop_patience,
        'grad_clip': args.grad_clip,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'save_dir': args.save_dir,
        'class_weights': args.class_weights
    }

    # Create the trainer
    trainer = Trainer(model, device, config)

    # Start training
    history = trainer.fit(train_loader, val_loader, args.epochs)

    # Save the training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining history saved to: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TransformerCPI2.0 fine-tuning script')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training-data CSV file')
    parser.add_argument('--smiles_col', type=str, default='smiles',
                        help='SMILES column name in the CSV')
    parser.add_argument('--seq_col', type=str, default='sequence',
                        help='Protein sequence column name in the CSV')
    parser.add_argument('--label_col', type=str, default='label',
                        help='Label column name in the CSV')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation-set fraction')

    # Model arguments
    parser.add_argument('--pretrained_model', type=str, default='ckpt/Virtual Screening.pt',
                        help='Path to the pretrained model')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Whether to freeze the encoder (recommended for small datasets)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold; 0 disables clipping')
    parser.add_argument('--class_weights', type=float, nargs=2, default=None,
                        help='Class weights [negative, positive] for handling class imbalance')

    # Early stopping and scheduler
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early-stopping patience')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='Learning-rate scheduler patience')

    # Other arguments
    parser.add_argument('--save_dir', type=str, default='finetune_ckpt',
                        help='Directory for saving models')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save a checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Print logs every N batches')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data-loading workers')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Print the configuration
    print("\n" + "="*50)
    print("Fine-tuning configuration:")
    print("="*50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*50 + "\n")

    main(args)
