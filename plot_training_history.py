import json
import matplotlib.pyplot as plt
import os

# Read training-history data
with open('finetune_ckpt/training_history.json', 'r') as f:
    history = json.load(f)

# Create the image folder
os.makedirs('image', exist_ok=True)

# Configure the font - use English to avoid font issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

epochs = range(1, len(history['train_loss']) + 1)

# Create a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Training History', fontsize=16, fontweight='bold')

# 1. Loss curve
axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Accuracy curve
axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 3. AUC curve
axes[1, 0].plot(epochs, history['val_auc'], 'g-', label='Val AUC', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('AUC', fontsize=12)
axes[1, 0].set_title('Validation AUC', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 4. AUPR curve
axes[1, 1].plot(epochs, history['val_aupr'], 'm-', label='Val AUPR', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('AUPR', fontsize=12)
axes[1, 1].set_title('Validation AUPR', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('image/training_history.png', dpi=300, bbox_inches='tight')
print(f"Training history plot saved to: image/training_history.png")

# Create separate detailed plots
# Standalone loss plot
plt.figure(figsize=(12, 6))
plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('image/loss_curve.png', dpi=300, bbox_inches='tight')
print(f"Loss curve saved to: image/loss_curve.png")

# Standalone accuracy plot
plt.figure(figsize=(12, 6))
plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=3)
plt.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2, marker='s', markersize=3)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('image/accuracy_curve.png', dpi=300, bbox_inches='tight')
print(f"Accuracy curve saved to: image/accuracy_curve.png")

# Combined AUC and AUPR plot
plt.figure(figsize=(12, 6))
plt.plot(epochs, history['val_auc'], 'g-', label='Val AUC', linewidth=2, marker='o', markersize=3)
plt.plot(epochs, history['val_aupr'], 'm-', label='Val AUPR', linewidth=2, marker='s', markersize=3)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Validation AUC and AUPR', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('image/auc_aupr_curve.png', dpi=300, bbox_inches='tight')
print(f"AUC and AUPR curve saved to: image/auc_aupr_curve.png")

# Print statistics
print("\n=== Training Statistics ===")
print(f"Total epochs: {len(history['train_loss'])}")
print(f"\nBest train loss: {min(history['train_loss']):.6f} (Epoch {history['train_loss'].index(min(history['train_loss'])) + 1})")
print(f"Best val loss: {min(history['val_loss']):.6f} (Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")
print(f"\nBest train accuracy: {max(history['train_acc']):.6f} (Epoch {history['train_acc'].index(max(history['train_acc'])) + 1})")
print(f"Best val accuracy: {max(history['val_acc']):.6f} (Epoch {history['val_acc'].index(max(history['val_acc'])) + 1})")
print(f"\nBest val AUC: {max(history['val_auc']):.6f} (Epoch {history['val_auc'].index(max(history['val_auc'])) + 1})")
print(f"Best val AUPR: {max(history['val_aupr']):.6f} (Epoch {history['val_aupr'].index(max(history['val_aupr'])) + 1})")
print(f"\nFinal epoch (Epoch {len(history['train_loss'])}):")
print(f"  Train Loss: {history['train_loss'][-1]:.6f}, Train Acc: {history['train_acc'][-1]:.6f}")
print(f"  Val Loss: {history['val_loss'][-1]:.6f}, Val Acc: {history['val_acc'][-1]:.6f}")
print(f"  Val AUC: {history['val_auc'][-1]:.6f}, Val AUPR: {history['val_aupr'][-1]:.6f}")
