import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

# =============================================================
# SYNTHETIC CLASSIFICATION DATA
# =============================================================
# 4 disease classes (0,1,2,3)
num_samples = 300
num_classes = 4

# True labels
true_labels = np.random.randint(0, num_classes, num_samples)

# Predicted labels (80–90% accurate)
pred_labels = true_labels.copy()
noise_idx = np.random.choice(num_samples, size=int(0.15 * num_samples))
pred_labels[noise_idx] = np.random.randint(0, num_classes, len(noise_idx))

# For ROC: synthetic probabilities
y_true_bin = label_binarize(true_labels, classes=list(range(num_classes)))
y_pred_prob = np.random.rand(num_samples, num_classes)
y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)

print("Synthetic classification data generated.")


# =============================================================
# FIGURE A — CONFUSION MATRIX
# =============================================================
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Disease Classification — Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("Saved: confusion_matrix.png")


# =============================================================
# FIGURE B — ROC CURVES
# =============================================================
plt.figure(figsize=(6, 5))
for cls in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, cls], y_pred_prob[:, cls])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Disease Classification")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300)
plt.close()
print("Saved: roc_curves.png")


# =============================================================
# FIGURE C — ACCURACY CURVES
# =============================================================
epochs = np.arange(1, 51)
train_acc = 0.5 + 0.45*(1 - np.exp(-0.1*epochs)) + 0.02*np.random.randn(len(epochs))
val_acc = train_acc - 0.05 + 0.02*np.random.randn(len(epochs))

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Classifier Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.close()
print("Saved: accuracy_curve.png")


# =============================================================
# FIGURE D — t-SNE FEATURE CLUSTERING
# =============================================================
# Generate fake feature vectors
feature_dim = 16
X_features = np.random.randn(num_samples, feature_dim)

X_emb = TSNE(n_components=2, perplexity=30).fit_transform(X_features)

plt.figure(figsize=(6, 5))
for cls in range(num_classes):
    idx = true_labels == cls
    plt.scatter(X_emb[idx, 0], X_emb[idx, 1], s=20, alpha=0.7, label=f"Class {cls}")

plt.title("t-SNE Clustering of Learned Disease Features")
plt.legend()
plt.tight_layout()
plt.savefig("tsne_clusters.png", dpi=300)
plt.close()
print("Saved: tsne_clusters.png")

print("ALL CLASSIFICATION FIGURES GENERATED.")
