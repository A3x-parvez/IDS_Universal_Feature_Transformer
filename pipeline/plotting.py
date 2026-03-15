import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve


def generate_all_plots(history, labels, probs, graphs_dir):

    os.makedirs(graphs_dir, exist_ok=True)

    # ---------------------------------
    # Training curves
    # ---------------------------------
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(history["train_f1"], label="Train F1")
    plt.plot(history["val_f1"], label="Val F1")
    plt.legend()
    plt.title("F1 Curve")

    plt.savefig(os.path.join(graphs_dir, "training_curves.svg"))
    plt.close()

    # ---------------------------------
    # ROC curve
    # ---------------------------------
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")

    plt.savefig(os.path.join(graphs_dir, "roc_curve.svg"))
    plt.close()

    # ---------------------------------
    # Precision Recall
    # ---------------------------------
    prec_vals, rec_vals, _ = precision_recall_curve(labels, probs)

    plt.figure()
    plt.plot(rec_vals, prec_vals)
    plt.title("Precision Recall")

    plt.savefig(os.path.join(graphs_dir, "precision_recall.svg"))
    plt.close()

    # ---------------------------------
    # Confusion Matrix
    # ---------------------------------
    preds = [1 if p>0.5 else 0 for p in probs]
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

    plt.savefig(os.path.join(graphs_dir, "confusion_matrix.svg"))
    plt.close()