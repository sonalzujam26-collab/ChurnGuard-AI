import matplotlib.pyplot as plt
import seaborn as sns

from src.model_evaluation import cm, fpr, tpr, roc_auc


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

def plot_confusion_matrix():

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"],
        ax=ax
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.tight_layout()

    return fig


# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

def plot_roc_curve():

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"Logistic Regression (AUC = {roc_auc:.2f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend()

    plt.tight_layout()

    return fig