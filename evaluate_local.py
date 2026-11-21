"""Run reproducible local comparisons and produce genuine result figures.

These figures are NOT AWS Console screenshots or proof of a managed job.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc

from prepare_data import prepare_data


CONFIGURATIONS = [(100, 4), (200, 8)]


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    prepare_data(data_dir)
    records = []
    experiments = []
    for n_estimators, max_depth in CONFIGURATIONS:
        name = f"rf_{n_estimators}_depth_{max_depth}"
        destination = output_dir / name
        command = [
            sys.executable,
            str(Path(__file__).parent / "train.py"),
            "--train", str(data_dir), "--test", str(data_dir),
            "--model-dir", str(destination / "model"),
            "--output-dir", str(destination / "results"),
            "--n-estimators", str(n_estimators), "--max-depth", str(max_depth),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        (destination / "training_log.txt").write_text(completed.stdout, encoding="utf-8")
        metrics = json.loads((destination / "results" / "metrics.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(destination / "results" / "predictions.csv")
        records.append({"configuration": name, **metrics})
        experiments.append((name, predictions))
        print(name)
        print(completed.stdout)

    comparison = pd.DataFrame(records)
    comparison.to_csv(output_dir / "hyperparameter_comparison.csv", index=False)
    # Accuracy-first choice for coursework; malignant recall also reported separately.
    best_name = str(comparison.sort_values(["accuracy", "auc"], ascending=False).iloc[0]["configuration"])
    (output_dir / "best_configuration.json").write_text(
        json.dumps({"selected": best_name, "selection_method": "highest accuracy, then ROC-AUC"}, indent=2),
        encoding="utf-8",
    )

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for ax, (name, prediction) in zip(axes, experiments):
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix(prediction.actual, prediction.predicted, labels=[0, 1])
        disp = ConfusionMatrixDisplay(matrix, display_labels=["Malignant (0)", "Benign (1)"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name.replace("_", " "))
    fig.suptitle("Test-set confusion matrices (114 independent test cases)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, prediction in experiments:
        fpr, tpr, _ = roc_curve(prediction.actual, prediction.probability_benign)
        ax.plot(fpr, tpr, linewidth=2.5, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(xlabel="False positive rate (benign class)", ylabel="True positive rate (benign class)")
    ax.set_title("Random Forest ROC curves on stratified test split")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    keys = ["accuracy", "precision", "recall", "f1", "auc", "malignant_recall"]
    x = np.arange(len(keys))
    width = 0.37
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for i, record in enumerate(records):
        values = [record[key] for key in keys]
        ax.bar(x + (i - 0.5) * width, values, width, label=record["configuration"])
        for xpos, value in zip(x + (i - 0.5) * width, values):
            ax.annotate(f"{value:.3f}", (xpos, value), ha="center", va="bottom", fontsize=7, rotation=90, xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(x, ["Accuracy", "Precision\n(benign)", "Recall\n(benign)", "F1", "ROC-AUC", "Recall\n(malignant)"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Test evaluation - two requested hyperparameter settings")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Local evidence generated at {output_dir.resolve()}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    run(parser.parse_args().output_dir)

