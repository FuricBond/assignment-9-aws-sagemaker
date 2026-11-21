"""SageMaker scikit-learn script-mode training AND inference contract.

The model is *not* a clinical diagnostic instrument. This is coursework only.
"""

import argparse
import io
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


N_FEATURES = 30


def model_fn(model_dir):
    """Called when the managed SageMaker endpoint starts."""
    return joblib.load(Path(model_dir) / "model.joblib")


def input_fn(request_body, content_type):
    """Accept exactly one row (30 numeric inputs) in text/csv or JSON."""
    body = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    if content_type.split(";")[0].strip().lower() == "text/csv":
        array = pd.read_csv(io.StringIO(body), header=None).to_numpy()
    elif content_type.split(";")[0].strip().lower() == "application/json":
        parsed = json.loads(body)
        array = np.asarray(parsed.get("features") if isinstance(parsed, dict) else parsed)
    else:
        raise ValueError("Content-Type must be text/csv or application/json")
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape != (1, N_FEATURES) or not np.isfinite(array).all():
        raise ValueError("Expected exactly one row of 30 finite numeric features")
    return array


def predict_fn(input_data, model):
    if hasattr(model, "feature_names_in_"):
        input_data = pd.DataFrame(input_data, columns=list(model.feature_names_in_))
    return model.predict(input_data)


def output_fn(prediction, accept):
    if accept.split(";")[0].strip().lower() not in ("application/json", "*/*"):
        raise ValueError("Accept must be application/json")
    return json.dumps([int(value) for value in prediction]), "application/json"


def run_training(train_dir, test_dir, model_dir, output_dir, n_estimators, max_depth):
    start = time.perf_counter()
    train = pd.read_csv(Path(train_dir) / "train.csv")
    test = pd.read_csv(Path(test_dir) / "test.csv")
    if "target" not in train or "target" not in test:
        raise ValueError("Both CSV files must contain the target column")
    feature_names = [column for column in train.columns if column != "target"]
    if len(feature_names) != N_FEATURES or list(test.drop(columns="target").columns) != feature_names:
        raise ValueError("Expected exactly the same 30 features in both data files")
    x_train, y_train = train[feature_names], train["target"]
    x_test, y_test = test[feature_names], test["target"]
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    probability_benign = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, pos_label=1)),
        "recall": float(recall_score(y_test, predicted, pos_label=1)),
        "f1": float(f1_score(y_test, predicted, pos_label=1)),
        "auc": float(roc_auc_score(y_test, probability_benign)),
        "malignant_recall": float(recall_score(y_test, predicted, pos_label=0)),
        "false_benign_count": int(((y_test == 0) & (predicted == 1)).sum()),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "training_wall_seconds": round(time.perf_counter() - start, 3),
    }
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "actual": y_test.astype(int),
            "predicted": predicted.astype(int),
            "probability_benign": probability_benign,
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_malignant", "actual_benign"],
        columns=["pred_malignant", "pred_benign"],
    ).to_csv(output_dir / "confusion_matrix.csv")
    # Each line can be captured by the SageMaker SDK metric_definitions regex.
    for key in ["accuracy", "precision", "recall", "f1", "auc", "malignant_recall"]:
        print(f"test_{key}={metrics[key]:.4f}", flush=True)
    print("test_metrics_json=" + json.dumps(metrics, sort_keys=True), flush=True)
    print("confusion_matrix_true_rows_0_1=" + str(matrix.tolist()), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "artifacts/model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "artifacts/data"))
    parser.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", "artifacts/data"))
    parser.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "artifacts/results"))
    args = parser.parse_args()
    run_training(
        args.train,
        args.test,
        args.model_dir,
        args.output_dir,
        args.n_estimators,
        args.max_depth,
    )

