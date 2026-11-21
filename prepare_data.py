"""Prepare the assignment's stratified Wisconsin diagnostic dataset."""

import argparse
import json
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def prepare_data(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    sample = test.drop(columns="target").iloc[0].tolist()
    (output_dir / "sample_request.json").write_text(
        json.dumps({"features": sample}, indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "samples": int(len(df)),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "n_features": 30,
        "class_mapping": {"0": "malignant", "1": "benign"},
        "overall_class_counts": {str(k): int(v) for k, v in df["target"].value_counts().items()},
        "train_class_counts": {str(k): int(v) for k, v in train["target"].value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in test["target"].value_counts().items()},
        "feature_names": list(data.feature_names),
        "random_state": 42,
        "stratified": True,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    args = parser.parse_args()
    prepare_data(args.output_dir)

