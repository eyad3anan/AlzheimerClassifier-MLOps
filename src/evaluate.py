"""Stage 3: score the trained model on the test split -> metrics.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.model import CLASSES, build_model
from src.train import MODEL_PATH, make_dataset


def main():
    p = yaml.safe_load(Path("params.yaml").read_text())["train"]
    test_df = pd.read_csv("splits/test.csv")
    model = build_model(dropout=p["dropout"], dense_units=p["dense_units"])
    model.load_weights(MODEL_PATH)

    y_true = np.array([CLASSES.index(label) for label in test_df["label"]])
    y_pred = model.predict(make_dataset(test_df, p["batch_size"], training=False), verbose=0).argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/confusion_matrix.json").write_text(
        json.dumps({"labels": CLASSES, "matrix": cm.tolist()}, indent=2)
    )
    print(metrics)


if __name__ == "__main__":
    main()
