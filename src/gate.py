"""CD gate: exit 1 unless metrics.json accuracy >= params evaluate.min_accuracy."""
import json
from pathlib import Path

import yaml


def passes(metrics, min_accuracy):
    return metrics["accuracy"] >= min_accuracy


def main():
    metrics = json.loads(Path("metrics.json").read_text())
    min_accuracy = yaml.safe_load(Path("params.yaml").read_text())["evaluate"]["min_accuracy"]
    ok = passes(metrics, min_accuracy)
    print(f"accuracy={metrics['accuracy']:.4f} min={min_accuracy} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
