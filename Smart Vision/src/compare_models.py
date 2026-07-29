from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_model_comparison(metrics_dir: str | Path) -> pd.DataFrame:
    metrics_dir = Path(metrics_dir)
    files = {
        "HOG + SVM": metrics_dir / "hog_svm_metrics.json",
        "YOLOv5": metrics_dir / "yolov5_metrics.json",
        "YOLOv10": metrics_dir / "yolov10_metrics.json",
    }
    rows: list[dict] = []
    for model_name, path in files.items():
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model_name,
                "task": data.get("task", "object detection"),
                "dataset": data.get("data_yaml"),
                "class_count": data.get("class_count") or len(data.get("class_names", [])) or None,
                "accuracy": data.get("accuracy"),
                "f1_macro": data.get("f1_macro"),
                "precision": data.get("precision_mean"),
                "recall": data.get("recall_mean"),
                "map50": data.get("map50"),
                "map50_95": data.get("map50_95"),
                "inference_ms": data.get("inference_ms"),
                "fps": data.get("fps_estimate"),
                "model_size_mb": data.get("model_size_mb"),
                "best_params": str(data.get("best_params", "")),
            }
        )
    return pd.DataFrame(rows)
