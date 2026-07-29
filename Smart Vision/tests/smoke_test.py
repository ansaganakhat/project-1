from __future__ import annotations

import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from src.classical_baseline import train_hog_svm
from src.config import ModelDatasetSpec
from src.runtime_dataset import ensure_runtime_yaml
from src.yolo_dataset_utils import build_image_and_box_tables, dataset_diagnostics


def make_image(path: Path, index: int) -> None:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (45, 45), (255, 255, 255), -1)
    cv2.circle(image, (70, 70), 15 + index % 3, (160, 160, 160), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_root = root / "model"
        weights_dir = model_root / "runs" / "weights"
        dataset_root = root / "dataset"
        for split in ("train", "val"):
            for index in range(12):
                image_path = dataset_root / "images" / split / f"{split}_{index}.jpg"
                label_path = dataset_root / "labels" / split / f"{split}_{index}.txt"
                make_image(image_path, index)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    "0 0.30 0.30 0.35 0.35\n1 0.72 0.72 0.30 0.30\n",
                    encoding="utf-8",
                )
        model_root.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)
        (model_root / "data.yaml").write_text(
            yaml.safe_dump({"nc": 2, "names": ["square", "circle"]}, sort_keys=False),
            encoding="utf-8",
        )
        spec = ModelDatasetSpec(
            name="mock",
            model_root=model_root,
            weights_dir=weights_dir,
            dataset_root=dataset_root,
            images_root=dataset_root / "images",
            explicit_train_images=dataset_root / "images" / "train",
            explicit_val_images=dataset_root / "images" / "val",
        )
        runtime_yaml = ensure_runtime_yaml(spec, force=True)
        diagnostics = dataset_diagnostics(runtime_yaml)
        assert diagnostics["images"].sum() == 36  # test duplicates val by design
        assert diagnostics["objects"].sum() == 72
        image_df, box_df, _ = build_image_and_box_tables(
            runtime_yaml,
            include_image_features=False,
            show_progress=False,
            splits=("train", "val"),
        )
        assert len(image_df) == 24
        assert len(box_df) == 48
        model_path = root / "hog.joblib"
        _, metrics, _ = train_hog_svm(
            runtime_yaml,
            output_model=model_path,
            max_samples_per_class=20,
        )
        assert model_path.is_file()
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
