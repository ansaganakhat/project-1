from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGE_COLUMNS = [
    "split", "image_path", "label_path", "width", "height", "brightness",
    "contrast", "objects_count", "has_label",
]
BOX_COLUMNS = [
    "split", "image_path", "label_path", "class_id", "class_name", "x_center",
    "y_center", "bbox_w", "bbox_h", "bbox_area", "bbox_aspect_ratio",
    "brightness", "contrast",
]


def _expand(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def load_data_yaml(data_yaml: str | Path) -> dict:
    yaml_path = _expand(data_yaml).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml табылмады:\n{yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config["yaml_path"] = str(yaml_path)
    config["yaml_dir"] = str(yaml_path.parent)
    raw_root = config.get("path")
    root = yaml_path.parent if raw_root in (None, "") else _expand(raw_root)
    if not root.is_absolute():
        root = yaml_path.parent / root
    config["dataset_root"] = str(root.resolve())
    return config


def _resolve_entries(config: dict, split: str) -> list[Path]:
    value = config.get(split)
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    root = Path(config["dataset_root"])
    yaml_dir = Path(config["yaml_dir"])
    resolved: list[Path] = []
    for item in values:
        path = _expand(item)
        if path.is_absolute():
            resolved.append(path.resolve())
            continue
        root_candidate = (root / path).resolve()
        yaml_candidate = (yaml_dir / path).resolve()
        resolved.append(root_candidate if root_candidate.exists() else yaml_candidate)
    return resolved


def _paths_from_txt(list_file: Path, config: dict) -> list[Path]:
    root = Path(config["dataset_root"])
    yaml_dir = Path(config["yaml_dir"])
    paths: list[Path] = []
    for raw_line in list_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip().strip('"').strip("'")
        if not line:
            continue
        path = _expand(line)
        if not path.is_absolute():
            candidates = [list_file.parent / path, root / path, yaml_dir / path]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path.resolve())
    return paths


def collect_images(entries: Iterable[Path], config: dict) -> list[Path]:
    images: list[Path] = []
    for entry in entries:
        if not entry.exists():
            continue
        if entry.is_file() and entry.suffix.lower() == ".txt":
            images.extend(_paths_from_txt(entry, config))
        elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry.resolve())
        elif entry.is_dir():
            images.extend(
                path.resolve()
                for path in entry.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    return sorted(dict.fromkeys(images))


def _replace_images_with_labels(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" not in lowered:
        return None
    index = len(parts) - 1 - lowered[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def image_to_label_path(image_path: str | Path, split: str | None = None) -> Path:
    image_path = Path(image_path)
    candidates: list[Path] = []
    replaced = _replace_images_with_labels(image_path)
    if replaced is not None:
        candidates.append(replaced)
    if len(image_path.parents) >= 2:
        candidates.append(
            image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        )
    if split and len(image_path.parents) >= 3:
        candidates.append(image_path.parents[2] / "labels" / split / f"{image_path.stem}.txt")
    candidates.append(image_path.parent / f"{image_path.stem}.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def read_yolo_label(label_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    label_path = Path(label_path)
    rows: list[tuple[int, float, float, float, float]] = []
    if not label_path.is_file():
        return rows
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append((int(float(parts[0])), *(float(value) for value in parts[1:5])))
        except ValueError:
            continue
    return rows


def get_class_names(config: dict) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {}


def dataset_diagnostics(data_yaml: str | Path) -> pd.DataFrame:
    config = load_data_yaml(data_yaml)
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        entries = _resolve_entries(config, split)
        images = collect_images(entries, config)
        label_files = 0
        objects = 0
        missing_examples: list[str] = []
        for image_path in images:
            label_path = image_to_label_path(image_path, split=split)
            if label_path.is_file():
                label_files += 1
            elif len(missing_examples) < 3:
                missing_examples.append(str(label_path))
            objects += len(read_yolo_label(label_path))
        rows.append(
            {
                "split": split,
                "resolved_path": "; ".join(str(path) for path in entries),
                "path_exists": bool(entries) and all(path.exists() for path in entries),
                "images": len(images),
                "label_files": label_files,
                "objects": objects,
                "missing_label_examples": " | ".join(missing_examples),
            }
        )
    return pd.DataFrame(rows)


def build_image_and_box_tables(
    data_yaml: str | Path,
    include_image_features: bool = True,
    max_images_per_split: int | None = None,
    show_progress: bool = True,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = load_data_yaml(data_yaml)
    names = get_class_names(config)
    image_rows: list[dict] = []
    box_rows: list[dict] = []

    for split in splits:
        images = collect_images(_resolve_entries(config, split), config)
        if max_images_per_split is not None:
            images = images[:max_images_per_split]
        iterator = tqdm(images, desc=f"{split} EDA", unit="image") if show_progress else images
        for image_path in iterator:
            width = height = brightness = contrast = np.nan
            if include_image_features:
                image = cv2.imread(str(image_path))
                if image is not None:
                    height, width = image.shape[:2]
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    brightness = float(np.mean(gray))
                    contrast = float(np.std(gray))
            label_path = image_to_label_path(image_path, split=split)
            labels = read_yolo_label(label_path)
            image_rows.append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "width": width,
                    "height": height,
                    "brightness": brightness,
                    "contrast": contrast,
                    "objects_count": len(labels),
                    "has_label": label_path.is_file(),
                }
            )
            for class_id, x_center, y_center, bbox_w, bbox_h in labels:
                box_rows.append(
                    {
                        "split": split,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "class_id": class_id,
                        "class_name": names.get(class_id, str(class_id)),
                        "x_center": x_center,
                        "y_center": y_center,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "bbox_area": bbox_w * bbox_h,
                        "bbox_aspect_ratio": bbox_w / bbox_h if bbox_h > 0 else np.nan,
                        "brightness": brightness,
                        "contrast": contrast,
                    }
                )

    return (
        pd.DataFrame(image_rows, columns=IMAGE_COLUMNS),
        pd.DataFrame(box_rows, columns=BOX_COLUMNS),
        config,
    )


def save_tables(image_df: pd.DataFrame, box_df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_df.to_csv(output_dir / "image_level_features.csv", index=False)
    box_df.to_csv(output_dir / "bbox_level_features.csv", index=False)
