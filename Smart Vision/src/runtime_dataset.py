from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .config import ModelDatasetSpec, RUNTIME_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _all_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _detect_split_dirs(spec: ModelDatasetSpec) -> tuple[Path | None, Path | None]:
    train_candidates = [
        spec.explicit_train_images,
        spec.images_root / "train",
        spec.images_root / "training",
    ]
    val_candidates = [
        spec.explicit_val_images,
        spec.images_root / "val",
        spec.images_root / "valid",
        spec.images_root / "validation",
    ]
    train_dir = next((path.resolve() for path in train_candidates if path and path.is_dir()), None)
    val_dir = next((path.resolve() for path in val_candidates if path and path.is_dir()), None)
    return train_dir, val_dir


def _candidate_yaml_files(spec: ModelDatasetSpec) -> list[Path]:
    candidates: list[Path] = []
    search_roots = [spec.model_root, spec.dataset_root]
    for preferred_name in spec.preferred_yaml_names:
        for root in search_roots:
            direct = root / preferred_name
            if direct.is_file():
                candidates.append(direct.resolve())
    recursive: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for pattern in ("*.yaml", "*.yml"):
            for path in root.rglob(pattern):
                lowered = {part.lower() for part in path.parts}
                if any(token in lowered for token in {"runs", ".venv", "site-packages"}):
                    continue
                recursive.append(path.resolve())

    preference = {name.lower(): index for index, name in enumerate(spec.preferred_yaml_names)}
    recursive.sort(
        key=lambda path: (
            preference.get(path.name.lower(), 999),
            len(path.relative_to(spec.model_root).parts) if path.is_relative_to(spec.model_root) else 999,
            path.name.lower(),
        )
    )
    candidates.extend(recursive)
    return list(dict.fromkeys(candidates))


def _normalize_names(raw_names: Any) -> list[str] | None:
    if isinstance(raw_names, list) and raw_names:
        return [str(value) for value in raw_names]
    if isinstance(raw_names, dict) and raw_names:
        ordered = sorted(raw_names.items(), key=lambda item: int(item[0]))
        return [str(value) for _, value in ordered]
    return None


def discover_class_names(spec: ModelDatasetSpec) -> tuple[list[str], Path]:
    inspected: list[str] = []
    for yaml_path in _candidate_yaml_files(spec):
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        names = _normalize_names(data.get("names"))
        nc = data.get("nc")
        inspected.append(str(yaml_path))
        if names and (nc is None or int(nc) == len(names)):
            return names, yaml_path

    inspected_text = "\n".join(f"  - {item}" for item in inspected) or "  (YAML табылмады)"
    raise FileNotFoundError(
        f"{spec.name}: класс атаулары бар data.yaml табылмады.\n"
        "Модель оқытылған кездегі YAML файлында names: бөлімі болуы қажет.\n"
        f"Тексерілген YAML файлдары:\n{inspected_text}"
    )


def _write_image_list(path: Path, images: list[Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(image.as_posix() for image in images) + "\n", encoding="utf-8")
    return path.resolve()


def _stable_train_val_split(images: list[Path], val_fraction: float = 0.2) -> tuple[list[Path], list[Path]]:
    if len(images) < 2:
        raise ValueError("Train/val бөлу үшін кемінде 2 сурет қажет.")
    ordered = sorted(
        images,
        key=lambda path: hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest(),
    )
    val_count = max(1, int(round(len(ordered) * val_fraction)))
    val_count = min(val_count, len(ordered) - 1)
    return ordered[val_count:], ordered[:val_count]


def resolve_train_val_sources(spec: ModelDatasetSpec) -> tuple[Path, Path, dict[str, Any]]:
    if not spec.images_root.is_dir():
        raise FileNotFoundError(
            f"{spec.name}: images папкасы табылмады:\n{spec.images_root}"
        )

    train_dir, val_dir = _detect_split_dirs(spec)
    metadata: dict[str, Any] = {"mode": "directories"}

    if train_dir and val_dir:
        train_count = len(_all_images(train_dir))
        val_count = len(_all_images(val_dir))
        if train_count == 0 or val_count == 0:
            raise ValueError(
                f"{spec.name}: train/val папкалары бар, бірақ суреттер табылмады.\n"
                f"train={train_dir} ({train_count})\nval={val_dir} ({val_count})"
            )
        metadata.update(train_images=train_count, val_images=val_count)
        return train_dir, val_dir, metadata

    source_dir = train_dir or spec.images_root
    images = _all_images(source_dir)
    if not images:
        raise ValueError(f"{spec.name}: сурет табылмады: {source_dir}")

    train_images, val_images = _stable_train_val_split(images)
    lists_dir = RUNTIME_DIR / spec.name / "lists"
    train_txt = _write_image_list(lists_dir / "train.txt", train_images)
    val_txt = _write_image_list(lists_dir / "val.txt", val_images)
    metadata.update(
        mode="deterministic_80_20_lists",
        source=str(source_dir),
        train_images=len(train_images),
        val_images=len(val_images),
    )
    return train_txt, val_txt, metadata


def ensure_runtime_yaml(spec: ModelDatasetSpec, force: bool = False) -> Path:
    output_path = RUNTIME_DIR / spec.name / "data_runtime.yaml"
    metadata_path = RUNTIME_DIR / spec.name / "runtime_metadata.yaml"
    if output_path.is_file() and not force:
        return output_path.resolve()

    names, source_yaml = discover_class_names(spec)
    train_source, val_source, split_metadata = resolve_train_val_sources(spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runtime_data = {
        "train": train_source.as_posix(),
        "val": val_source.as_posix(),
        "test": val_source.as_posix(),
        "nc": len(names),
        "names": names,
    }
    output_path.write_text(
        yaml.safe_dump(runtime_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    metadata = {
        "model": spec.name,
        "source_class_yaml": source_yaml.as_posix(),
        "dataset_root": spec.dataset_root.as_posix(),
        "images_root": spec.images_root.as_posix(),
        **split_metadata,
    }
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path.resolve()


def prepare_all_runtime_yamls(specs: list[ModelDatasetSpec], force: bool = False) -> dict[str, Path]:
    return {spec.name: ensure_runtime_yaml(spec, force=force) for spec in specs}
