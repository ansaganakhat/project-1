from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
METRICS_DIR = OUTPUT_DIR / "metrics"
PRED_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = OUTPUT_DIR / "models"
RUNTIME_DIR = OUTPUT_DIR / "runtime"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

for directory in (
    OUTPUT_DIR,
    EDA_DIR,
    METRICS_DIR,
    PRED_DIR,
    MODELS_DIR,
    RUNTIME_DIR,
    TEST_IMAGES_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelDatasetSpec:
    name: str
    model_root: Path
    weights_dir: Path
    dataset_root: Path
    images_root: Path
    explicit_train_images: Path | None = None
    explicit_val_images: Path | None = None
    preferred_yaml_names: tuple[str, ...] = ("data.yaml", "data2.yaml")


YOLOV10_SPEC = ModelDatasetSpec(
    name="yolov10",
    model_root=Path(r"C:\Users\Ansagan\Documents\Ansagan\Ansagan\yolov10"),
    weights_dir=Path(
        r"C:\Users\Ansagan\Documents\Ansagan\Ansagan\yolov10"
        r"\runs\detect\runs\runs\detect\yolov10x_custom\weights"
    ),
    dataset_root=Path(r"C:\Users\Ansagan\Documents\Ansagan\Ansagan\dataset"),
    images_root=Path(r"C:\Users\Ansagan\Documents\Ansagan\Ansagan\dataset\images"),
    explicit_train_images=Path(
        r"C:\Users\Ansagan\Documents\Ansagan\Ansagan\dataset\images\train"
    ),
    explicit_val_images=Path(
        r"C:\Users\Ansagan\Documents\Ansagan\Ansagan\dataset\images\val"
    ),
    preferred_yaml_names=("data.yaml", "data2.yaml"),
)

YOLOV5_SPEC = ModelDatasetSpec(
    name="yolov5",
    model_root=Path(r"C:\Users\Ansagan\Documents\Project\yolov5"),
    weights_dir=Path(r"C:\Users\Ansagan\Documents\Project\yolov5\runs\train\exp\weights"),
    dataset_root=Path(r"C:\Users\Ansagan\Documents\Project\dataset"),
    images_root=Path(r"C:\Users\Ansagan\Documents\Project\dataset\images"),
    preferred_yaml_names=("data2.yaml", "data.yaml"),
)


def find_best_pt(weights_dir: str | Path) -> Path:
    weights_dir = Path(weights_dir)
    preferred = weights_dir / "best.pt"
    if preferred.is_file():
        return preferred.resolve()
    checkpoints = sorted(weights_dir.glob("*.pt")) if weights_dir.is_dir() else []
    if checkpoints:
        return checkpoints[0].resolve()
    raise FileNotFoundError(
        "Модель checkpoint табылмады. Тексерілетін папка:\n"
        f"{weights_dir}\n"
        "Папка ішінде best.pt немесе басқа .pt файл болуы керек."
    )


def yolo5_weights() -> Path:
    return find_best_pt(YOLOV5_SPEC.weights_dir)


def yolo10_weights() -> Path:
    return find_best_pt(YOLOV10_SPEC.weights_dir)


def yolo5_data_yaml() -> Path:
    from .runtime_dataset import ensure_runtime_yaml

    return ensure_runtime_yaml(YOLOV5_SPEC)


def yolo10_data_yaml() -> Path:
    from .runtime_dataset import ensure_runtime_yaml

    return ensure_runtime_yaml(YOLOV10_SPEC)


def analysis_data_yaml() -> Path:
    """EDA және HOG+SVM үшін көлемі үлкен YOLOv10 dataset қолданылады."""
    return yolo10_data_yaml()


def print_config_status(prepare_yaml: bool = True) -> None:
    items: list[tuple[str, Path]] = [
        ("Project root", PROJECT_ROOT),
        ("YOLOv5 model root", YOLOV5_SPEC.model_root),
        ("YOLOv5 weights dir", YOLOV5_SPEC.weights_dir),
        ("YOLOv5 dataset root", YOLOV5_SPEC.dataset_root),
        ("YOLOv5 images root", YOLOV5_SPEC.images_root),
        ("YOLOv10 model root", YOLOV10_SPEC.model_root),
        ("YOLOv10 weights dir", YOLOV10_SPEC.weights_dir),
        ("YOLOv10 dataset root", YOLOV10_SPEC.dataset_root),
        ("YOLOv10 train images", YOLOV10_SPEC.explicit_train_images or Path("")),
        ("YOLOv10 val images", YOLOV10_SPEC.explicit_val_images or Path("")),
        ("Test images", TEST_IMAGES_DIR),
    ]
    if prepare_yaml:
        for label, getter in (
            ("YOLOv5 runtime YAML", yolo5_data_yaml),
            ("YOLOv10 runtime YAML", yolo10_data_yaml),
        ):
            try:
                items.append((label, getter()))
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                print(f"\n{label}: ERROR\n{error}")

    print("=" * 96)
    print("SMART VISION CONFIGURATION")
    print("=" * 96)
    for label, path in items:
        exists = path.exists()
        print(f"{label:<28}: {path}")
        print(f"{'':<28}  {'OK' if exists else 'ТАБЫЛМАДЫ'}")
    print("=" * 96)


if __name__ == "__main__":
    print_config_status(prepare_yaml=True)
