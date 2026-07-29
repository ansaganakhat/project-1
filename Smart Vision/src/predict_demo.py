from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import PRED_DIR, YOLOV5_SPEC, yolo5_weights, yolo10_weights


def _normalize_source(source: str | Path | int) -> str | int:
    if isinstance(source, int):
        return source
    text = str(source).strip()
    return int(text) if text.isdigit() else text


def _validate_source(source: str | int) -> None:
    if isinstance(source, int) or source.startswith(("http://", "https://", "rtsp://")):
        return
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Test source табылмады:\n{path}")


def _ultralytics_predict(
    weights: Path,
    model_name: str,
    source: str | int,
    conf: float,
    imgsz: int,
    save: bool,
    device: int | str | None,
) -> list[Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs: dict[str, Any] = {
        "source": source,
        "conf": conf,
        "imgsz": imgsz,
        "save": save,
        "project": str(PRED_DIR),
        "name": f"{model_name}_demo",
        "exist_ok": True,
        "stream": False,
        "verbose": True,
    }
    if device is not None:
        kwargs["device"] = device
    return list(model.predict(**kwargs))


def _yolov5_repo_predict(
    source: str | int,
    conf: float,
    imgsz: int,
    device: int | str | None,
) -> list[Any]:
    detect_script = YOLOV5_SPEC.model_root / "detect.py"
    if not detect_script.is_file():
        raise FileNotFoundError(f"YOLOv5 detect.py табылмады:\n{detect_script}")
    command = [
        sys.executable,
        str(detect_script),
        "--weights",
        str(yolo5_weights()),
        "--source",
        str(source),
        "--conf-thres",
        str(conf),
        "--imgsz",
        str(imgsz),
        "--project",
        str(PRED_DIR),
        "--name",
        "yolov5_demo",
        "--exist-ok",
    ]
    if device is not None:
        command.extend(["--device", str(device)])
    process = subprocess.run(command, cwd=str(YOLOV5_SPEC.model_root), check=False)
    if process.returncode != 0:
        raise RuntimeError(f"YOLOv5 detect.py қатемен аяқталды: returncode={process.returncode}")
    print(f"YOLOv5 нәтижелері сақталды: {PRED_DIR / 'yolov5_demo'}")
    return []


def predict_source(
    model_name: str,
    source: str | Path | int,
    conf: float = 0.25,
    imgsz: int = 640,
    save: bool = True,
    device: int | str | None = None,
) -> list[Any]:
    normalized_name = model_name.strip().lower()
    normalized_source = _normalize_source(source)
    _validate_source(normalized_source)

    if normalized_name in {"yolov10", "v10", "10"}:
        return _ultralytics_predict(
            yolo10_weights(), "yolov10", normalized_source, conf, imgsz, save, device
        )
    if normalized_name in {"yolov5", "v5", "5"}:
        try:
            return _ultralytics_predict(
                yolo5_weights(), "yolov5", normalized_source, conf, imgsz, save, device
            )
        except (RuntimeError, TypeError, AttributeError, ModuleNotFoundError) as error:
            print(f"Ultralytics YOLOv5 checkpoint-ті аша алмады: {error}")
            print("Original YOLOv5 detect.py қолданылады.")
            return _yolov5_repo_predict(normalized_source, conf, imgsz, device)
    raise ValueError("model_name 'yolov5' немесе 'yolov10' болуы керек.")
