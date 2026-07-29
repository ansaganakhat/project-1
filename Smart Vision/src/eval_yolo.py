from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _float_or_none(value: Any) -> float | None:
    """Мәнді float форматына келтіреді."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def model_file_size_mb(weights: str | Path) -> float:
    """Checkpoint көлемін мегабайтпен қайтарады."""
    weights_path = Path(weights).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint табылмады:\n{weights_path}"
        )

    return round(
        weights_path.stat().st_size / (1024**2),
        3,
    )


def eval_with_ultralytics(
    weights: str | Path,
    data_yaml: str | Path,
    imgsz: int = 640,
    conf: float = 0.001,
    device: int | str | None = None,
) -> dict[str, Any]:
    """
    Ultralytics форматындағы модельді бағалайды.

    Ескерту:
    Бұл функцияны original YOLOv5 repository ішінде
    оқытылған legacy checkpoint үшін қолданбау керек.
    """
    from ultralytics import YOLO

    weights_path = Path(weights).resolve()
    data_yaml_path = Path(data_yaml).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint табылмады:\n{weights_path}"
        )

    if not data_yaml_path.is_file():
        raise FileNotFoundError(
            f"Runtime YAML табылмады:\n{data_yaml_path}"
        )

    model = YOLO(str(weights_path))

    validation_arguments: dict[str, Any] = {
        "data": str(data_yaml_path),
        "imgsz": imgsz,
        "conf": conf,
        "split": "val",
        "plots": True,
        "verbose": True,
        "batch": 1,
        "workers": 0,
    }

    if device is not None:
        validation_arguments["device"] = device

    start_time = time.perf_counter()

    validation_result = model.val(
        **validation_arguments
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    speed = (
        getattr(validation_result, "speed", {})
        or {}
    )

    inference_ms = _float_or_none(
        speed.get("inference")
    )

    fps = (
        1000.0 / inference_ms
        if inference_ms is not None
        and inference_ms > 0
        else None
    )

    parameter_count: int | None = None

    try:
        parameter_count = int(
            sum(
                parameter.numel()
                for parameter
                in model.model.parameters()
            )
        )
    except (AttributeError, TypeError):
        pass

    model_names = (
        getattr(model, "names", {})
        or {}
    )

    return {
        "weights": str(weights_path),
        "data_yaml": str(data_yaml_path),
        "imgsz": imgsz,
        "conf": conf,
        "eval_seconds": float(
            elapsed_seconds
        ),
        "model_size_mb": model_file_size_mb(
            weights_path
        ),
        "parameter_count": parameter_count,
        "class_count": len(model_names),
        "precision_mean": _float_or_none(
            getattr(
                validation_result.box,
                "mp",
                None,
            )
        ),
        "recall_mean": _float_or_none(
            getattr(
                validation_result.box,
                "mr",
                None,
            )
        ),
        "map50": _float_or_none(
            getattr(
                validation_result.box,
                "map50",
                None,
            )
        ),
        "map50_95": _float_or_none(
            getattr(
                validation_result.box,
                "map",
                None,
            )
        ),
        "preprocess_ms": _float_or_none(
            speed.get("preprocess")
        ),
        "inference_ms": inference_ms,
        "postprocess_ms": _float_or_none(
            speed.get("postprocess")
        ),
        "fps_estimate": _float_or_none(fps),
        "save_dir": str(
            getattr(
                validation_result,
                "save_dir",
                "",
            )
        ),
    }


def _validate_numpy_for_yolov5() -> None:
    """
    YOLOv5 метрикаларын есептеуге қажетті
    NumPy интеграция функцияларын тексереді.

    Ескі NumPy:
        np.trapz

    Жаңа NumPy:
        np.trapezoid
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "YOLOv5 бағалауы үшін NumPy "
            "импортталмады."
        ) from exc

    if hasattr(np, "trapz"):
        return

    if hasattr(np, "trapezoid"):
        return

    numpy_location = getattr(
        np,
        "__file__",
        "unknown",
    )

    raise RuntimeError(
        "NumPy ішінде trapz және trapezoid "
        "функциялары табылмады.\n"
        f"numpy.__file__ = {numpy_location}"
    )


def _patch_yolov5_numpy_compatibility(
    yolov5_repo: str | Path,
) -> None:
    """
    Ескі YOLOv5 кодындағы np.trapz шақыруын
    жаңа NumPy үшін np.trapezoid-қа ауыстырады.

    Бастапқы metrics.py файлы бір рет backup
    ретінде сақталады.
    """
    import numpy as np

    if hasattr(np, "trapz"):
        return

    if not hasattr(np, "trapezoid"):
        raise RuntimeError(
            "NumPy ішінде trapz және trapezoid "
            "функциялары табылмады."
        )

    repository_path = Path(
        yolov5_repo
    ).resolve()

    metrics_path = (
        repository_path
        / "utils"
        / "metrics.py"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            "YOLOv5 metrics.py табылмады:\n"
            f"{metrics_path}"
        )

    source_code = metrics_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    # Код бұрын түзетілген болса, қайта өзгертпейміз.
    if "np.trapz(" not in source_code:
        return

    backup_path = metrics_path.with_name(
        "metrics_original_backup.py"
    )

    if not backup_path.exists():
        backup_path.write_text(
            source_code,
            encoding="utf-8",
        )

    corrected_source = source_code.replace(
        "np.trapz(",
        "np.trapezoid(",
    )

    metrics_path.write_text(
        corrected_source,
        encoding="utf-8",
    )

    print(
        "YOLOv5 NumPy compatibility түзетілді:"
    )
    print(metrics_path)


def _build_yolov5_environment() -> dict[str, str]:
    """
    Ескі YOLOv5 checkpoint-ін жаңа PyTorch-та
    жүктеуге арналған subprocess environment жасайды.
    """
    process_environment = os.environ.copy()

    # torch.load(weights_only=False) режимін мәжбүрлейді.
    # Тек сенімді, өзің оқытқан checkpoint үшін қолдану керек.
    process_environment[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    # Windows терминалындағы UTF-8 output үшін.
    process_environment[
        "PYTHONIOENCODING"
    ] = "utf-8"

    return process_environment


def _stream_process(
    command: list[str],
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> tuple[int, str]:
    """
    Subprocess output-ын бір уақытта терминалға шығарып,
    толық мәтінді кейінгі талдау үшін жинайды.
    """
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    output_lines: list[str] = []

    if process.stdout is None:
        process.kill()

        raise RuntimeError(
            "YOLOv5 subprocess output "
            "оқылмады."
        )

    for line in process.stdout:
        print(
            line,
            end="",
            flush=True,
        )
        output_lines.append(line)

    return_code = process.wait()
    full_output = "".join(output_lines)

    return return_code, full_output


def _remove_ansi_codes(text: str) -> str:
    """Терминалдың ANSI түстерін мәтіннен алып тастайды."""
    ansi_pattern = re.compile(
        r"\x1b\[[0-9;?]*[A-Za-z]"
    )

    return ansi_pattern.sub(
        "",
        text,
    )


def _parse_yolov5_all_line(
    text: str,
) -> dict[str, float | None]:
    """
    YOLOv5 console output ішіндегі 'all' жолынан
    Precision, Recall, mAP50 және mAP50-95 алады.
    """
    cleaned_text = _remove_ansi_codes(text)

    for line in reversed(
        cleaned_text.splitlines()
    ):
        stripped_line = line.strip()

        if not stripped_line.lower().startswith(
            "all"
        ):
            continue

        numeric_values: list[float] = []

        for token in stripped_line.split()[1:]:
            try:
                numeric_values.append(
                    float(token)
                )
            except ValueError:
                continue

        # YOLOv5 all жолы:
        # all Images Instances P R mAP50 mAP50-95
        if len(numeric_values) >= 6:
            return {
                "images": int(
                    numeric_values[-6]
                ),
                "instances": int(
                    numeric_values[-5]
                ),
                "precision_mean": (
                    numeric_values[-4]
                ),
                "recall_mean": (
                    numeric_values[-3]
                ),
                "map50": numeric_values[-2],
                "map50_95": (
                    numeric_values[-1]
                ),
            }

    return {
        "images": None,
        "instances": None,
        "precision_mean": None,
        "recall_mean": None,
        "map50": None,
        "map50_95": None,
    }


def eval_yolov5_repo(
    yolov5_repo: str | Path,
    weights: str | Path,
    data_yaml: str | Path,
    imgsz: int = 640,
    device: str = "0",
) -> dict[str, Any]:
    """
    Original YOLOv5 repository ішіндегі val.py
    арқылы модельді бағалайды.
    """
    _validate_numpy_for_yolov5()

    repository_path = Path(
        yolov5_repo
    ).resolve()

    weights_path = Path(
        weights
    ).resolve()

    data_yaml_path = Path(
        data_yaml
    ).resolve()

    validation_script = (
        repository_path
        / "val.py"
    )

    if not repository_path.is_dir():
        raise FileNotFoundError(
            "YOLOv5 repository табылмады:\n"
            f"{repository_path}"
        )

    if not validation_script.is_file():
        raise FileNotFoundError(
            "Original YOLOv5 val.py табылмады:\n"
            f"{validation_script}"
        )

    if not weights_path.is_file():
        raise FileNotFoundError(
            "YOLOv5 checkpoint табылмады:\n"
            f"{weights_path}"
        )

    if not data_yaml_path.is_file():
        raise FileNotFoundError(
            "YOLOv5 runtime YAML табылмады:\n"
            f"{data_yaml_path}"
        )

    _patch_yolov5_numpy_compatibility(
        repository_path
    )

    command = [
        sys.executable,
        str(validation_script),
        "--weights",
        str(weights_path),
        "--data",
        str(data_yaml_path),
        "--imgsz",
        str(imgsz),
        "--task",
        "val",
        "--device",
        str(device),
        "--batch-size",
        "1",
        "--workers",
        "0",
        "--verbose",
    ]

    print("\nYOLOv5 validation командасы:")
    print(
        subprocess.list2cmdline(command)
    )
    print()

    start_time = time.perf_counter()

    return_code, console_output = (
        _stream_process(
            command=command,
            cwd=repository_path,
            environment=(
                _build_yolov5_environment()
            ),
        )
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    parsed_metrics = (
        _parse_yolov5_all_line(
            console_output
        )
    )

    result: dict[str, Any] = {
        "weights": str(weights_path),
        "data_yaml": str(data_yaml_path),
        "imgsz": imgsz,
        "device": str(device),
        "batch_size": 1,
        "workers": 0,
        "eval_seconds": float(
            elapsed_seconds
        ),
        "model_size_mb": model_file_size_mb(
            weights_path
        ),
        "returncode": int(return_code),
        **parsed_metrics,
        "console_tail": console_output[-8000:],
    }

    if return_code != 0:
        error_tail = console_output[-4000:]

        raise RuntimeError(
            "YOLOv5 val.py қатемен аяқталды.\n"
            "Соңғы output:\n"
            f"{error_tail}"
        )

    if result["map50"] is None:
        print(
            "Ескерту: validation аяқталды, бірақ "
            "'all' метрика жолы автоматты "
            "түрде оқылмады."
        )

    return result


def save_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Метрикаларды JSON файлына сақтайды."""
    output_file = Path(
        output_path
    ).resolve()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Метрикалар сақталды:\n{output_file}"
    )