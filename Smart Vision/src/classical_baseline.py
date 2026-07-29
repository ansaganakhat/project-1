from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from .yolo_dataset_utils import build_image_and_box_tables, dataset_diagnostics, get_class_names

RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 8


def crop_from_yolo(
    image: np.ndarray,
    x_center: float,
    y_center: float,
    bbox_width: float,
    bbox_height: float,
) -> np.ndarray | None:
    image_height, image_width = image.shape[:2]
    x1 = max(0, int((x_center - bbox_width / 2) * image_width))
    y1 = max(0, int((y_center - bbox_height / 2) * image_height))
    x2 = min(image_width, int((x_center + bbox_width / 2) * image_width))
    y2 = min(image_height, int((y_center + bbox_height / 2) * image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def extract_hog_features(crop: np.ndarray, output_size: tuple[int, int] = (96, 96)) -> np.ndarray:
    resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def make_hog_dataset(
    data_yaml: str | Path,
    max_samples_per_class: int = 300,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    _, box_df, config = build_image_and_box_tables(
        data_yaml=data_yaml,
        include_image_features=False,
        show_progress=False,
        splits=("train",),
    )
    if box_df.empty:
        diagnostics = dataset_diagnostics(data_yaml)
        raise ValueError(
            "YOLO label объектілері табылмады. Dataset диагностикасы:\n\n"
            + diagnostics.to_string(index=False)
        )

    features: list[np.ndarray] = []
    targets: list[int] = []
    groups: list[str] = []

    for class_id, class_rows in box_df.groupby("class_id", sort=True):
        sampled = class_rows.sample(
            n=min(max_samples_per_class, len(class_rows)),
            random_state=random_state + int(class_id),
        )
        iterator = tqdm(
            sampled.itertuples(index=False),
            total=len(sampled),
            desc=f"HOG class {class_id}",
            unit="object",
        )
        for row in iterator:
            image = cv2.imread(str(row.image_path))
            if image is None:
                continue
            crop = crop_from_yolo(
                image,
                float(row.x_center),
                float(row.y_center),
                float(row.bbox_w),
                float(row.bbox_h),
            )
            if crop is None:
                continue
            try:
                descriptor = extract_hog_features(crop)
            except cv2.error:
                continue
            features.append(descriptor)
            targets.append(int(class_id))
            groups.append(str(row.image_path))

    if not features:
        raise ValueError("Label табылды, бірақ жарамды HOG crop жасалмады.")

    x_data = np.vstack(features).astype(np.float32)
    y_data = np.asarray(targets, dtype=np.int32)
    group_data = np.asarray(groups)

    counts = Counter(y_data.tolist())
    valid_classes = {class_id for class_id, count in counts.items() if count >= MIN_SAMPLES_PER_CLASS}
    mask = np.asarray([class_id in valid_classes for class_id in y_data], dtype=bool)
    x_data, y_data, group_data = x_data[mask], y_data[mask], group_data[mask]

    if len(np.unique(y_data)) < 2:
        raise ValueError(
            "HOG+SVM үшін кемінде екі класс және әр класта кемінде "
            f"{MIN_SAMPLES_PER_CLASS} объект қажет."
        )

    config["hog_sample_counts"] = {
        str(class_id): int(np.sum(y_data == class_id)) for class_id in sorted(valid_classes)
    }
    return x_data, y_data, group_data, config


def _group_train_test_split(
    x_data: np.ndarray,
    y_data: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=RANDOM_STATE)
    all_classes = set(np.unique(y_data).tolist())
    for train_index, test_index in splitter.split(x_data, y_data, groups):
        if set(np.unique(y_data[train_index]).tolist()) == all_classes and set(
            np.unique(y_data[test_index]).tolist()
        ) == all_classes:
            return x_data[train_index], x_data[test_index], y_data[train_index], y_data[test_index]
    raise ValueError(
        "Image-group split барлық класты train және test бөлігіне бөле алмады. "
        "max_samples_per_class мәнін көбейтіңіз немесе сирек кластарды тексеріңіз."
    )


def train_hog_svm(
    data_yaml: str | Path,
    output_model: str | Path | None = None,
    max_samples_per_class: int = 300,
    out_model: str | Path | None = None,
) -> tuple[
    Pipeline,
    dict,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """HOG + LinearSVC. `out_model` ескі notebook кодтарымен үйлесімді."""
    model_path = output_model if output_model is not None else out_model
    if model_path is None:
        raise ValueError("output_model немесе out_model берілуі керек.")

    x_data, y_data, groups, config = make_hog_dataset(
        data_yaml=data_yaml,
        max_samples_per_class=max_samples_per_class,
    )
    x_train, x_test, y_train, y_test = _group_train_test_split(x_data, y_data, groups)

    minimum_class_count = min(Counter(y_train.tolist()).values())
    cv_folds = min(3, minimum_class_count)
    if cv_folds < 2:
        raise ValueError("GridSearchCV үшін training бөлігінде әр класта кемінде 2 үлгі керек.")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "svc",
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=20_000,
                    dual="auto",
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={"svc__C": [0.01, 0.1, 1.0, 10.0]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )
    grid.fit(x_train, y_train)
    predictions = grid.predict(x_test)

    class_ids = sorted(np.unique(y_data).tolist())
    names = get_class_names(config)
    target_names = [names.get(class_id, str(class_id)) for class_id in class_ids]
    metrics = {
        "model": "HOG + LinearSVC",
        "task": "object crop classification",
        "data_yaml": str(Path(data_yaml).resolve()),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "feature_count": int(x_data.shape[1]),
        "class_ids": class_ids,
        "class_names": target_names,
        "class_sample_counts": config.get("hog_sample_counts", {}),
        "cv_folds": int(cv_folds),
        "best_params": grid.best_params_,
        "best_cv_f1_macro": float(grid.best_score_),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=class_ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=class_ids).tolist(),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": grid.best_estimator_, "config": config, "metrics": metrics},
        model_path,
    )
    return grid.best_estimator_, metrics, (x_train, x_test, y_train, y_test)
