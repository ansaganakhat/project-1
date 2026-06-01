from __future__ import annotations

import argparse
import gzip
import json
import struct
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

DATASET_URLS = {
    "train_images": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
    "train_labels": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
    "test_images": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fashion-MNIST deep learning project")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for each model")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root where figures, models and results are saved",
    )
    return parser.parse_args()


def prepare_dirs(project_dir: Path) -> dict[str, Path]:
    paths = {
        "data": project_dir / "data" / "raw",
        "figures": project_dir / "figures",
        "models": project_dir / "models",
        "results": project_dir / "results",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        return
    print(f"Downloading {output_path.name}...")
    urllib.request.urlretrieve(url, output_path)


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image file magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count)


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = {
        name: data_dir / url.rsplit("/", 1)[-1]
        for name, url in DATASET_URLS.items()
    }
    for name, url in DATASET_URLS.items():
        download_file(url, files[name])

    x_train = read_idx_images(files["train_images"])
    y_train = read_idx_labels(files["train_labels"])
    x_test = read_idx_images(files["test_images"])
    y_test = read_idx_labels(files["test_labels"])
    return x_train, y_train, x_test, y_test


def stratified_train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)
        val_count = int(round(len(class_indices) * test_size))
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return x[train_indices], x[val_indices], y[train_indices], y[val_indices]


def make_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def make_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    matrix = make_confusion_matrix(y_true, y_pred, len(CLASS_NAMES))
    rows = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = matrix[class_id, class_id]
        fp = matrix[:, class_id].sum() - tp
        fn = matrix[class_id, :].sum() - tp
        support = matrix[class_id, :].sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support,
            }
        )

    report = pd.DataFrame(rows).set_index("class")
    accuracy = np.trace(matrix) / matrix.sum()
    macro_avg = report[["precision", "recall", "f1-score"]].mean()
    weighted_avg = (
        report[["precision", "recall", "f1-score"]]
        .multiply(report["support"], axis=0)
        .sum()
        / report["support"].sum()
    )

    report.loc["accuracy", ["precision", "recall", "f1-score"]] = accuracy
    report.loc["accuracy", "support"] = report.loc[CLASS_NAMES, "support"].sum()
    report.loc["macro avg", ["precision", "recall", "f1-score"]] = macro_avg
    report.loc["macro avg", "support"] = report.loc[CLASS_NAMES, "support"].sum()
    report.loc["weighted avg", ["precision", "recall", "f1-score"]] = weighted_avg
    report.loc["weighted avg", "support"] = report.loc[CLASS_NAMES, "support"].sum()

    return report


def save_eda_plots(
    x_train: np.ndarray,
    y_train: np.ndarray,
    figures_dir: Path,
) -> None:
    class_counts = pd.Series(y_train).value_counts().sort_index()
    class_df = pd.DataFrame(
        {
            "class_id": class_counts.index,
            "class_name": [CLASS_NAMES[i] for i in class_counts.index],
            "count": class_counts.values,
        }
    )

    plt.figure(figsize=(10, 5))
    plt.bar(class_df["class_name"], class_df["count"], color="#2a9d8f")
    plt.title("Fashion-MNIST class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for class_id, class_name in enumerate(CLASS_NAMES):
        idx = np.where(y_train == class_id)[0][0]
        plt.subplot(2, 5, class_id + 1)
        plt.imshow(x_train[idx], cmap="gray")
        plt.title(class_name, fontsize=9)
        plt.axis("off")
    plt.suptitle("One sample image from each class", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "sample_images.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(x_train.flatten(), bins=50, color="#2a9d8f")
    plt.title("Pixel intensity distribution before normalization")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figures_dir / "pixel_distribution.png", dpi=160)
    plt.close()


def preprocess(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    x_train_norm = x_train.astype("float32") / 255.0
    x_test_norm = x_test.astype("float32") / 255.0

    x_train_part, x_val_part, y_train_part, y_val_part = stratified_train_val_split(
        x_train_norm,
        y_train,
        test_size=0.1,
        seed=seed,
    )

    return {
        "x_train_mlp": x_train_part.reshape(-1, 28 * 28),
        "x_val_mlp": x_val_part.reshape(-1, 28 * 28),
        "x_test_mlp": x_test_norm.reshape(-1, 28 * 28),
        "x_train_cnn": x_train_part[..., np.newaxis],
        "x_val_cnn": x_val_part[..., np.newaxis],
        "x_test_cnn": x_test_norm[..., np.newaxis],
        "y_train": y_train_part,
        "y_val": y_val_part,
    }


def build_mlp() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(784,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ],
        name="mlp_model",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ],
        name="cnn_model",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(
    model_name: str,
    model: keras.Model,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    paths: dict[str, Path],
) -> tuple[dict[str, float], keras.callbacks.History, np.ndarray]:
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    report = make_classification_report(y_test, y_pred)
    report.to_csv(paths["results"] / f"{model_name}_classification_report.csv")
    model.save(paths["models"] / f"{model_name}.keras")

    return (
        {
            "model": model_name,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "parameters": int(model.count_params()),
        },
        history,
        y_pred,
    )


def plot_history(histories: dict[str, keras.callbacks.History], figures_dir: Path) -> None:
    plt.figure(figsize=(11, 4))

    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history["accuracy"], label=f"{name} train")
        plt.plot(history.history["val_accuracy"], linestyle="--", label=f"{name} val")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history["loss"], label=f"{name} train")
        plt.plot(history.history["val_loss"], linestyle="--", label=f"{name} val")
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "training_history.png", dpi=160)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    matrix = make_confusion_matrix(y_true, y_pred, len(CLASS_NAMES))
    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=35, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=8)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    paths = prepare_dirs(args.project_dir)

    print("Loading Fashion-MNIST dataset from official GitHub raw files...")
    x_train, y_train, x_test, y_test = load_dataset(paths["data"])
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

    save_eda_plots(x_train, y_train, paths["figures"])
    processed = preprocess(x_train, x_test, y_train, args.seed)

    experiments = {
        "mlp": (
            build_mlp(),
            processed["x_train_mlp"],
            processed["x_val_mlp"],
            processed["x_test_mlp"],
        ),
        "cnn": (
            build_cnn(),
            processed["x_train_cnn"],
            processed["x_val_cnn"],
            processed["x_test_cnn"],
        ),
    }

    metrics = []
    histories = {}
    predictions = {}

    for model_name, (model, x_train_model, x_val_model, x_test_model) in experiments.items():
        print(f"\nTraining {model_name.upper()}...")
        model_metrics, history, y_pred = train_and_evaluate(
            model_name=model_name,
            model=model,
            x_train=x_train_model,
            x_val=x_val_model,
            x_test=x_test_model,
            y_train=processed["y_train"],
            y_val=processed["y_val"],
            y_test=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            paths=paths,
        )
        metrics.append(model_metrics)
        histories[model_name] = history
        predictions[model_name] = y_pred

    plot_history(histories, paths["figures"])
    for model_name, y_pred in predictions.items():
        plot_confusion_matrix(
            y_test,
            y_pred,
            title=f"{model_name.upper()} confusion matrix",
            output_path=paths["figures"] / f"{model_name}_confusion_matrix.png",
        )

    comparison = pd.DataFrame(metrics).sort_values("test_accuracy", ascending=False)
    comparison.to_csv(paths["results"] / "model_comparison.csv", index=False)
    with (paths["results"] / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    best_model = comparison.iloc[0]["model"]
    print(f"\nBest model by test accuracy: {best_model.upper()}")


if __name__ == "__main__":
    main()
