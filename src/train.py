import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BATCH_SIZE,
    EPOCHS,
    FIGURES_DIR,
    MAX_TFIDF_FEATURES,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
)
from data import clean_dataset, load_raw_dataset, save_processed_dataset, save_json
from eda import run_eda
from neural_models import (
    build_bilstm_model,
    build_mlp_tfidf_model,
    build_text_vectorizer,
)


MODEL_BUILDERS = {
    "mlp_tfidf": build_mlp_tfidf_model,
    "bilstm": build_bilstm_model,
}


def set_reproducibility() -> None:
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def prepare_train_validation_test_split(clean_df: pd.DataFrame):
    X = clean_df["message"].astype(str).to_numpy()
    y = clean_df["label_encoded"].astype(int).to_numpy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def vectorize_texts(vectorizer: tf.keras.layers.TextVectorization, texts: np.ndarray) -> np.ndarray:
    return vectorizer(tf.constant(texts.astype(str))).numpy()


def class_distribution(y: np.ndarray) -> dict:
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def plot_training_curves(history: tf.keras.callbacks.History, model_name: str) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(RESULTS_DIR / f"history_{model_name}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history_df["loss"], label="train loss", marker="o")
    axes[0].plot(history_df["val_loss"], label="validation loss", marker="o")
    axes[0].set_title(f"{model_name}: loss curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_df["accuracy"], label="train accuracy", marker="o")
    axes[1].plot(history_df["val_accuracy"], label="validation accuracy", marker="o")
    axes[1].set_title(f"{model_name}: accuracy curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"training_curves_{model_name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    plt.figure(figsize=(5.5, 4.5))
    axis = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    axis.set_title(f"{model_name}: confusion matrix")
    axis.set_xlabel("Болжанған класс")
    axis.set_ylabel("Нақты класс")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    metric_df = comparison_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1_score"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(9, 4.8))
    axis = sns.barplot(data=metric_df, x="metric", y="score", hue="model")
    axis.set_ylim(0, 1.02)
    axis.set_title("Модельдерді метрикалар бойынша салыстыру")
    axis.set_xlabel("Метрика")
    axis.set_ylabel("Балл")
    axis.legend(title="Модель")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_metrics.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_model_summary(model: tf.keras.Model, model_name: str) -> None:
    lines = []
    model.summary(print_fn=lines.append)
    (RESULTS_DIR / f"model_summary_{model_name}.txt").write_text("\n".join(lines), encoding="utf-8")


def find_best_threshold(model: tf.keras.Model, X_val_tokens, y_val) -> dict:
    probabilities = model.predict(X_val_tokens, batch_size=BATCH_SIZE, verbose=0).ravel()
    thresholds = np.arange(0.20, 0.81, 0.01)
    threshold_scores = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        threshold_scores.append(
            {
                "threshold": float(threshold),
                "f1_score": f1_score(y_val, predictions, zero_division=0),
                "precision": precision_score(y_val, predictions, zero_division=0),
                "recall": recall_score(y_val, predictions, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(threshold_scores).sort_values(
        ["f1_score", "recall", "precision"],
        ascending=False,
    )
    return {key: float(value) for key, value in threshold_df.iloc[0].to_dict().items()}


def evaluate_model(model: tf.keras.Model, X_test_tokens, X_test_text, y_test, model_name: str, threshold: float) -> dict:
    probabilities = model.predict(X_test_tokens, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm, model_name)

    report = classification_report(
        y_test,
        predictions,
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )
    save_json(report, RESULTS_DIR / f"classification_report_{model_name}.json")

    prediction_df = pd.DataFrame(
        {
            "message": X_test_text,
            "true_label": y_test,
            "predicted_label": predictions,
            "spam_probability": probabilities,
        }
    )
    prediction_df.to_csv(RESULTS_DIR / f"predictions_{model_name}.csv", index=False)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "threshold": float(threshold),
        "test_loss": float(model.evaluate(X_test_tokens, y_test, batch_size=BATCH_SIZE, verbose=0)[0]),
        "confusion_matrix": cm.tolist(),
    }


def train_and_evaluate():
    set_reproducibility()
    sns.set_theme(style="whitegrid", font_scale=1.0)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset()
    clean_df = clean_dataset(raw_df)
    save_processed_dataset(clean_df)
    eda_summary = run_eda(raw_df, clean_df)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_validation_test_split(clean_df)
    split_summary = {
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "train_class_distribution": class_distribution(y_train),
        "validation_class_distribution": class_distribution(y_val),
        "test_class_distribution": class_distribution(y_test),
    }
    save_json(split_summary, RESULTS_DIR / "split_summary.json")

    vectorizer = build_text_vectorizer(X_train)
    vocabulary_size = len(vectorizer.get_vocabulary())
    X_train_tokens = vectorize_texts(vectorizer, X_train)
    X_val_tokens = vectorize_texts(vectorizer, X_val)
    X_test_tokens = vectorize_texts(vectorizer, X_test)

    tfidf_vectorizer = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).astype(np.float32).toarray()
    X_val_tfidf = tfidf_vectorizer.transform(X_val).astype(np.float32).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(X_test).astype(np.float32).toarray()

    class_weights = get_class_weights(y_train)
    vectorizer_vocabulary_path = MODELS_DIR / "text_vectorizer_vocabulary.json"
    with open(vectorizer_vocabulary_path, "w", encoding="utf-8") as file:
        json.dump(vectorizer.get_vocabulary(), file, indent=2, ensure_ascii=False)
    tfidf_vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
    save_json(
        {
            "computed_class_weights": class_weights,
            "used_class_weights": False,
            "vocabulary_size": vocabulary_size,
            "tfidf_feature_count": int(X_train_tfidf.shape[1]),
            "vectorizer_vocabulary_path": str(vectorizer_vocabulary_path),
            "tfidf_vectorizer_path": str(tfidf_vectorizer_path),
        },
        RESULTS_DIR / "training_setup.json",
    )

    metrics = []
    histories = {}
    model_inputs = {
        "mlp_tfidf": {
            "builder_argument": int(X_train_tfidf.shape[1]),
            "X_train": X_train_tfidf,
            "X_val": X_val_tfidf,
            "X_test": X_test_tfidf,
        },
        "bilstm": {
            "builder_argument": vocabulary_size,
            "X_train": X_train_tokens,
            "X_val": X_val_tokens,
            "X_test": X_test_tokens,
        },
    }

    for model_name, builder in MODEL_BUILDERS.items():
        print(f"\nTraining {model_name}")
        inputs = model_inputs[model_name]
        model = builder(inputs["builder_argument"])
        save_model_summary(model, model_name)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            inputs["X_train"],
            y_train,
            validation_data=(inputs["X_val"], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,
        )
        histories[model_name] = history.history
        plot_training_curves(history, model_name)

        model.save(MODELS_DIR / f"{model_name}.keras")
        threshold_info = find_best_threshold(model, inputs["X_val"], y_val)
        save_json(threshold_info, RESULTS_DIR / f"threshold_{model_name}.json")
        metrics.append(
            evaluate_model(
                model,
                inputs["X_test"],
                X_test,
                y_test,
                model_name,
                threshold=threshold_info["threshold"],
            )
        )

    comparison_df = pd.DataFrame(metrics).sort_values("f1_score", ascending=False)
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    comparison_payload = {
        "eda_summary": eda_summary,
        "split_summary": split_summary,
        "models": metrics,
        "best_model_by_f1": comparison_df.iloc[0]["model"],
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(comparison_payload, file, indent=2, ensure_ascii=False)

    plot_model_comparison(comparison_df)
    return comparison_payload


if __name__ == "__main__":
    payload = train_and_evaluate()
    print("\nTraining complete. Best model:", payload["best_model_by_f1"])
