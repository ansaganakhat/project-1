import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. БАПТАУЛАР
# ============================================================

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TRANSFER_EPOCHS = 15
FINE_TUNE_EPOCHS = 15

DATA_ROOT = Path("data")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. НӘТИЖЕНІ ҚАЙТАЛАУ ҮШІН SEED
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 3. ДАТАСЕТ ПАПКАЛАРЫН АВТОМАТТЫ ТҮРДЕ ТАБУ
# ============================================================

REQUIRED_CLASSES = {"healthy", "powdery", "rust"}


def find_split_directory(root: Path, split_name: str) -> Path:
    """
    data папкасынан Healthy, Powdery және Rust папкалары бар
    Train, Validation немесе Test папкасын іздейді.
    """

    split_name = split_name.lower()

    candidates = []

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        if path.name.lower() != split_name:
            continue

        child_directories = {
            child.name.lower()
            for child in path.iterdir()
            if child.is_dir()
        }

        if REQUIRED_CLASSES.issubset(child_directories):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"'{split_name}' датасет папкасы табылмады.\n"
            f"Ізделген орын: {root.resolve()}\n"
            "Папка ішінде Healthy, Powdery және Rust болуы керек."
        )

    return candidates[0]


TRAIN_DIR = find_split_directory(DATA_ROOT, "Train")
VAL_DIR = find_split_directory(DATA_ROOT, "Validation")
TEST_DIR = find_split_directory(DATA_ROOT, "Test")

print("\nДатасет папкалары:")
print("Train:", TRAIN_DIR.resolve())
print("Validation:", VAL_DIR.resolve())
print("Test:", TEST_DIR.resolve())


# ============================================================
# 4. ДЕРЕКТЕРДІ ЖҮКТЕУ
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

class_names = train_ds.class_names
num_classes = len(class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nКласстар:", class_names)
print("Класс саны:", num_classes)

with open(RESULT_DIR / "class_names.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file, ensure_ascii=False, indent=4)


# ============================================================
# 5. CLASS WEIGHT ЕСЕПТЕУ
# ============================================================

train_labels = np.concatenate(
    [labels.numpy() for _, labels in train_ds],
    axis=0,
)

unique_classes = np.unique(train_labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=train_labels,
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(unique_classes, weights)
}

print("\nClass weights:")
for class_id, weight in class_weights.items():
    print(f"{class_names[class_id]}: {weight:.4f}")


# ============================================================
# 6. DATA PIPELINE ОҢТАЙЛАНДЫРУ
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            mode="horizontal_and_vertical",
            seed=SEED,
        ),
        layers.RandomRotation(
            factor=0.15,
            seed=SEED,
        ),
        layers.RandomZoom(
            height_factor=(-0.15, 0.15),
            width_factor=(-0.15, 0.15),
            seed=SEED,
        ),
        layers.RandomContrast(
            factor=0.15,
            seed=SEED,
        ),
        layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10,
            seed=SEED,
        ),
    ],
    name="data_augmentation",
)


# ============================================================
# 8. AUGMENTATION НӘТИЖЕСІН КӨРСЕТУ
# ============================================================

def save_augmented_examples(dataset):
    for images, labels in dataset.take(1):
        plt.figure(figsize=(10, 10))

        sample_image = images[0]
        sample_label = int(labels[0].numpy())

        for index in range(9):
            augmented_image = data_augmentation(
                tf.expand_dims(sample_image, axis=0),
                training=True,
            )

            plt.subplot(3, 3, index + 1)
            plt.imshow(
                tf.cast(
                    tf.clip_by_value(augmented_image[0], 0, 255),
                    tf.uint8,
                )
            )
            plt.title(class_names[sample_label])
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            RESULT_DIR / "augmentation_examples.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


save_augmented_examples(train_ds)


# ============================================================
# 9. EFFICIENTNETB0 TRANSFER LEARNING МОДЕЛІ
# ============================================================

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
)

# Бірінші кезеңде алдын ала үйретілген қабаттар өзгермейді
base_model.trainable = False

inputs = keras.Input(
    shape=IMG_SIZE + (3,),
    name="input_image",
)

x = data_augmentation(inputs)

# BatchNormalization статистикасын өзгертпеу үшін training=False
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling",
)(x)

x = layers.BatchNormalization(
    name="head_batch_normalization",
)(x)

x = layers.Dropout(
    rate=0.40,
    name="dropout_1",
)(x)

x = layers.Dense(
    units=256,
    activation="relu",
    name="dense_256",
)(x)

x = layers.Dropout(
    rate=0.30,
    name="dropout_2",
)(x)

outputs = layers.Dense(
    units=num_classes,
    activation="softmax",
    name="classification_output",
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="plant_disease_efficientnetb0",
)

model.summary()


# ============================================================
# 10. TRANSFER LEARNING КЕЗЕҢІ
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

transfer_callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_DIR / "best_transfer_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),

    keras.callbacks.CSVLogger(
        filename=RESULT_DIR / "transfer_training_log.csv",
    ),
]

print("\n" + "=" * 70)
print("1-КЕЗЕҢ: TRANSFER LEARNING")
print("=" * 70)

transfer_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TRANSFER_EPOCHS,
    class_weight=class_weights,
    callbacks=transfer_callbacks,
)


# ============================================================
# 11. FINE-TUNING
# ============================================================

base_model.trainable = True

# EfficientNetB0 моделінің тек соңғы 30 қабатын үйретеміз
for layer in base_model.layers[:-30]:
    layer.trainable = False

# BatchNormalization қабаттарын frozen күйінде қалдырамыз
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

fine_tune_callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_DIR / "best_fine_tuned_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-8,
        verbose=1,
    ),

    keras.callbacks.CSVLogger(
        filename=RESULT_DIR / "fine_tuning_log.csv",
    ),
]

print("\n" + "=" * 70)
print("2-КЕЗЕҢ: FINE-TUNING")
print("=" * 70)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=len(transfer_history.history["loss"]),
    epochs=len(transfer_history.history["loss"]) + FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=fine_tune_callbacks,
)


# ============================================================
# 12. ЕҢ ЖАҚСЫ МОДЕЛЬДІ ЖҮКТЕУ
# ============================================================

best_model_path = MODEL_DIR / "best_fine_tuned_model.keras"

if best_model_path.exists():
    model = keras.models.load_model(best_model_path)
    print("\nFine-tuned ең жақсы модель жүктелді.")
else:
    print("\nFine-tuned модель табылмады. Қазіргі модель қолданылады.")


# ============================================================
# 13. TEST DATASET БОЙЫНША БАҒАЛАУ
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1,
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# ============================================================
# 14. БОЛЖАМ ЖАСАУ
# ============================================================

prediction_probabilities = model.predict(
    test_ds,
    verbose=1,
)

y_pred = np.argmax(
    prediction_probabilities,
    axis=1,
)

y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds],
    axis=0,
)


# ============================================================
# 15. МЕТРИКАЛАР
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

metrics = {
    "test_loss": test_loss,
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "precision_weighted": precision_weighted,
    "recall_weighted": recall_weighted,
    "f1_weighted": f1_weighted,
}

print("\n" + "=" * 70)
print("TEST МЕТРИКАЛАРЫ")
print("=" * 70)

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

metrics_df = pd.DataFrame(
    [metrics],
    index=["EfficientNetB0 Fine-tuned"],
)

metrics_df.to_csv(
    RESULT_DIR / "test_metrics.csv",
    encoding="utf-8-sig",
)

with open(
    RESULT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

report_text = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print("\nClassification Report:")
print(report_text)

with open(
    RESULT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(report_text)

report_dictionary = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dictionary).transpose()

report_df.to_csv(
    RESULT_DIR / "classification_report.csv",
    encoding="utf-8-sig",
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

figure, axis = plt.subplots(figsize=(7, 7))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(
    ax=axis,
    values_format="d",
    colorbar=False,
)

axis.set_title("Confusion Matrix — EfficientNetB0 Fine-tuned")

plt.tight_layout()
plt.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=250,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 18. TRAINING ГРАФИКТЕРІ
# ============================================================

transfer_accuracy = transfer_history.history["accuracy"]
transfer_val_accuracy = transfer_history.history["val_accuracy"]
transfer_loss = transfer_history.history["loss"]
transfer_val_loss = transfer_history.history["val_loss"]

fine_accuracy = fine_tune_history.history["accuracy"]
fine_val_accuracy = fine_tune_history.history["val_accuracy"]
fine_loss = fine_tune_history.history["loss"]
fine_val_loss = fine_tune_history.history["val_loss"]

all_accuracy = transfer_accuracy + fine_accuracy
all_val_accuracy = transfer_val_accuracy + fine_val_accuracy
all_loss = transfer_loss + fine_loss
all_val_loss = transfer_val_loss + fine_val_loss

epochs = range(1, len(all_accuracy) + 1)
fine_tune_start = len(transfer_accuracy)


plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    all_accuracy,
    label="Train Accuracy",
)

plt.plot(
    epochs,
    all_val_accuracy,
    label="Validation Accuracy",
)

plt.axvline(
    x=fine_tune_start,
    linestyle="--",
    label="Fine-tuning басталды",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training және Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "accuracy_history.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    all_loss,
    label="Train Loss",
)

plt.plot(
    epochs,
    all_val_loss,
    label="Validation Loss",
)

plt.axvline(
    x=fine_tune_start,
    linestyle="--",
    label="Fine-tuning басталды",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training және Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "loss_history.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 19. ҚАТЕ БОЛЖАМДАРДЫ КӨРСЕТУ
# ============================================================

all_test_images = np.concatenate(
    [images.numpy() for images, _ in test_ds],
    axis=0,
)

incorrect_indices = np.where(y_true != y_pred)[0]

if len(incorrect_indices) > 0:
    number_to_show = min(9, len(incorrect_indices))

    plt.figure(figsize=(12, 12))

    for plot_index, image_index in enumerate(
        incorrect_indices[:number_to_show]
    ):
        plt.subplot(3, 3, plot_index + 1)

        plt.imshow(
            all_test_images[image_index].astype("uint8")
        )

        true_class = class_names[y_true[image_index]]
        predicted_class = class_names[y_pred[image_index]]

        confidence = np.max(
            prediction_probabilities[image_index]
        )

        plt.title(
            f"True: {true_class}\n"
            f"Pred: {predicted_class}\n"
            f"Confidence: {confidence:.2%}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "incorrect_predictions.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

else:
    print("\nTest dataset ішінде қате болжам табылмады.")


# ============================================================
# 20. СОҢҒЫ МОДЕЛЬДІ САҚТАУ
# ============================================================

model.save(
    MODEL_DIR / "plant_disease_final.keras"
)

print("\n" + "=" * 70)
print("ОҚЫТУ АЯҚТАЛДЫ")
print("=" * 70)

print("Модель:")
print(MODEL_DIR / "plant_disease_final.keras")

print("\nНәтижелер:")
print(RESULT_DIR.resolve())
