import json
import os
import sys
import traceback
from pathlib import Path

# TensorFlow-дың ақпараттық хабарламаларын азайту
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ============================================================
# 1. ТЕКСЕРІЛЕТІН СУРЕТТІҢ АТАУЫ
# ============================================================

# Сурет predict.py файлының жанында орналасуы керек.
IMAGE_NAME = "test_healthy.jpg"


# ============================================================
# 2. НЕГІЗГІ БАПТАУЛАР
# ============================================================

IMG_SIZE = (224, 224)

# predict.py орналасқан жоба папкасы
BASE_DIR = Path(__file__).resolve().parent

# Үйретілген модельдің жолы
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "plant_disease_final.keras"
)

# Класс атаулары сақталған файл
CLASS_NAMES_PATH = (
    BASE_DIR
    / "results"
    / "class_names.json"
)

# Тексерілетін сурет
IMAGE_PATH = BASE_DIR / IMAGE_NAME

# Нәтиже сақталатын папка
PREDICTION_RESULT_DIR = BASE_DIR / "prediction_results"


# ============================================================
# 3. ЖҮЙЕ АҚПАРАТЫН КӨРСЕТУ
# ============================================================

def print_system_info():
    print("=" * 65)
    print("ӨСІМДІК АУРУЫН АНЫҚТАУ")
    print("=" * 65)

    print(f"Python нұсқасы     : {sys.version.split()[0]}")
    print(f"TensorFlow нұсқасы : {tf.__version__}")

    try:
        keras_version = tf.keras.__version__
    except AttributeError:
        keras_version = "TensorFlow құрамындағы Keras"

    print(f"Keras нұсқасы      : {keras_version}")
    print(f"Жоба папкасы       : {BASE_DIR}")
    print(f"Тексерілетін сурет : {IMAGE_NAME}")


# ============================================================
# 4. КЛАСС АТАУЛАРЫН ЖҮКТЕУ
# ============================================================

def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "\nКласс атауларының файлы табылмады:\n"
            f"{CLASS_NAMES_PATH}\n\n"
            "results папкасында class_names.json файлы болуы керек."
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise ValueError(
            "class_names.json файлында класстар тізім түрінде болуы керек."
        )

    if len(class_names) == 0:
        raise ValueError(
            "class_names.json файлында класс атаулары жоқ."
        )

    print("\nКласстар:")
    for index, class_name in enumerate(class_names):
        print(f"  {index}: {class_name}")

    return class_names


# ============================================================
# 5. МОДЕЛЬДІ ЖҮКТЕУ
# ============================================================

def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nҮйретілген модель табылмады:\n"
            f"{MODEL_PATH}\n\n"
            "Алдымен train.py файлын іске қосып, модельді үйретіңіз."
        )

    print("\nМодель жүктеліп жатыр...")
    print(f"Модель жолы: {MODEL_PATH}")

    try:
        # compile=False — prediction кезінде optimizer және loss қажет емес
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
        )

        print("Модель сәтті жүктелді.")
        return model

    except TypeError as first_error:
        print("\nҚалыпты жүктеу кезінде TypeError пайда болды.")
        print("safe_mode=False параметрімен қайта жүктеледі...")

        try:
            model = tf.keras.models.load_model(
                str(MODEL_PATH),
                compile=False,
                safe_mode=False,
            )

            print("Модель safe_mode=False арқылы сәтті жүктелді.")
            return model

        except Exception as second_error:
            raise RuntimeError(
                "\nМодельді жүктеу мүмкін болмады.\n"
                "Модель басқа TensorFlow/Keras нұсқасында "
                "сақталған болуы мүмкін.\n\n"
                f"Бірінші қате:\n{first_error}\n\n"
                f"Екінші қате:\n{second_error}"
            ) from second_error


# ============================================================
# 6. СУРЕТТІ ДАЙЫНДАУ
# ============================================================

def prepare_image():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            "\nТексерілетін сурет табылмады:\n"
            f"{IMAGE_PATH}\n\n"
            "Суретті predict.py файлының жанына салыңыз "
            "немесе IMAGE_NAME мәнін дұрыс жазыңыз."
        )

    print("\nСурет өңделіп жатыр...")

    image = tf.keras.utils.load_img(
        str(IMAGE_PATH),
        target_size=IMG_SIZE,
        color_mode="rgb",
    )

    image_array = tf.keras.utils.img_to_array(image)

    # (224, 224, 3) → (1, 224, 224, 3)
    input_batch = np.expand_dims(
        image_array,
        axis=0,
    )

    print(f"Сурет өлшемі: {image_array.shape}")
    print(f"Batch өлшемі: {input_batch.shape}")

    return image, input_batch


# ============================================================
# 7. БОЛЖАМ НӘТИЖЕСІН ЕСЕПТЕУ
# ============================================================

def make_prediction(
    model,
    input_batch,
    class_names,
):
    print("\nБолжам жасалып жатыр...")

    probabilities = model.predict(
        input_batch,
        verbose=0,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            f"Модель нәтижесінің өлшемі қате: {probabilities.shape}"
        )

    probabilities = probabilities[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "\nМодель шығаратын класс саны мен "
            "class_names.json ішіндегі класс саны сәйкес емес.\n"
            f"Модель класс саны: {len(probabilities)}\n"
            f"JSON класс саны: {len(class_names)}"
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = class_names[predicted_index]
    confidence = float(
        probabilities[predicted_index]
    )

    return (
        probabilities,
        predicted_index,
        predicted_class,
        confidence,
    )


# ============================================================
# 8. НӘТИЖЕНІ КОНСОЛЬГЕ ШЫҒАРУ
# ============================================================

def print_prediction_result(
    probabilities,
    predicted_class,
    confidence,
    class_names,
):
    print("\n" + "=" * 65)
    print("БОЛЖАМ НӘТИЖЕСІ")
    print("=" * 65)

    print(f"Анықталған класс : {predicted_class}")
    print(f"Сенімділік       : {confidence:.2%}")

    print("\nБарлық класстың ықтималдығы:")

    sorted_results = sorted(
        zip(class_names, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    for class_name, probability in sorted_results:
        print(
            f"  {class_name:<20} "
            f"{float(probability):>8.2%}"
        )


# ============================================================
# 9. НӘТИЖЕНІ ГРАФИКПЕН КӨРСЕТУ
# ============================================================

def show_and_save_result(
    image,
    probabilities,
    predicted_class,
    confidence,
    class_names,
):
    PREDICTION_RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_filename = (
        f"prediction_{Path(IMAGE_NAME).stem}.png"
    )

    result_path = (
        PREDICTION_RESULT_DIR
        / result_filename
    )

    figure = plt.figure(
        figsize=(12, 6)
    )

    # Бірінші бөлік — сурет
    image_axis = figure.add_subplot(1, 2, 1)

    image_axis.imshow(image)
    image_axis.set_title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2%}",
        fontsize=14,
    )
    image_axis.axis("off")

    # Екінші бөлік — ықтималдықтар
    probability_axis = figure.add_subplot(1, 2, 2)

    positions = np.arange(
        len(class_names)
    )

    probability_axis.barh(
        positions,
        probabilities * 100,
    )

    probability_axis.set_yticks(
        positions
    )

    probability_axis.set_yticklabels(
        class_names
    )

    probability_axis.set_xlabel(
        "Ықтималдық, %"
    )

    probability_axis.set_title(
        "Класстар бойынша нәтиже"
    )

    probability_axis.set_xlim(
        0,
        100,
    )

    for index, probability in enumerate(probabilities):
        probability_axis.text(
            float(probability * 100) + 1,
            index,
            f"{float(probability):.2%}",
            va="center",
        )

    plt.tight_layout()

    plt.savefig(
        result_path,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\nНәтиже сақталды:\n{result_path}")

    plt.show()


# ============================================================
# 10. НЕГІЗГІ ФУНКЦИЯ
# ============================================================

def main():
    print_system_info()

    class_names = load_class_names()

    model = load_trained_model()

    image, input_batch = prepare_image()

    (
        probabilities,
        predicted_index,
        predicted_class,
        confidence,
    ) = make_prediction(
        model=model,
        input_batch=input_batch,
        class_names=class_names,
    )

    print_prediction_result(
        probabilities=probabilities,
        predicted_class=predicted_class,
        confidence=confidence,
        class_names=class_names,
    )

    show_and_save_result(
        image=image,
        probabilities=probabilities,
        predicted_class=predicted_class,
        confidence=confidence,
        class_names=class_names,
    )

    print("\n" + "=" * 65)
    print("БАҒДАРЛАМА СӘТТІ АЯҚТАЛДЫ")
    print("=" * 65)


# ============================================================
# 11. БАҒДАРЛАМАНЫ ІСКЕ ҚОСУ
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except FileNotFoundError as error:
        print("\n" + "=" * 65)
        print("ФАЙЛ ТАБЫЛМАДЫ")
        print("=" * 65)
        print(error)

    except ValueError as error:
        print("\n" + "=" * 65)
        print("ДЕРЕКТЕР ҚАТЕСІ")
        print("=" * 65)
        print(error)

    except Exception:
        print("\n" + "=" * 65)
        print("БАҒДАРЛАМАДА ҚАТЕ ПАЙДА БОЛДЫ")
        print("=" * 65)

        # Қатенің толық мәтінін шығарады
        traceback.print_exc()

    finally:
        input(
            "\nБағдарламаны жабу үшін Enter басыңыз..."
        )
