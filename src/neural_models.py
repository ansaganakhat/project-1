import tensorflow as tf

from config import MAX_TOKENS, SEQUENCE_LENGTH


def build_text_vectorizer(train_texts):
    """Adapt a TextVectorization layer only on training data."""
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
        name="text_vectorizer",
    )
    vectorizer.adapt(train_texts)
    return vectorizer


def build_mlp_tfidf_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="tfidf_features")
    x = inputs
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout_1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="spam_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="mlp_tfidf_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_bilstm_model(vocabulary_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int64, name="token_ids")
    x = inputs
    x = tf.keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=64,
        mask_zero=True,
        name="embedding",
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, dropout=0.2),
        name="bidirectional_lstm",
    )(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_relu")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="spam_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="bilstm_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
