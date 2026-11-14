"""Stage 2: train ResNet50 on splits, write weights + per-epoch history."""
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
import yaml

from src.model import CLASSES, IMG_SIZE, build_model

MODEL_PATH = Path("models/model.weights.h5")
HISTORY_PATH = Path("reports/history.json")

# Same augmentation as the notebook's ImageDataGenerator (rotation 1°, shift 1%, zoom 10%).
_augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(1 / 360, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.01, 0.01, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
])


def _load(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.image.resize(img, IMG_SIZE, method="nearest")


def make_dataset(df, batch_size, training):
    labels = [CLASSES.index(label) for label in df["label"]]
    ds = tf.data.Dataset.from_tensor_slices((df["image_path"].tolist(), labels))
    if training:
        ds = ds.shuffle(len(df), seed=42)
    ds = ds.map(lambda p, y: (_load(p), tf.one_hot(y, len(CLASSES))), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if training:
        ds = ds.map(lambda x, y: (_augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    preprocess = tf.keras.applications.resnet50.preprocess_input
    return ds.map(lambda x, y: (preprocess(x), y)).prefetch(tf.data.AUTOTUNE)


def main():
    p = yaml.safe_load(Path("params.yaml").read_text())["train"]
    tf.keras.utils.set_random_seed(42)
    train_df = pd.read_csv("splits/train.csv")
    val_df = pd.read_csv("splits/val.csv")

    model = build_model(weights=p["weights"], dropout=p["dropout"], dense_units=p["dense_units"])
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=p["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=p["patience"], min_delta=0.001, restore_best_weights=True, verbose=1
    )
    history = model.fit(
        make_dataset(train_df, p["batch_size"], training=True),
        validation_data=make_dataset(val_df, p["batch_size"], training=False),
        epochs=p["epochs"],
        callbacks=[early_stopping],
        verbose=2,
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    model.save_weights(MODEL_PATH)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    HISTORY_PATH.write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    main()
