"""Model architecture and preprocessing shared by training and the Streamlit app."""
import numpy as np
import tensorflow as tf
from PIL import Image

# Alphabetical, the order the training labels are indexed in.
CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
IMG_SIZE = (224, 224)


def build_model(weights=None, dropout=0.45, dense_units=256):
    """ResNet50 + classification head from the notebook. weights="imagenet" for training."""
    base = tf.keras.applications.ResNet50(
        include_top=False, weights=weights, input_shape=(*IMG_SIZE, 3), pooling="max"
    )
    reg = tf.keras.regularizers
    return tf.keras.Sequential([
        tf.keras.Input(shape=(*IMG_SIZE, 3)),
        base,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=reg.l2(0.001),
            activity_regularizer=reg.l1(0.0001),
            bias_regularizer=reg.l1(0.00001),
        ),
        tf.keras.layers.Dropout(dropout, seed=42),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])


def preprocess_image(image: Image.Image) -> np.ndarray:
    """PIL image -> (1, 224, 224, 3) float32, same resize + normalisation as training."""
    image = image.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    x = tf.keras.applications.resnet50.preprocess_input(np.array(image, dtype=np.float32))
    return np.expand_dims(x, axis=0)
