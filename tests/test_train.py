import numpy as np
from PIL import Image

from src.model import preprocess_image
from src.prepare import list_images
from src.train import make_dataset


def test_make_dataset_shapes():
    df = list_images("Data/SampleTestDataset")
    for training in (True, False):
        x, y = next(iter(make_dataset(df, batch_size=8, training=training)))
        assert x.shape == (8, 224, 224, 3)
        assert x.dtype == "float32"
        assert y.shape == (8, 4)
        assert float(y.numpy().sum()) == 8.0  # one-hot


def test_eval_input_matches_app_preprocessing():
    # The accuracy gate must score the same input the app feeds the model.
    df = list_images("Data/SampleTestDataset").iloc[:1]
    x, _ = next(iter(make_dataset(df, batch_size=1, training=False)))
    app_x = preprocess_image(Image.open(df["image_path"].iloc[0]))
    assert np.abs(x.numpy() - app_x).max() <= 3.0  # TF vs PIL nearest differ by a couple of grey levels
