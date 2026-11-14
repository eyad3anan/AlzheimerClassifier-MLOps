from src.prepare import list_images
from src.train import make_dataset


def test_make_dataset_shapes():
    df = list_images("Data/SampleTestDataset")
    for training in (True, False):
        x, y = next(iter(make_dataset(df, batch_size=8, training=training)))
        assert x.shape == (8, 224, 224, 3)
        assert y.shape == (8, 4)
        assert float(y.numpy().sum()) == 8.0  # one-hot
