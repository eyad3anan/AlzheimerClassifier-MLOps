import numpy as np
from PIL import Image

from src.model import CLASSES, build_model, preprocess_image


def test_classes_order_matches_training():
    assert CLASSES == ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


def test_build_model_outputs_4_probs():
    probs = build_model().predict(np.zeros((1, 224, 224, 3), np.float32), verbose=0)
    assert probs.shape == (1, 4)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_preprocess_handles_modes():
    for mode in ("L", "RGBA", "RGB"):
        x = preprocess_image(Image.new(mode, (100, 80)))
        assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32


def test_weights_roundtrip(tmp_path):
    path = str(tmp_path / "x.weights.h5")
    a = build_model()
    a.save_weights(path)
    b = build_model()
    b.load_weights(path)
    x = np.random.default_rng(0).random((1, 224, 224, 3), np.float32)
    np.testing.assert_allclose(a.predict(x, verbose=0), b.predict(x, verbose=0), rtol=1e-5)
