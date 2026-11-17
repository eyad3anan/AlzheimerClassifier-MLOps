import importlib.util

import pytest

# kaggle/ is not a package (the name clashes with the kaggle pip package), so load by path
spec = importlib.util.spec_from_file_location("train_kernel", "kaggle/train_kernel.py")
train_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_kernel)


@pytest.mark.parametrize("layout", [
    "augmented-alzheimer-mri-dataset/OriginalDataset",
    "datasets/uraninjo/augmented-alzheimer-mri-dataset/OriginalDataset",
])
def test_find_dataset_any_mount_layout(tmp_path, layout):
    (tmp_path / layout / "NonDemented").mkdir(parents=True)
    (tmp_path / "x/AugmentedAlzheimerDataset/NonDemented").mkdir(parents=True)  # the other folder
    assert train_kernel.find_dataset(tmp_path) == (tmp_path / layout).as_posix()


def test_find_dataset_missing_says_what_is_there(tmp_path):
    (tmp_path / "something-else").mkdir()
    with pytest.raises(SystemExit, match="something-else"):
        train_kernel.find_dataset(tmp_path)


def test_set_raw_dir_rewrites_only_that_line():
    text = "data:\n  raw_dir: /old/path\n  seed: 42\n"
    assert train_kernel.set_raw_dir(text, "/kaggle/input/x/OriginalDataset") == (
        "data:\n  raw_dir: /kaggle/input/x/OriginalDataset\n  seed: 42\n")
