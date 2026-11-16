from src.prepare import list_images, split

SAMPLE = "Data/SampleTestDataset"


def test_list_images_sample():
    df = list_images(SAMPLE)
    assert len(df) == 20
    assert sorted(df["label"].unique()) == [
        "MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


def test_list_images_ignores_non_images(tmp_path):
    (tmp_path / "NonDemented").mkdir()
    (tmp_path / "NonDemented" / "a.JPG").write_bytes(b"x")
    (tmp_path / "NonDemented" / "notes.txt").write_text("x")
    (tmp_path / "stray.jpg").write_bytes(b"x")  # not inside a class folder
    df = list_images(str(tmp_path))
    assert df["label"].tolist() == ["NonDemented"]


def test_split_stratified_no_overlap():
    train, val, test = split(list_images(SAMPLE), test_size=0.2, val_size=0.25, seed=42)
    for part in (train, val, test):
        assert part["label"].nunique() == 4
    paths = [set(p["image_path"]) for p in (train, val, test)]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    assert sum(map(len, paths)) == 20
