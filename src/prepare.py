"""Stage 1: list images under data.raw_dir/<class>/ and write stratified splits."""
import os
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data/splits")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(raw_dir):
    rows = [
        (p.as_posix(), p.parent.name)
        for p in sorted(Path(raw_dir).glob("*/*"))
        if p.suffix.lower() in IMAGE_EXTS
    ]
    return pd.DataFrame(rows, columns=["image_path", "label"])


def split(df, test_size, val_size, seed):
    rest, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(rest, test_size=val_size, stratify=rest["label"], random_state=seed)
    return train, val, test


def main():
    p = yaml.safe_load(Path("params.yaml").read_text())["data"]
    df = list_images(p["raw_dir"])
    if df.empty:
        raise SystemExit(f"No images found under {p['raw_dir']}/<class>/")
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, part in zip(("train", "val", "test"), split(df, p["test_size"], p["val_size"], p["seed"])):
        part.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(part)}")


if __name__ == "__main__":
    main()
