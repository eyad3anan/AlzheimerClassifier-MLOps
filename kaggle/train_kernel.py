"""Runs on a Kaggle GPU kernel: clone the repo at the CI commit, `dvc repro`, export outputs.

__REPO__ and __SHA__ are filled in by .github/workflows/train.yml. No secrets come here:
MLflow logging and `dvc push` happen back on the GitHub runner.
"""
import re
import shutil
import subprocess
from pathlib import Path

REPO, SHA = "__REPO__", "__SHA__"
WORK, OUT, INPUT = Path("/tmp/repo"), Path("/kaggle/working"), Path("/kaggle/input")
EXPORTS = ("params.yaml", "dvc.lock", "metrics.json", "models", "reports", "splits")


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def find_dataset(root=INPUT):
    """Kaggle's mount layout for attached datasets changes over time, so search for it."""
    for depth in range(1, 6):
        found = sorted(Path(root).glob("*/" * depth + "OriginalDataset"))
        if found:
            return found[0].as_posix()
    mounted = sorted(p.relative_to(root).as_posix() for g in ("*", "*/*") for p in Path(root).glob(g))
    raise SystemExit(f"OriginalDataset not found under {root}. Mounted: {mounted or 'nothing'}")


def set_raw_dir(params_text, raw_dir):
    return re.sub(r"(?m)^(\s*raw_dir:).*$", rf"\g<1> {raw_dir}", params_text, count=1)


def main():
    import tensorflow as tf  # preinstalled on Kaggle

    run(["git", "clone", "--quiet", REPO, str(WORK)])
    run(["git", "checkout", "--quiet", SHA], cwd=WORK)
    run(["pip", "install", "--quiet", "dvc>=3.50"])
    print("TensorFlow", tf.__version__, "GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

    raw_dir = find_dataset()
    print("Dataset:", raw_dir, flush=True)
    params = WORK / "params.yaml"
    params.write_text(set_raw_dir(params.read_text(), raw_dir))
    run(["dvc", "repro"], cwd=WORK)

    for rel in EXPORTS:
        src, dst = WORK / rel, OUT / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    (OUT / "sha.txt").write_text(SHA)  # lets the workflow reject outputs from an older version
    print((WORK / "metrics.json").read_text())


if __name__ == "__main__":
    main()
