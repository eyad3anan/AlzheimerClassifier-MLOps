"""Runs on a Kaggle GPU kernel: clone the repo at the CI commit, `dvc repro`, export outputs.

__REPO__ and __SHA__ are filled in by .github/workflows/train.yml. No secrets come here:
MLflow logging and `dvc push` happen back on the GitHub runner.
"""
import shutil
import subprocess
from pathlib import Path

REPO, SHA = "__REPO__", "__SHA__"
WORK, OUT = Path("/tmp/repo"), Path("/kaggle/working")


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


run(["git", "clone", "--quiet", REPO, str(WORK)])
run(["git", "checkout", "--quiet", SHA], cwd=WORK)
run(["pip", "install", "--quiet", "dvc>=3.50"])
run(["dvc", "repro"], cwd=WORK)  # params.yaml raw_dir already points at /kaggle/input

for rel in ("dvc.lock", "metrics.json", "models", "reports", "splits"):
    src, dst = WORK / rel, OUT / rel
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
print((WORK / "metrics.json").read_text())
