# MLOps Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `dvc repro` pipeline (prepare → train → evaluate) driven by `params.yaml`, MLflow logging, and GitHub Actions for CI, Kaggle-GPU CT and Hugging Face CD.

**Architecture:** Pure-Python stage scripts in `src/` read `params.yaml`, write files declared in `dvc.yaml`. MLflow logging is a separate script that reads those files, so it runs wherever results land. Workflows orchestrate: CI smoke-runs the pipeline, `train.yml` runs it on Kaggle, `cd.yml` gates and ships.

**Tech Stack:** Python 3.11, uv, TensorFlow 2.21 / Keras 3, DVC, MLflow, DagsHub, Kaggle CLI, huggingface_hub, pytest, ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-14-mlops-pipeline-design.md`

## Global Constraints

- Class order is alphabetical: `MildDemented=0, ModerateDemented=1, NonDemented=2, VeryMildDemented=3` (matches app and notebook).
- Image size 224×224 RGB, nearest-neighbour resize, `resnet50.preprocess_input` — identical in train and app.
- Model head: ResNet50(include_top=False, pooling="max") → BatchNorm → Dense(256 relu, l2 0.001 kernel / l1 0.0001 activity / l1 1e-5 bias) → Dropout(0.45) → Dense(4 softmax). Optimizer Adamax(lr=0.001).
- Weights file: `models/model.weights.h5`; HF model repo `Eyaddddddd/AlzheimerClassifierTL`, filename `model.weights.h5`.
- Kaggle dataset `uraninjo/augmented-alzheimer-mri-dataset`, raw dir `/kaggle/input/augmented-alzheimer-mri-dataset/OriginalDataset`.
- Runtime deps unchanged (streamlit, tensorflow, huggingface-hub); training deps in uv group `train`, test tooling in `dev`.
- No secrets in Kaggle or in the repo.

## Review Focus

1. Sample data too small to stratify (20 images, 4 classes) → smoke params must give ≥4 test and ≥4 val images; test pins `prepare` on sample data with smoke params.
2. Non-`.jpg` / uppercase-extension files in the raw dir → `.jpg`/`.jpeg`/`.png` any case accepted, others ignored; test with a stray `.txt`.
3. Grayscale/RGBA uploads to the app → `preprocess_image` returns shape (1,224,224,3); test with "L" and "RGBA" images.
4. Model accuracy below threshold → CD gate exits non-zero and nothing is uploaded; test for `gate.py`.
5. Weights saved by train load into the app's `build_model` → round-trip test save/load.

---

### Task 1: Shared model + preprocessing, app uses it

**Files:** Create `src/__init__.py`, `src/model.py`, `tests/test_model.py`; Modify `app.py`, `pyproject.toml` (dependency groups, ruff/pytest config).

**Interfaces — Produces:**
- `CLASSES: list[str]` (order above), `IMG_SIZE = (224, 224)`
- `build_model(weights: str | None = None, dropout: float = 0.45, dense_units: int = 256) -> keras.Model` (uncompiled)
- `preprocess_image(image: PIL.Image.Image) -> np.ndarray` shape (1,224,224,3) float32

- [ ] Tests: `test_build_model_outputs_4_probs` (predict on zeros → shape (1,4), sums to 1), `test_preprocess_handles_modes` (L, RGBA, RGB 100×80 → (1,224,224,3)), `test_weights_roundtrip` (save_weights to tmp `x.weights.h5`, load into new `build_model()`, predictions equal).
- [ ] Run → FAIL (module missing). Implement. Run → PASS.
- [ ] `app.py`: delete local `build_model`/`preprocess_image`/`class_mapping`, import from `src.model`, download `model.weights.h5`.
- [ ] Commit.

### Task 2: params.yaml + prepare stage

**Files:** Create `params.yaml`, `src/prepare.py`, `tests/test_prepare.py`.

**Interfaces — Produces:** `list_images(raw_dir: str) -> pd.DataFrame[image_path,label]`; `split(df, test_size, val_size, seed) -> (train, val, test)`; CLI writes `data/splits/{train,val,test}.csv`.

params.yaml keys: `data.raw_dir, data.test_size=0.1, data.val_size=0.1, data.seed=42`; `train.epochs=20, batch_size=32, learning_rate=0.001, dropout=0.45, dense_units=256, patience=10, weights=imagenet`; `evaluate.min_accuracy=0.95`.

- [ ] Tests: `test_list_images_sample` (Data/SampleTestDataset → 20 rows, 4 labels, ignores stray .txt in tmp dir), `test_split_stratified_no_overlap` (smoke sizes 0.2/0.25 on sample → each split has all 4 labels, paths disjoint, total 20).
- [ ] FAIL → implement → PASS → commit.

### Task 3: train + evaluate stages, dvc.yaml

**Files:** Create `src/train.py`, `src/evaluate.py`, `dvc.yaml`; init DVC (`.dvc/`, `.dvcignore`).

**Interfaces — Produces:** `make_dataset(df, batch_size, training: bool) -> tf.data.Dataset`; train writes `models/model.weights.h5`, `reports/history.json` ({metric: [per-epoch]}); evaluate writes `metrics.json` {accuracy, precision, recall, f1} (macro) and `reports/confusion_matrix.json`.

- [ ] Test `test_make_dataset_shapes` (batch from sample → x (b,224,224,3), y one-hot (b,4)).
- [ ] Implement; augmentation via RandomRotation(1/360), RandomTranslation(0.01,0.01), RandomZoom(0.1) on train only.
- [ ] Verify: `dvc exp run -S data.raw_dir=Data/SampleTestDataset -S data.test_size=0.2 -S data.val_size=0.25 -S train.epochs=1` completes; `dvc metrics show` prints metrics. Commit.

### Task 4: MLflow logging + CD gate

**Files:** Create `src/log_mlflow.py`, `src/gate.py`, `tests/test_gate.py`.

**Interfaces:** `log_mlflow.py` CLI: logs params.yaml (flattened), history per epoch (step=epoch), metrics.json, artifacts (weights, confusion matrix). Experiment `alzheimer-resnet50`. `gate.py`: `passes(metrics: dict, min_accuracy: float) -> bool`; CLI exits 1 on failure.

- [ ] Tests: `test_gate_pass` (0.97 ≥ 0.95), `test_gate_fail` (0.90).
- [ ] Verify: `python -m src.log_mlflow` after smoke run creates a run in `./mlruns`. Commit.

### Task 5: Workflows + Kaggle kernel

**Files:** Create `.github/workflows/{ci,train,cd}.yml`, `kaggle/kernel-metadata.json`, `kaggle/train_kernel.py`; update README.

- `ci.yml`: uv sync groups → ruff → pytest → smoke `dvc exp run` (Task 3 command) → `log_mlflow` locally.
- `train.yml`: kernel script templated with repo URL + SHA, `kaggle kernels push`, poll status (30 s → 20 min, max 5 h), `kaggle kernels output`, copy outs into workspace, `dvc commit -f`, `dvc push`, `log_mlflow` (DagsHub env), commit `dvc.lock metrics.json reports/`, then `uses: ./.github/workflows/cd.yml`.
- `cd.yml`: gate → `dvc pull train` → `huggingface_hub upload` weights + metrics → `upload_folder` app files (`app.py`, `src/model.py`, `src/__init__.py`, `Alzheimer.png`, generated `requirements.txt`) to `HF_SPACE_ID`.
- [ ] Verify: YAML parses (`python -c yaml.safe_load`), `actionlint` if available; kernel script `python -m py_compile`. Commit.
