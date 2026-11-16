# MLOps pipeline (CI / CT / CD + DVC + MLflow) — design

## Intent
Turn the Kaggle notebook + Streamlit app into a reproducible MLOps project:
`params.yaml` + `dvc.yaml` (`dvc repro`), MLflow experiment tracking, and GitHub
Actions for CI (quality), CT (retrain on Kaggle GPU) and CD (ship a passing model).

## Decisions (user)
- **CT:** real training runs on a **Kaggle GPU kernel**, launched and collected by
  GitHub Actions. CI also smoke-runs the whole `dvc repro` on the 20 sample images.
- **Tracking/storage:** DagsHub — hosted MLflow server + DVC remote.
- **CD:** Hugging Face — weights to model repo `Eyaddddddd/AlzheimerClassifierTL`,
  app code to the HF Space.

## Approaches considered
| Approach | Trade-off | Picked |
|---|---|---|
| GitHub Actions drives a Kaggle GPU kernel | Free GPU, fully automated; job waits/polls while Kaggle trains | ✅ |
| Smoke-train in CI, real train local | Free, simple; real training is manual | (still works as fallback) |
| Full train on GitHub CPU runner | Hours on CPU, hits 6h limit | |
| Self-hosted GPU runner | Machine must stay online | |

## Design
```
params.yaml ──► prepare ──► train ──► evaluate          (dvc.yaml, run by `dvc repro`)
            data/splits/*.csv  models/model.weights.h5   metrics.json, history.json
```
- `src/model.py` — `build_model`, `preprocess_image`, shared by training **and** `app.py`.
- `src/prepare.py` — scan `data.raw_dir`, stratified train/val/test split → CSVs.
- `src/train.py` — tf.data pipeline (ImageDataGenerator is gone in Keras 3),
  ResNet50 + notebook head, EarlyStopping; writes weights + `history.json`.
- `src/evaluate.py` — accuracy / macro precision / recall / F1 → `metrics.json`.
- `src/log_mlflow.py` — logs params, per-epoch history, test metrics and weights to
  MLflow from those files. Runs wherever the results land (GitHub runner, local), so
  DagsHub credentials never go to Kaggle. No `MLFLOW_TRACKING_URI` → local `./mlruns`.
- **Data:** the Kaggle dataset is the source of truth (static, versioned by Kaggle).
  `data.raw_dir` defaults to its `/kaggle/input/...` path; it is a param, not a DVC dep.
  CI/local override it with `dvc exp run -S data.raw_dir=...`.

### Workflows
- `ci.yml` (PR + push): ruff, pytest, smoke `dvc exp run` on `Data/SampleTestDataset`
  with 1 epoch.
- `train.yml` (manual `workflow_dispatch`, plus push to main touching `params.yaml`/`src/`):
  1. `kaggle kernels push` a GPU kernel (`kaggle/kernel-metadata.json`, dataset
     `uraninjo/augmented-alzheimer-mri-dataset`) whose script clones this repo at the
     triggering SHA, `pip install dvc`, runs `dvc repro`.
  2. Poll `kaggle kernels status` until complete; `kaggle kernels output` downloads
     `dvc.lock`, model, splits, metrics.
  3. `dvc commit` + `dvc push` to DagsHub, `log_mlflow.py` to DagsHub MLflow,
     commit `dvc.lock` + `metrics.json` to main.
  4. Call `cd.yml` (reusable) — pushes made by the Actions token don't trigger workflows.
- `cd.yml` (`workflow_call`, `workflow_dispatch`, push to main touching `dvc.lock`):
  gate `accuracy >= evaluate.min_accuracy` → `dvc pull` model → upload to HF model repo
  → upload app files to the HF Space.
- App downloads `model.weights.h5` (Keras 3 weights format) instead of the legacy `.h5`.
- Dependencies: runtime stays streamlit/tensorflow/huggingface-hub; `train` group adds
  dvc, mlflow, scikit-learn, pandas, kaggle; `dev` group adds pytest, ruff.

## Non-goals
Docker image, model registry stages, champion-vs-challenger comparison, monitoring.

## Left to the user (needs accounts)
GitHub secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`, `DAGSHUB_TOKEN`, `HF_TOKEN`; variable
`HF_SPACE_ID`; DagsHub repo connected to the GitHub repo. Phone-verified Kaggle account
(required for GPU + internet in kernels).
