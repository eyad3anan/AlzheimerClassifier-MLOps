# **AlzheimerClassifierTL: ResNet50-Based Alzheimer’s MRI Classifier**

AlzheimerClassifierTL applies **transfer learning with ResNet50** to classify **brain MRI scans** into four cognitive health categories:  
🟢 **NonDemented** • 🟡 **Very Mild Demented** • 🟠 **Mild Demented** • 🔴 **Moderate Demented**

This project delivers a complete **computer vision pipeline**—from **data exploration and augmentation** to **model training, evaluation, and real-time deployment** using **Streamlit** and **Hugging Face**.

---

## **Demo**
![App Demo](./Alzheimer%20Detection%20App.png)  
![No Dementia Example](./No%20Dementia.png)  
![Very Mild Dementia Example](./Very%20Mild%20Dementia.png)  
![Mild Dementia Example](./Mild%20Dementia.png)  
![Moderate Dementia Example](./Moderate%20Dementia.png)

---

## **Objective**

Develop and deploy a robust deep learning model to support **early detection of Alzheimer’s disease**, a progressive neurological disorder affecting **memory and cognitive function**.  
Accurate staging of dementia enables timely medical consultation and care.

---

## **Dataset**

- **Source**: [Kaggle – Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data)  
- **Preparation**:  
  - Images resized to **224×224 (RGB)**  
  - Normalized with **ResNet50 preprocess_input**  
  - Balanced **train/validation/test** splits  

---

## **Modeling Approach**

- **Base Network**: **ResNet50** pre-trained on ImageNet  
- **Custom Layers**: Global max pooling, batch normalization, dense layer (256 units, L1/L2 regularization), dropout (0.45)  
- **Output Layer**: Softmax activation for 4-class prediction  
- **Training Setup**:  
  - Loss: **Categorical Crossentropy**  
  - Optimizer: **Adamax** (learning rate 0.001)  
  - Augmentation: rotation, width/height shift, zoom (training set only)  
  - Callbacks: **EarlyStopping** & **ModelCheckpoint** to prevent overfitting  

---

## **Performance**

Final model results on the test set:  
- **Accuracy**: `98.3%`  
- **Precision**: `98.9%`  
- **Recall**: `93.8%`  
- **F1 Score**: `96.1%`

---

## **Deployment**

The trained model powers an interactive **Streamlit web app** and a **Hugging Face Space** for real-time predictions.  
Users can upload an MRI image and receive **instant, color-coded stage classification**.

---

## **Tech Stack**

**Languages & Libraries**  
- Python, Pandas, NumPy  
- TensorFlow / Keras, scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  

**Techniques**  
- Transfer Learning (**ResNet50 fine-tuning**)  
- Data Augmentation (rotation, shift, zoom)  
- EarlyStopping & ModelCheckpoint  
- Real-time deployment with **Streamlit** & **Hugging Face**

---

## **MLOps: CI / CT / CD**

```
params.yaml ─► prepare ─► train ─► evaluate        (dvc.yaml — `dvc repro`)
                splits/   models/model.weights.h5   metrics.json
```

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | every PR / push to main | ruff, pytest, full `dvc` pipeline smoke run on `Data/SampleTestDataset` (1 epoch), MLflow log |
| `train.yml` (CT) | manual, or push to main changing `params.yaml` / `dvc.yaml` / training code | runs `dvc repro` on a **Kaggle GPU** kernel, `dvc push` to DagsHub, logs the run to DagsHub MLflow, commits `dvc.lock` + `metrics.json`, then calls CD |
| `cd.yml` (CD) | after CT, push changing `dvc.lock`, or manual | gate: test accuracy ≥ `evaluate.min_accuracy` → upload weights to the HF model repo → deploy the app to the HF Space |

### One-time setup
1. **DagsHub:** create a repo connected to this GitHub repo (DVC remote + MLflow server). If your DagsHub
   username isn't `eyad3anan`, edit the URL in `.dvc/config`.
2. **Kaggle:** phone-verify the account (GPU + internet in kernels), create an API token (Settings → API).
3. **GitHub → Settings → Secrets and variables → Actions:**
   - secrets: `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN` (the `KGAT_...` token), `DAGSHUB_TOKEN`, `HF_TOKEN` (write access)
   - variables: `HF_SPACE_ID` (e.g. `user/space`), optional `DAGSHUB_USER`, `HF_MODEL_REPO`
   - optional: add required reviewers to the `production` environment for a manual approval before deploy.
4. Create the HF Space with SDK **Docker** (free CPU basic). CD uploads `deploy/space/Dockerfile` and
   `deploy/space/README.md` with the app. Add an `HF_TOKEN` secret in the Space settings so the app can
   download the private model.

### Local commands
```bash
uv sync                          # runtime + train + dev deps
uv run pytest                    # unit tests
# smoke-run the whole pipeline on the sample images
uv run dvc exp run -S data.raw_dir=Data/SampleTestDataset -S data.test_size=0.2 \
  -S data.val_size=0.25 -S train.epochs=1 -S train.batch_size=4
uv run python -m src.log_mlflow  # log the run (./mlflow.db unless MLFLOW_TRACKING_URI is set)
uv run mlflow ui                 # browse runs
```
To retrain: change `params.yaml`, push to main (or run **Train (Kaggle GPU)** from the Actions tab).
