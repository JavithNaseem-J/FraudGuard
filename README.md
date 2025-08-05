# FraudGuard 

---

### End 2 End Bank Transaction Fraud Detection

---

## ✨ Project Overview

FraudGuard is an end-to-end machine learning system that detects fraudulent bank transactions with high precision. Built with production-grade MLOps practices, the system handles everything from data ingestion to real-time fraud prediction.

It combines robust data engineering pipelines, modern classification algorithms (XGBoost, CatBoost, LightGBM), automated MLflow tracking, and a CI/CD workflow for seamless deployment.

---

## 🔧 Key Features & Technical Innovations

* **Full ML Lifecycle:** Ingestion → Validation → Transformation → Training → Evaluation → Inference
* **MLflow Model Registry** integrated with DagsHub
* **Optuna Hyperparameter Optimization** with Stratified K-Fold
* **SMOTE-Tomek Hybrid Resampling** to tackle class imbalance
* **Real-Time Predictions** via FastAPI web interface
* **Dockerized App + GitHub Actions CI/CD + AWS ECR deployment**
* **DVC for Data & Pipeline Versioning**
* **Automated Model Promotion Based on F1 Thresholds**

---

## 📚 Detailed Project Structure

```
FraudGuard/
├── app.py                     # FastAPI web app
├── main.py                    # CLI pipeline runner
├── config_file/               # YAML configuration files
├── notebooks/                 # Jupyter notebooks for exploration
├── src/FraudGuard/            # Core source code
│   ├── components/            # Ingestion, Validation, Transformation, etc.
│   ├── config/                # Configuration manager
│   ├── constants/             # Path constants
│   ├── entity/                # Typed dataclass schemas
│   ├── pipeline/              # Orchestrated ML pipelines
│   └── utils/                 # Logging, exceptions, helpers
├── templates/                 # HTML for web UI
├── tests/                     # Unit tests for each module
├── .github/workflows/         # GitHub Actions CI/CD
├── Dockerfile
├── dvc.yaml / dvc.lock        # DVC pipeline definitions
├── requirements.txt / setup.py
```

---

## 🛠️ Installation & Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/JavithNaseem-J/FraudGuard.git
cd FraudGuard
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

* Set AWS & DagsHub credentials as environment variables
* Adjust `config.yaml`, `params.yaml`, `schema.yaml` as needed

### 5. Run the Application

```bash
# Launch CLI pipelines:
python main.py --stage feature_pipeline
python main.py --stage model_pipeline

# Or run full API server:
python app.py  # Or: uvicorn app:app --reload --port 8080
```

---

## 🔎 Usage Instructions

### ✏️ CLI Pipeline (via main.py)

```bash
# Feature pipeline: ingestion → validation → transformation
python main.py --stage feature_pipeline

# Model pipeline: training → evaluation → registry
python main.py --stage model_pipeline
```

### 🛋️ Web App (via app.py)

* Go to `http://localhost:8080`
* Fill in transaction form
* Submit to receive fraud probability & confidence

### 🌐 API Endpoints

* `POST /predict` → fraud prediction
* `GET /results` → UI display

---

## 🏗️ Development & Contribution Workflow

### Adding a New Component

* Create module under `src/FraudGuard/components/`
* Update `pipeline/feature_pipeline.py` or `model_pipeline.py`
* Add test in `tests/`

### Add CI/CD Integration

* Edit `.github/workflows/cicd.yaml`
* Triggered on `push` to `main` branch

### Docker Image Build

```bash
docker build -t fraudguard .
docker run -p 8080:8080 fraudguard
```

---

## 🔢 Robust Testing Methodology

```bash
# Run all tests
pytest tests/
```

Covers:

* Ingestion correctness
* Schema validation
* Preprocessing output shapes
* Model accuracy thresholds
* End-to-end predictions

---

## 🛠️ CI/CD Pipeline Configuration

Powered by **GitHub Actions** with three stages:

1. **Integration**

   * Code linting
   * Unit tests

2. **Build & Push Docker to AWS ECR**

   * Docker image built from latest code
   * Pushed to ECR with secrets & permissions

3. **Deployment** (on self-hosted runner)

   * Pull latest image
   * Stop previous container
   * Run updated container


---

### For Future Enhancements:

* Add Continuously track drift (Evidently)
* Monitor latency + throughput
* Retrain on latest labeled data
* Add **SHAP-based explainability**
* Real-time data ingestion pipeline
* Integrate with **payment gateway APIs**


---

## 📊 Tech Stack

* **ML Libraries:** XGBoost, LightGBM, CatBoost, Optuna
* **Pipeline:** DVC + MLflow + Dagshub
* **Backend:** FastAPI
* **Deployment:** Docker + AWS ECR + GitHub Actions
* **Monitoring:** MLflow, Confusion Matrix, AUC, F1

---

## 📘️ Licensing

Licensed under [MIT License](LICENSE).


![front](https://github.com/user-attachments/assets/8804714a-5cc6-4a69-a21e-24ce76c79f79)

---

![result](https://github.com/user-attachments/assets/078cff2d-71e0-498c-a088-db9c9c714819)
