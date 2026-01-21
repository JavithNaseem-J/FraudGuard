# 🛡️ FraudGuard

<div align="center">

**End-to-End Bank Transaction Fraud Detection System**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-purple.svg)](https://dvc.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

*A production-grade ML pipeline for detecting fraudulent bank transactions with modern MLOps practices*

</div>

---

## 🏗️ Architecture Overview

```mermaid
flowchart TB
    subgraph DATA["📥 DATA LAYER"]
        direction TB
        S3[("☁️ AWS S3<br>Raw Data")]
        S3 --> ING["📂 Ingestion<br>Download & Store"]
        ING --> VAL["✅ Validation<br>Schema Check"]
        VAL --> PRE["⚙️ Preprocessing<br>Transform & Split"]
    end
    
    subgraph ML["🤖 ML LAYER"]
        direction TB
        PRE --> SMT["⚖️ SMOTE-Tomek<br>Class Balancing"]
        SMT --> TRN["🎯 Training<br>XGBoost & CatBoost"]
        TRN --> HPO["🔧 Optuna HPO<br>Stratified K-Fold"]
        HPO --> EVL["📊 Evaluation<br>Metrics & SHAP"]
    end
    
    subgraph TRACK["📈 TRACKING LAYER"]
        direction TB
        EVL --> MLF["📋 MLflow<br>Experiment Tracking"]
        MLF --> DH["🗄️ DagsHub<br>Model Registry"]
    end
    
    subgraph DEPLOY["🚀 DEPLOYMENT LAYER"]
        direction TB
        DH --> API["⚡ FastAPI<br>REST Service"]
        API --> DCK["🐳 Docker<br>Container"]
        DCK --> ECR["☁️ AWS ECR<br>Production"]
    end
    
    DATA --> ML --> TRACK --> DEPLOY
    
    style DATA fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style ML fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style TRACK fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style DEPLOY fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

---

## 🔄 DVC Pipeline Stages

```mermaid
flowchart TB
    A["🗃️ <b>ingestion</b><br>python -m FraudGuard.components.ingestion"]
    B["✅ <b>validation</b><br>python -m FraudGuard.components.validation"]
    C["⚙️ <b>preprocess</b><br>python -m FraudGuard.components.preprocess"]
    D["🎯 <b>training</b><br>python -m FraudGuard.components.training"]
    E["📊 <b>evaluation</b><br>python -m FraudGuard.components.evaluation"]
    
    A --> B --> C --> D --> E
    
    style A fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style B fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style C fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    style D fill:#ffccbc,stroke:#e64a19,stroke-width:2px
    style E fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
```

> 💡 **Pro Tip:** Run `dvc repro` to execute the pipeline. DVC caches completed stages, so if training fails, just fix the error and run `dvc repro` again - it resumes from where it stopped!

---

## ⚡ Quickstart

### Prerequisites
- Python 3.9+
- AWS credentials (for S3 data access)
- Git & DVC installed

### 1️⃣ Clone & Setup
```bash
git clone https://github.com/JavithNaseem-J/FraudGuard.git
cd FraudGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.lock
```

### 2️⃣ Configure Environment
```bash
# Windows PowerShell
$env:AWS_PROFILE = "your-aws-profile"
$env:AWS_REGION = "us-east-1"
$env:MLFLOW_TRACKING_USERNAME = "your-dagshub-username"
$env:MLFLOW_TRACKING_PASSWORD = "your-dagshub-token"

# Linux/Mac
export AWS_PROFILE=your-aws-profile
export AWS_REGION=us-east-1
export MLFLOW_TRACKING_USERNAME=your-dagshub-username
export MLFLOW_TRACKING_PASSWORD=your-dagshub-token
```

### 3️⃣ Run Pipeline
```bash
# Run full pipeline with caching
dvc repro

# Or run individual stages
python -m FraudGuard.components.ingestion
python -m FraudGuard.components.training
```

### 4️⃣ Start Web App
```bash
uvicorn app:app --reload --port 8080
# Navigate to http://localhost:8080
```

---

## 🚀 Key Features

| Feature | Description |
|:--------|:------------|
| 🔄 **DVC Pipeline** | Cached, reproducible ML pipeline with `dvc repro` |
| 📊 **Experiment Tracking** | MLflow + DagsHub for metrics, parameters, artifacts |
| ⚖️ **Class Imbalance** | SMOTE-Tomek hybrid resampling (train only, no leakage) |
| 🔧 **HPO** | Optuna with Stratified K-Fold cross-validation |
| 🧠 **Interpretability** | SHAP feature importance plots |
| 🎯 **Dynamic Threshold** | Optimal threshold from Precision-Recall curve |
| ⚡ **Production API** | FastAPI with HTML templates |
| 🐳 **Docker Ready** | One-command containerized deployment |

---

## 📁 Project Structure

```
FraudGuard/
├── 📄 app.py                      # FastAPI web application
├── 📄 dvc.yaml                    # DVC pipeline definition
├── 🐳 Dockerfile                  # Container configuration
│
├── 📁 config_file/
│   ├── config.yaml                # Paths and artifact locations
│   ├── params.yaml                # Hyperparameters
│   └── schema.yaml                # Data schema
│
├── 📁 src/FraudGuard/
│   ├── 📁 components/             # 🎯 Pipeline stages (DVC entry points)
│   │   ├── ingestion.py           # S3 data download
│   │   ├── validation.py          # Schema validation
│   │   ├── preprocess.py          # Feature engineering + SMOTE
│   │   ├── training.py            # Model training with HPO
│   │   └── evaluation.py          # Metrics + SHAP plots
│   │
│   ├── 📁 pipeline/
│   │   ├── feature_pipeline.py    # Data processing pipeline
│   │   ├── model_pipeline.py      # Training + evaluation
│   │   └── inference_pipeline.py  # Production inference
│   │
│   ├── 📁 entity/
│   │   └── config_entity.py       # Pydantic config models
│   │
│   └── 📁 utils/
│       ├── helpers.py             # Utility functions
│       └── logging.py             # Custom logger
│
├── 📁 templates/                  # HTML templates for web UI
├── 📁 artifacts/                  # Generated outputs (DVC tracked)
└── 📁 tests/
    └── test_core.py               # Unit tests
```

---

## 🧪 Testing

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"
pytest tests/test_core.py -v

# Linux/Mac
PYTHONPATH=src pytest tests/test_core.py -v
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t fraudguard .

# Run
docker run -p 8080:8080 \
  -e AWS_PROFILE=your-profile \
  -e MLFLOW_TRACKING_USERNAME=your-username \
  -e MLFLOW_TRACKING_PASSWORD=your-token \
  fraudguard
```

---

## 📈 Model Performance

| Metric | Description |
|:-------|:------------|
| 🎯 **F1 Score (Weighted)** | Primary optimization target |
| ⚖️ **Precision / Recall** | Managed via optimal threshold |
| 📈 **AUC-ROC** | Overall discrimination ability |
| 🔲 **Confusion Matrix** | Visual prediction analysis |
| 🧠 **SHAP Plots** | Feature importance & interpretability |

---

## ⚙️ Configuration

### `config_file/params.yaml`
```yaml
train_test_split:
  test_size: 0.2
  random_state: 42

cross_validation:
  cv_folds: 5
  scoring: f1
  n_iter: 20      # Optuna trials
  n_jobs: -1      # Parallel jobs
```

---

## 🔧 DVC Commands Reference

| Command | Description |
|:--------|:------------|
| `dvc repro` | Run full pipeline (cached) |
| `dvc repro training` | Run up to training stage |
| `dvc repro -s training` | Run only training stage |
| `dvc dag` | View pipeline DAG |
| `dvc metrics show` | Show evaluation metrics |
| `dvc plots show` | Generate metric plots |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ for Production ML**

[⬆ Back to Top](#️-fraudguard)

</div>
