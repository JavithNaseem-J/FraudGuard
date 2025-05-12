# End-to-End Credit Fraud Detection

![image](https://github.com/user-attachments/assets/8d0ebeac-1e3c-4d77-9c31-18b75fdd9783)

**End-to-End Credit Fraud Detection**
This is a comprehensive machine learning project designed to detect fraudulent credit transactions using a modular pipeline, modern MLOps practices, and a user-friendly web application.it showcases advanced skills in data science, machine learning, MLOps, and deployment, making it an ideal portfolio piece for roles in data science, ML engineering, and fintech.


## Project Overview
Credit fraud is a critical issue in the financial industry, leading to billions in losses annually. This project builds an end-to-end machine learning pipeline to detect fraudulent transactions using a dataset with features like `Transaction_Amount` and `Previous_Fraudulent_Transactions`. The pipeline includes data ingestion, validation, transformation, model training with hyperparameter optimization, evaluation, and deployment via a FastAPI web app. The project emphasizes production readiness with Docker containerization, CI/CD via GitHub Actions, and experiment tracking using MLflow and Dagshub.

**Key Objectives**:
- Develop a scalable and modular ML pipeline for fraud detection.
- Implement MLOps practices for reproducibility and experiment tracking.
- Deploy a user-facing web app for real-time fraud predictions.
- Demonstrate industry-relevant skills in ML, MLOps, and deployment.

## Features
- **End-to-End ML Pipeline**:
  - **Data Ingestion**: Downloads and extracts a ZIP file from a remote source.
  - **Data Validation**: Ensures schema consistency using `schema.yaml`.
  - **Data Transformation**: Handles class imbalance with SMOTETomek, preprocesses features with `StandardScaler` and `LabelEncoder`.
  - **Model Training**: Trains multiple models (XGBoost, Random Forest, Logistic Regression) with Optuna hyperparameter optimization, saving the best model.
  - **Model Evaluation**: Evaluates the best model with metrics (accuracy, precision, recall, F1, AUC) and visualizations (confusion matrix, ROC curve).
- **MLOps Integration**:
  - Data versioning with DVC.
  - Experiment tracking with MLflow and Dagshub.
  - CI/CD pipeline using GitHub Actions for building and deploying to AWS ECR.
- **Web Application**:
  - FastAPI-based app for inputting transaction details and predicting fraud status.
  - User-friendly interface with Jinja2 templates (`index.html`, `result.html`).
- **Production-Ready**:
  - Modular code with utility functions (`utils.common`) for consistent file handling.
  - Dockerized application for consistent deployment.
  - Configuration management via `config.yaml` and `schema.yaml`.

## Tech Stack
- **Programming**: Python
- **Machine Learning**: scikit-learn, XGBoost, Optuna, SMOTETomek
- **Models**: XGBoost, Random Forest, Logistic Regression
- **Data Processing**: pandas, numpy
- **MLOps**: DVC, MLflow, Dagshub
- **Web Framework**: FastAPI, Uvicorn, Jinja2
- **Deployment**: Docker, AWS ECR, GitHub Actions
- **Visualization**: matplotlib, seaborn
- **Other**: joblib, pyYAML, boto3

## Project Structure
```
📦 E2E-Credit-Fraud-Detection
├─ .dvc/                  # DVC configuration
├─ artifacts/             # Pipeline outputs (data, models, metrics)
├─ config/                # Configuration files
│  ├─ config.yaml        # Pipeline configurations
│  ├─ params.yaml        # Hyperparameters
│  └─ schema.yaml        # Data schema
├─ src/project/           # Source code
│  ├─ components/        # Pipeline components (ingestion, validation, etc.)
│  ├─ config/            # Configuration manager
│  ├─ entity/            # Data classes for config
│  ├─ pipeline/          # Pipeline stages and prediction
│  └─ utils/             # Utility functions (common.py)
├─ templates/             # HTML templates for FastAPI
├─ Exp/                   # Jupyter notebooks for EDA
├─ app.py                 # FastAPI web application
├─ main.py                # Pipeline orchestrator
├─ test.py                # Prediction tests
├─ Dockerfile             # Docker configuration
├─ requirements.txt       # Python dependencies
├─ dvc.yaml               # DVC pipeline definitions
├─ cicd.yaml              # GitHub Actions CI/CD
└─ README.md              # Project documentation
```

## Setup Instructions
### Prerequisites
- Python 3.8+
- Docker (for containerized deployment)
- AWS CLI (for ECR deployment)
- Git, DVC, and pip

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JavithNaseem-J/E2E-Credit-Fraud-Detection.git
   cd E2E-Credit-Fraud-Detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up DVC**:
   ```bash
   dvc init
   dvc remote add -d myremote s3://your-bucket-name  # Update with your S3 bucket
   dvc pull
   ```

5. **Configure AWS Credentials** (for ECR deployment):
   - Set up AWS credentials in `~/.aws/credentials` or as environment variables:
     ```bash
     export AWS_ACCESS_KEY_ID=your_access_key
     export AWS_SECRET_ACCESS_KEY=your_secret_key
     export AWS_REGION=your_region
     ```

## Running the Pipeline
The pipeline is orchestrated via `main.py` and can be run stage-by-stage or end-to-end.

1. **Run a Specific Stage**:
   ```bash
   python main.py --stage data_ingestion
   python main.py --stage data_validation
   python main.py --stage data_transformation
   python main.py --stage model_training
   python main.py --stage model_evaluation
   ```

2. **Run All Stages**:
   ```bash
   python main.py
   ```

3. **Reproduce with DVC**:
   ```bash
   dvc repro
   ```

**Outputs**:
- Data: `artifacts/data_ingestion/Fraud-data.csv`
- Preprocessed data: `artifacts/data_transformation/split/{train,test}.csv`
- Models: `artifacts/model_trainer/{XGBoost,RandomForest,LogisticRegression}_model.joblib`
- Best Model: `artifacts/model_trainer/model.joblib`
- Metrics: `artifacts/model_evaluation/metrics.json`
- Visualizations: `artifacts/model_evaluation/{cm,roc}.png`

## Using the Web App
The FastAPI web app allows users to input transaction details and predict fraud status.

1. **Start the Server**:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8080
   ```

2. **Access the App**:
   - Open `http://127.0.0.1:8080` in a browser.
   - Enter transaction details (e.g., `Transaction_Amount`, `Device_Used`).
   - Submit to view the fraud status and probability.

**Screenshots**:
![image](https://github.com/user-attachments/assets/68ae819c-5197-4520-b7b3-bd2d82b7d2bc)

![image](https://github.com/user-attachments/assets/a3943512-cbec-4f4d-92de-cd13ebd26f25)


## CI/CD and Deployment
The project uses GitHub Actions for continuous integration and deployment to AWS ECR.

1. **CI/CD Workflow** (`cicd.yaml`):
   - **Integration**: Lints code and runs tests (to be implemented).
   - **Build and Push**: Builds a Docker image and pushes it to AWS ECR.
   - **Deployment**: Pulls the image and runs it on a self-hosted server.

2. **Deploy Locally with Docker**:
   ```bash
   docker build -t credit-fraud-detection .
   docker run -d -p 8080:8080 \
     -e AWS_ACCESS_KEY_ID=your_access_key \
     -e AWS_SECRET_ACCESS_KEY=your_secret_key \
     -e AWS_REGION=your_region \
     credit-fraud-detection
   ```

## ML Experiments
Experiments are tracked using MLflow and hosted on Dagshub.

- **View Experiments**:
  - URL: [Dagshub MLflow Dashboard](https://dagshub.com/JavithNaseem-J/E2E-Credit-Fraud-Detection.mlflow)
  - Metrics: Best cross-validation scores for each model.
  - Models: XGBoost, Random Forest, Logistic Regression.

- **Key Experiment**:
  - Models trained: XGBoost, Random Forest, Logistic Regression.
  - Hyperparameter tuning: Optuna with Stratified K-Fold cross-validation.
  - Best model selected based on accuracy and saved for evaluation.

## Challenges and Solutions
1. **Class Imbalance**:
   - **Challenge**: The fraud dataset had imbalanced classes (few fraudulent transactions).
   - **Solution**: Applied SMOTETomek to balance the dataset during data transformation.

2. **Model Selection**:
   - **Challenge**: Choosing the best model for fraud detection.
   - **Solution**: Trained multiple models (XGBoost, Random Forest, Logistic Regression) with Optuna hyperparameter optimization, selecting the best performer.

3. **Code Maintainability**:
   - **Challenge**: Ensuring consistent file handling across pipeline stages.
   - **Solution**: Centralized file operations (e.g., saving/loading models, JSON files) in `utils.common` for consistency and maintainability.


## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

