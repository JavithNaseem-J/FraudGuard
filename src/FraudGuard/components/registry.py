import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from FraudGuard import logger

class Registry:
    def __init__(self, model_name: str = "FraudGuardModel", min_f1_score: float = 0.90):
        mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/FraudGuard.mlflow")
        self.client = MlflowClient()
        self.model_name = model_name
        self.min_f1 = min_f1_score

    def promote_model(self):
        logger.info(f"Checking model versions for promotion: {self.model_name}")
        latest_versions = self.client.search_model_versions(f"name='{self.model_name}'")

        for version in latest_versions:
            run_id = version.run_id
            run = self.client.get_run(run_id)
            metrics = run.data.metrics
            f1 = metrics.get("f1_weighted", 0)

            if f1 >= self.min_f1:
                stage = "Staging" if f1 < 0.95 else "Production"
                logger.info(f"Promoting version {version.version} to {stage} (F1: {f1:.3f})")

                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=version.version,
                    stage=stage,
                    archive_existing_versions=True,
                )
                return True

        logger.warning("No model meets F1 criteria for promotion.")
        return False 
    

