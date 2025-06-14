
from FraudGuard import logger
from FraudGuard.components.model_registry import ModelRegistry

class ModelRegistryPipeline:
    def run(self):
        logger.info("Starting MLflow Model Registry promotion stage...")
        model_name = "FraudGuardModel"
        min_f1_score = 0.90

        registry = ModelRegistry(model_name=model_name, min_f1_score=min_f1_score)
        promoted = registry.promote_model()

        if promoted:
            logger.info("Model promotion completed.")
        else:
            logger.info("No model promoted.")
