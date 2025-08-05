from FraudGuard.utils.logging import logger
from FraudGuard.config.config import ConfigurationManager
from FraudGuard.components.training import Trainer
from FraudGuard.components.evaluation import Evaluation
from FraudGuard.components.registry import Registry    

class ModelPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()

        model_training_config = config.get_model_training_config()
        model_trainer = Trainer(config=model_training_config)
        model_trainer.train()

        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = Evaluation(config=model_evaluation_config)
        model_evaluation.evaluation()



        logger.info("Starting MLflow Model Registry promotion stage...")
        model_name = "FraudGuardModel"
        min_f1_score = 0.90

        registry = Registry(model_name=model_name, min_f1_score=min_f1_score)
        promoted = registry.promote_model()

        if promoted:
            logger.info("Model promotion completed.")
        else:
            logger.info("No model promoted.")