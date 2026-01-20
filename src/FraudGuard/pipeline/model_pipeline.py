from FraudGuard.utils.logging import logger
from FraudGuard.config.config import ConfigurationManager
from FraudGuard.components.training import Trainer
from FraudGuard.components.evaluation import Evaluation

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
