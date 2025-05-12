from project.config.config import ConfigurationManager
from project.components.model_training import ModelTrainer


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_training_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()