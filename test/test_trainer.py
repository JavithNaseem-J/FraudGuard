import pytest

def test_model_training():
    from FraudGuard.config.config import ConfigurationManager
    from FraudGuard.components.model_training import ModelTrainer
    
    config = ConfigurationManager().get_model_training_config()
    trainer = ModelTrainer(config)
    result = trainer.train()
    assert "model_name" in result