import pytest

def test_model_evaluation():
    from FraudGuard.config.config import ConfigurationManager
    from FraudGuard.components.model_evaluation import ModelEvaluation

    config = ConfigurationManager().get_model_evaluation_config()
    evaluator = ModelEvaluation(config)
    metrics = evaluator.evaluation()
    assert "accuracy" in metrics