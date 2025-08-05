import pytest

def test_data_validation():
    from FraudGuard.config.config import ConfigurationManager
    from FraudGuard.components.validation import DataValidation

    config = ConfigurationManager().get_data_validation_config()
    validation = DataValidation(config)
    status = validation.validation()
    assert isinstance(status, bool)