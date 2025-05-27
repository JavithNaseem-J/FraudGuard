import pytest

def test_data_transformation():
    from FraudGuard.config.config import ConfigurationManager
    from FraudGuard.components.data_transformation import DataTransformation
    
    config = ConfigurationManager().get_data_transformation_config()
    transformer = DataTransformation(config)
    train, test = transformer.train_test_splitting()
    assert not train.empty and not test.empty
    train_proc, test_proc = transformer.preprocess_features(train, test)
    assert train_proc.shape[0] == train.shape[0]