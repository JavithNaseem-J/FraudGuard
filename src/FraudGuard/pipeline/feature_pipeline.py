from FraudGuard.config.config import ConfigurationManager
from FraudGuard.components.ingestion import DataIngestion
from FraudGuard.components.validation import Validation
from FraudGuard.components.preprocess import Transform


class FeaturePipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()

        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()

        data_validation_config = config.get_data_validation_config()
        data_validation = Validation(data_validation_config)
        data_validation.validation()

        data_transformation_config = config.get_data_transformation_config()
        data_transformation = Transform(config=data_transformation_config)
        data_transformation.train_test_splitting()
