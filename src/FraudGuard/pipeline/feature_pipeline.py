from FraudGuard.config.config import ConfigurationManager
from FraudGuard.components.ingestion import Ingestion
from FraudGuard.components.validation import Validation     
from FraudGuard.components.preprocess  import Transform

class FeaturePipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()


        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Ingestion(config=data_ingestion_config)
        data_ingestion.download_file()

        data_validation_config = config.get_data_validation_config()
        data_validation = Validation(data_validation_config)
        data_validation.validation()

        data_transformation_config = config.get_data_transformation_config()
        data_transformation = Transform(config=data_transformation_config)
        train, test = data_transformation.train_test_splitting()
        train_processed, test_processed = data_transformation.preprocess_features(train, test)


