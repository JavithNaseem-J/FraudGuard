from project.config.config import ConfigurationManager
from project.components.data_transformation import DataTransformation



class DataTransformationPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train, test = data_transformation.train_test_splitting()
        train_processed, test_processed = data_transformation.preprocess_features(train, test)