from FraudGuard.config.config import ConfigurationManager
from FraudGuard.components.data_ingestion import DataIngestion   

class DataIngestionPipeline:
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()