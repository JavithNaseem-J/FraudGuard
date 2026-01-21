import os
import zipfile
from urllib import request
from pathlib import Path
from FraudGuard import logger
from FraudGuard.utils.helpers import *
from FraudGuard.entity.config_entity import DataIngestionConfig

class Ingestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        local_csv_path = self.config.download_data
        if not os.path.exists(local_csv_path):
            success = download_from_s3(
                bucket=self.config.bucket,
                s3_path=self.config.data_path,
                local_path=local_csv_path,
                aws_region=self.config.region
            )
            if success:
                logger.info(f"Downloaded file: {local_csv_path}")
            else:
                raise Exception(f"Failed to download {self.config.data_path} from S3")
        else:
            logger.info(f"File already exists: {local_csv_path} ({get_size(local_csv_path)})")


if __name__ == "__main__":
    from FraudGuard.config.config import ConfigurationManager
    
    logger.info(">>>>>> Stage: Ingestion started <<<<<<")
    config = ConfigurationManager()
    ingestion = Ingestion(config=config.get_data_ingestion_config())
    ingestion.download_file()
    logger.info(">>>>>> Stage: Ingestion completed <<<<<<")