import shutil
import sys
from pathlib import Path
from FraudGuard.utils.logging import logger
from FraudGuard.entity.config_entity import DataIngestionConfig
from FraudGuard.config.config import ConfigurationManager


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logger.info(f"Initiating data ingestion from {self.config.local_data_path}")

        try:
            source_path = Path(self.config.local_data_path)
            dest_path = Path(self.config.download_data)

            # Create destination directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if source_path.exists():
                logger.info(f"Copying data from {source_path} to {dest_path}")
                shutil.copy(source_path, dest_path)
                logger.info("Data ingestion completed successfully")
            elif dest_path.exists():
                logger.warning(
                    f"Source file {source_path} not found, but destination {dest_path} exists. Using existing file."
                )
            else:
                raise FileNotFoundError(
                    f"Source data file not found at {source_path}. Please place the data file there."
                )

        except Exception as e:
            logger.error(f"Error during data ingestion: {str(e)}")
            raise e


if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logger.error(e)
        sys.exit(1)
