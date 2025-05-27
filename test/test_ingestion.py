import pytest
from pathlib import Path

def test_data_ingestion():
    from FraudGuard.config.config import ConfigurationManager
    from FraudGuard.components.data_ingestion import DataIngestion

    config = ConfigurationManager().get_data_ingestion_config()
    ingestion = DataIngestion(config)
    ingestion.download_file()
    ingestion.extract_zip_file()
    assert Path(config.zip_file).exists()
    print("✅ Data ingestion test passed.")