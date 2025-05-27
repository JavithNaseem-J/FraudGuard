import os
import zipfile
import gdown
from urllib import request
from pathlib import Path
from FraudGuard import logger
from FraudGuard.utils.helpers import *
from FraudGuard.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.zip_file):
            gdown.download(id=self.config.source_id, output=self.config.zip_file, quiet=False)
            print(f"[INFO] Downloaded file: {self.config.zip_file}")
        else:
            print(f"[INFO] File already exists: {self.config.zip_file} ({get_size(Path(self.config.zip_file))})")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_file
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            print(f"[INFO] Extracted files to: {unzip_path}")