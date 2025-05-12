import pandas as pd
from project.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validation(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_file)
            all_cols = list(data.columns)
            expected_cols = set(self.config.all_schema.keys())  
            validation_status = set(all_cols).issubset(expected_cols)

            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation_status: {validation_status}")
            return validation_status

        except Exception as e:
            raise e