import json
import pandas as pd
from FraudGuard.entity.config_entity import DataValidationConfig
from FraudGuard import logger

class Validation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_data_types(self, data: pd.DataFrame, schema: dict) -> bool:
        """Validates the data types of columns against the schema."""
        type_mapping = {
            'int': ['int64', 'int32'],
            'float': ['float64', 'float32'],
            'object': ['object'],
            'str': ['object'], 
        }

        for col, expected_type in schema.items():
            if col not in data.columns:
                continue 
                
            actual_dtype = str(data[col].dtype)
            allowed_dtypes = type_mapping.get(expected_type, [expected_type])

            if actual_dtype not in allowed_dtypes:
                logger.error(f"Column '{col}': Expected type '{expected_type}', got '{actual_dtype}'")
                return False
        return True

    def validate_column_presence(self, data: pd.DataFrame, schema: dict) -> bool:
        """Validates that all required columns are present in the data."""
        all_cols = list(data.columns)
        expected_cols = set(schema.keys())
        missing_cols = expected_cols - set(all_cols)

        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False
        return True


    def validation(self) -> bool:
            data = pd.read_csv(self.config.unzip_file, low_memory=False)
            schema = self.config.all_schema

            logger.info(f"Starting validation for data with shape: {data.shape}")
            
            validation_results = {}
            
            validation_results['column_presence'] = self.validate_column_presence(data, schema)
            validation_results['data_types'] = self.validate_data_types(data, schema)
            
            is_valid = all(validation_results.values())
            
            for check, result in validation_results.items():
                logger.info(f"{check}: {'PASSED' if result else 'FAILED'}")
            
            logger.info(f"Overall validation status: {'PASSED' if is_valid else 'FAILED'}")
            
            with open(self.config.status_file, 'w') as f:
                json.dump({"validation_status": is_valid}, f)
                
            return is_valid


if __name__ == "__main__":
    from FraudGuard.config.config import ConfigurationManager
    
    logger.info(">>>>>> Stage: Validation started <<<<<<")
    config = ConfigurationManager()
    validation = Validation(config=config.get_data_validation_config())
    validation.validation()
    logger.info(">>>>>> Stage: Validation completed <<<<<<")