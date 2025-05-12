from project.constants import *
from project.utils.common import *
from project.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_PATH,
        params_filepath = PARAMS_PATH,
        schema_filepath = SCHEMA_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_id=config.source_id,
            zip_file=config.zip_file,
            unzip_file=config.unzip_file 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.columns
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            unzip_file=config.unzip_file,
            all_schema=schema,
        )
        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        schema = self.schema
        params = self.params.train_test_split

        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            target_column=schema.target_column.name,
            preprocessor_path=config.preprocessor_path,
            label_encoder=config.label_encoder,
            categorical_columns=schema.categorical_columns,
            numeric_columns=schema.numeric_columns,
            columns_to_drop=schema.data_cleaning.columns_to_drop,
            test_size=params.test_size,
            random_state=params.random_state
        )
        
        return data_transformation_config
    


    def get_model_training_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        schema = self.schema
        cv_params = self.params.cross_validation

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_preprocess=config.train_preprocess,
            test_preprocess=config.test_preprocess,
            model_name=config.model_name,
            target_column=schema.target_column.name,
            cv_folds=cv_params.cv_folds,            
            scoring=cv_params.scoring,             
            n_jobs=cv_params.n_jobs,
            n_iter=cv_params.n_iter          
        )
        
        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        schema = self.schema.target_column

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_path=config.test_path,
            model_path=config.model_path,
            preprocess_path=config.preprocess_path,
            metrics_path=config.metrics_path,
            target_column=schema.name,
            cm_path=config.cm_path,
            roc_path=config.roc_path,
        )

        return model_evaluation_config