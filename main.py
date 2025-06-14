import os
import sys
import argparse
from FraudGuard import logger
from FraudGuard.utils.exceptions import CustomException
from FraudGuard.pipeline.stage_01 import DataIngestionPipeline
from FraudGuard.pipeline.stage_02 import DataValidationPipeline
from FraudGuard.pipeline.stage_03 import DataTransformationPipeline
from FraudGuard.pipeline.stage_04 import ModelTrainingPipeline
from FraudGuard.pipeline.stage_05 import ModelEvaluationPipeline
from FraudGuard.pipeline.stage_06 import ModelRegistryPipeline


def run_stage(stage_name):
    logger.info(f">>>>>> Stage {stage_name} started <<<<<<")

    try:
        if stage_name == "data_ingestion":
            stage = DataIngestionPipeline()
            stage.run()

        elif stage_name == "data_validation":
            stage = DataValidationPipeline()
            stage.run()


        elif stage_name == "data_transformation":
            stage = DataTransformationPipeline()
            stage.run()

        elif stage_name == "model_training":
            stage = ModelTrainingPipeline()
            stage.run()

        elif stage_name == "model_evaluation":
            stage = ModelEvaluationPipeline()
            stage.run()
        
        elif stage_name == "model_registry":
            stage = ModelRegistryPipeline()
            stage.run()


        else:
            raise ValueError(f"Unknown stage: {stage_name}")

        logger.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")

    except Exception as e:
        raise CustomException(str(e), sys)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific pipeline stage.")
    parser.add_argument("--stage", help="Name of the stage to run")
    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage)
    else:
        stages = [
            "data_ingestion",
            "data_validation",
            "data_transformation",
            "model_training",
            "model_evaluation",
            "model_registry"
        ]
        for stage in stages:
            run_stage(stage)
