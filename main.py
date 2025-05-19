import os
import argparse
from project import logger
from project.pipeline.stage_01 import DataIngestionPipeline
from project.pipeline.stage_02 import DataValidationPipeline
from project.pipeline.stage_03 import DataTransformationPipeline
from project.pipeline.stage_04 import ModelTrainingPipeline
from project.pipeline.stage_05 import ModelEvaluationPipeline


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

        else:
            raise ValueError(f"Unknown stage: {stage_name}")

        logger.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Missing key in configuration or data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

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
        ]
        for stage in stages:
            run_stage(stage)
