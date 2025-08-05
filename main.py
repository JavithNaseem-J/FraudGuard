import os
import sys
import argparse
from FraudGuard.utils.logging import logger
from FraudGuard.utils.exceptions import CustomException
from FraudGuard.pipeline.feature_pipeline import FeaturePipeline
from FraudGuard.pipeline.model_pipeline import ModelPipeline


def run_stage(stage_name):
    logger.info(f">>>>>> Stage {stage_name} started <<<<<<")

    try:
        if stage_name == "feature_pipeline":
            stage = FeaturePipeline()
            stage.run()

        elif stage_name == "model_pipeline":
            stage = ModelPipeline()
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
            "feature_pipeline",
            "model_pipeline", 
        ]
        for stage in stages:
            run_stage(stage)
