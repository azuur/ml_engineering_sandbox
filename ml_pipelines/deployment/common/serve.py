from collections.abc import Callable
from logging import Logger

import uvicorn

from ml_pipelines.logic.serve.serve import PredictionLoggingFunc, create_fastapi_app
from ml_pipelines.pipeline.train_pipeline import TrainArtifacts


def run_serve(
    train_version: str,
    get_train_artifacts_func: Callable[[str], TrainArtifacts],
    prediction_logging_func: PredictionLoggingFunc,
    logger: Logger,
    uvicorn_kwargs: dict,
):
    logger.info(f"Serving model {train_version}.")
    train_artifacts = get_train_artifacts_func(train_version)
    model = train_artifacts["model"]
    feature_eng_params = train_artifacts["feature_eng_params"]
    app = create_fastapi_app(model, feature_eng_params, logger, prediction_logging_func)
    logger.info("Loaded model, set up endpoint.")

    uvicorn.run(app=app, **uvicorn_kwargs)  # type: ignore
