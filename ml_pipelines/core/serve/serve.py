from collections.abc import Callable
from logging import Logger

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from ml_pipelines.core.common.feature_eng import (
    FeatureEngineeringParams,
    transform_features,
)
from ml_pipelines.core.common.model import predict


class Point(BaseModel):
    prediction_id: str
    x1: float = Field(..., description="Value of x1, must be smaller than 5", lt=5)
    x2: float


PredictionLoggingFunc = Callable[
    [list[tuple[Point, float]]], tuple[bool, Exception | None]
]


def create_fastapi_app(
    model: LogisticRegression,
    feature_eng_params: FeatureEngineeringParams,
    logger: Logger,
    prediction_logging_func: PredictionLoggingFunc,
):
    app = FastAPI()

    @app.post("/predict/")
    async def _(inputs: Point | list[Point]):
        """
        Endpoint to predict probabilities for the given list of points.
        :param points: List of points (x1, x2) (or single point)
        :return: List of probability predictions (or single prediction)
        """
        try:
            logger.info("Responding to request for predictions. (GET /predict/)")
            single = False
            if isinstance(inputs, Point):
                single = True
                inputs = [inputs]
            logger.info(f"Number of samples for inference: {len(inputs)}")

            x_matrix = pd.DataFrame([{"X1": p.x1, "X2": p.x2} for p in inputs])
            x_matrix = transform_features(x_matrix, feature_eng_params, logger)
            predictions: list[float] = predict(model, x_matrix, logger).tolist()
            logger.info("Computed inference.")

            predictions_to_log = list(zip(inputs, predictions))
            log_success, log_exception = prediction_logging_func(predictions_to_log)
            if log_success and log_exception is None:
                logger.info("Logged predictions for samples.")
            else:
                logger.warning(
                    "Unable to log predictions for samples.", exc_info=log_exception
                )

            output = predictions[0] if single else predictions
            logger.info("Returning predictions.")
            return {"predictions": output}

        except Exception as e:
            logger.error("Inference failed.", exc_info=e)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
