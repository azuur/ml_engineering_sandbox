from logging import Logger

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from ml_pipelines.logic.common.feature_eng import (
    FeatureEngineeringParams,
    transform_features,
)
from ml_pipelines.logic.common.model import predict


class Point(BaseModel):
    prediction_id: str
    x1: float = Field(..., description="Value of x1, must be smaller than 5", lt=5)
    x2: float


def create_fastapi_app(
    model: LogisticRegression,
    feature_eng_params: FeatureEngineeringParams,
    logger: Logger,
):
    app = FastAPI()

    @app.get("/predict/")
    async def _(inputs: Point | list[Point]):
        """
        Endpoint to predict probabilities for the given list of points.
        :param points: List of points (x1, x2) (or single point)
        :return: List of probability predictions (or single prediction)
        """
        try:
            single = False
            if isinstance(inputs, Point):
                single = True
                inputs = [inputs]
            # Convert the input points to a numpy array
            x_matrix = pd.DataFrame([{"X1": p.x1, "X2": p.x2} for p in inputs])
            x_matrix = transform_features(x_matrix, feature_eng_params, logger)

            # Make predictions using the pre-trained model
            predictions = predict(model, x_matrix, logger).tolist()

            if single:
                predictions = predictions[0]

            return {"predictions": predictions}

        except Exception:
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
    return app
