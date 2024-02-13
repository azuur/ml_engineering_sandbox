from logging import Logger

import numpy as np
import pandas as pd
from pydantic import BaseModel


class FeatureEngineeringParams(BaseModel):
    x1_exp_mean: float
    x2_log_mean: float


def transform_features(
    data: pd.DataFrame, feature_eng_params: FeatureEngineeringParams, logger: Logger
):
    logger.info(f"Transforming features for {len(data)} samples.")
    data = data.copy()
    data["X1"] = np.exp(data["X1"]) - feature_eng_params.x1_exp_mean
    data["X2"] = np.log(data["X2"]) - feature_eng_params.x2_log_mean
    return data


def fit_feature_transform(data: pd.DataFrame, logger: Logger):
    logger.info(f"Fitting feature transforms on {len(data)} samples.")
    x1_exp_mean = np.exp(data["X1"]).mean()
    x2_log_mean = np.log(data["X2"]).mean()
    return FeatureEngineeringParams(x1_exp_mean=x1_exp_mean, x2_log_mean=x2_log_mean)
