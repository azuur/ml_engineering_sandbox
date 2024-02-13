from logging import Logger
from typing import NotRequired, TypedDict

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_pipelines.core.common.feature_eng import (
    FeatureEngineeringParams,
    fit_feature_transform,
    transform_features,
)
from ml_pipelines.core.train.train import split_data, train_model


class TrainArtifacts(TypedDict):
    model: LogisticRegression
    feature_eng_params: FeatureEngineeringParams
    raw_train_data: NotRequired[pd.DataFrame]
    raw_test_data: NotRequired[pd.DataFrame]
    train_data: NotRequired[pd.DataFrame]
    test_data: NotRequired[pd.DataFrame]


def train_pipeline(data: pd.DataFrame, split_random_state: int, logger: Logger):
    logger.info("Starting train pipeline.")
    raw_train_data, raw_test_data = split_data(data, split_random_state, logger)
    feature_eng_params = fit_feature_transform(raw_train_data, logger)
    train_data = transform_features(raw_train_data, feature_eng_params, logger)
    test_data = transform_features(raw_test_data, feature_eng_params, logger)
    model = train_model(train_data, logger)
    logger.info("Finished train pipeline.")
    return TrainArtifacts(
        model=model,
        feature_eng_params=feature_eng_params,
        raw_train_data=raw_train_data,
        raw_test_data=raw_test_data,
        train_data=train_data,
        test_data=test_data,
    )
