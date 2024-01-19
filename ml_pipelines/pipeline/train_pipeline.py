import logging
import pickle
import sys
from logging import Logger

import pandas as pd

from ml_pipelines.logic.common.feature_eng import (
    fit_feature_transform,
    transform_features,
)
from ml_pipelines.logic.train.train import split_data, train_model

# Input


def train_pipeline(data: pd.DataFrame, split_random_state: int, logger: Logger):
    logger.info("Starting train pipeline.")
    raw_train_data, raw_test_data = split_data(data, split_random_state, logger)
    feature_eng_params = fit_feature_transform(raw_train_data, logger)
    train_data = transform_features(raw_train_data, feature_eng_params, logger)
    test_data = transform_features(raw_test_data, feature_eng_params, logger)
    model = train_model(train_data, logger)
    logger.info("Finished train pipeline.")
    return (
        model,
        feature_eng_params,
        raw_train_data,
        raw_test_data,
        train_data,
        test_data,
    )


logger = Logger(__file__)
logger.addHandler(logging.StreamHandler(sys.stdout))
data = pd.read_csv("data.csv")
(
    model,
    feature_eng_params,
    raw_train_data,
    raw_test_data,
    train_data,
    test_data,
) = train_pipeline(data, split_random_state=3825, logger=logger)


# Outputs
with open("model.pickle", "wb") as f:
    pickle.dump(model, f)
raw_train_data.to_csv("raw_train_data.csv", index=False)
raw_test_data.to_csv("raw_test_data.csv", index=False)
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)
with open("feature_eng_params.json", "w") as f:  # type: ignore
    f.write(feature_eng_params.model_dump_json())
test_data.to_csv("test_data.csv", index=False)
with open("feature_eng_params.json", "w") as f:  # type: ignore
    f.write(feature_eng_params.model_dump_json())
