import pandas as pd

from ml_pipelines.logic.common.feature_eng import (
    fit_feature_transform,
    transform_features,
)
from ml_pipelines.logic.train.train import save_model, split_data, train_model

# Input


def train_pipeline(data: pd.DataFrame, split_random_state: int):
    raw_train_data, raw_test_data = split_data(data, random_state=split_random_state)
    feature_eng_params = fit_feature_transform(raw_train_data)
    train_data = transform_features(raw_train_data, feature_eng_params)
    test_data = transform_features(raw_test_data, feature_eng_params)
    model = train_model(train_data=train_data)
    return (
        model,
        feature_eng_params,
        raw_train_data,
        raw_test_data,
        train_data,
        test_data,
    )


data = pd.read_csv("data.csv")
(
    model,
    feature_eng_params,
    raw_train_data,
    raw_test_data,
    train_data,
    test_data,
) = train_pipeline(data, split_random_state=3825)


# Outputs
save_model(model)
raw_train_data.to_csv("raw_train_data.csv", index=False)
raw_test_data.to_csv("raw_test_data.csv", index=False)
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)
with open("feature_eng_params.json", "w") as f:
    f.write(feature_eng_params.model_dump_json())
test_data.to_csv("test_data.csv", index=False)
with open("feature_eng_params.json", "w") as f:
    f.write(feature_eng_params.model_dump_json())
