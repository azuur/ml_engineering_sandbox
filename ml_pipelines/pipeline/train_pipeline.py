import pandas as pd

from ml_pipelines.logic.common.feature_eng import (
    fit_feature_transform,
    transform_features,
)
from ml_pipelines.logic.train.train import save_model, split_data, train_model

# Input
data = pd.read_csv("data.csv")

raw_train_data, raw_test_data = split_data(data, random_state=3397)
feature_eng_params = fit_feature_transform(raw_train_data)
train_data = transform_features(raw_train_data, feature_eng_params)
test_data = transform_features(raw_test_data, feature_eng_params)
model = train_model(train_data=train_data)

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
