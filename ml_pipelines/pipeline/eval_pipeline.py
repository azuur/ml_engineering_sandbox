import json
import logging
import pickle
import sys
from logging import Logger

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_pipelines.logic.common.feature_eng import (
    FeatureEngineeringParams,
    transform_features,
)
from ml_pipelines.logic.common.model import predict
from ml_pipelines.logic.eval.eval import (
    calculate_metrics,
    make_calibration_plot,
    make_roc_plot,
)

# Input


def eval_pipeline(
    model: LogisticRegression,
    feature_eng_params: FeatureEngineeringParams,
    data: pd.DataFrame,
    logger: Logger,
):
    logger.info("Starting evaluation pipeline.")
    data = transform_features(data, feature_eng_params, logger)
    y_score = predict(model, data[["X1", "X2"]], logger)
    metrics = calculate_metrics(data["Y"], y_score, logger)

    # Output
    plots, axs = plt.subplots(1, 2, figsize=(10, 5))
    make_roc_plot(model, data, logger)(ax=axs[0])
    make_calibration_plot(data, y_score, logger)(ax=axs[1])
    logger.info("Finished evaluation pipeline.")
    return (metrics, plots)


with open("model.pickle", "rb") as f:  # type: ignore
    model: LogisticRegression = pickle.load(f)

with open("feature_eng_params.json") as f:  # type: ignore
    feature_eng_params = FeatureEngineeringParams(**json.loads(f.read()))

test_data = pd.read_csv("raw_test_data.csv")
logger = Logger(__file__)
logger.addHandler(logging.StreamHandler(sys.stdout))
(metrics, plots) = eval_pipeline(model, feature_eng_params, test_data, logger)

with open("metrics.txt", "w+") as f:  # type: ignore
    f.write(str(metrics))  # type: ignore
plots.savefig("calibration_plot.png")


# fig, ax = plt.subplots()
# for i in [0, 1]:
#     tmp = test_data.loc[test_data.Y == i, :]
#     ax.scatter(
#         tmp["X1"],
#         tmp["X2"],
#         c=tmp["Y"].map({0: "lightgray", 1: "red"}),
#         label=f"Y={i}",
#         s=2,
#         alpha=0.7,
#     )
# ax.set_xlabel("X1")
# ax.set_ylabel("X2")
# ax.set_title("Scatter plot of training data")
# ax.legend(framealpha=1)
# plt.show()
# plt.show()
# ax.legend(framealpha=1)
# plt.show()
# plt.show()
