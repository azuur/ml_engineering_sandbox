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

    plots, axs = plt.subplots(1, 2, figsize=(10, 5))
    make_roc_plot(model, data, logger)(ax=axs[0])
    make_calibration_plot(data, y_score, logger)(ax=axs[1])
    logger.info("Finished evaluation pipeline.")
    return (metrics, plots)
