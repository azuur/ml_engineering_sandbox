from collections.abc import Callable
from logging import Logger

import pandas as pd

from ml_pipelines.pipeline.eval_pipeline import eval_pipeline
from ml_pipelines.pipeline.train_pipeline import TrainArtifacts


def run_eval_comparison_pipeline(  # noqa: PLR0913
    raw_data_version: str,
    train_versions: list[str],
    get_raw_data_func: Callable[[str], pd.DataFrame],
    get_train_artifacts_func: Callable[[str], TrainArtifacts],
    tag_best_model_func: Callable[[str], None],
    tag_best_model: bool,
    logger: Logger,
):
    logger.info(f"Running eval pipeline on model versions: {train_versions}.")
    logger.info(f"Raw data version {raw_data_version}.")
    raw_data = get_raw_data_func(raw_data_version)
    all_metrics = []
    for v in train_versions:
        train_artifacts = get_train_artifacts_func(v)
        metrics, _ = eval_pipeline(
            train_artifacts["model"],
            train_artifacts["feature_eng_params"],
            raw_data,
            logger,
        )
        all_metrics.append((v, metrics))
    best_version = max(all_metrics, key=lambda t: t[1])[0]
    if tag_best_model and len(train_versions) > 1:
        logger.info(f"Tagging best version as {best_version}")
        tag_best_model_func(best_version)
