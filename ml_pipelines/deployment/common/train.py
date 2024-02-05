from collections.abc import Callable
from logging import Logger

import pandas as pd
from matplotlib.figure import Figure

from ml_pipelines.pipeline.eval_pipeline import eval_pipeline
from ml_pipelines.pipeline.train_pipeline import TrainArtifacts, train_pipeline


def run_train_pipeline(  # noqa: PLR0913
    raw_data_version: str,
    train_version: str,
    logger: Logger,
    get_raw_data_func: Callable[[str], pd.DataFrame],
    save_train_artifacts_func: Callable[[str, TrainArtifacts], None],
    save_eval_artifacts_func: Callable[[str, float, Figure], None],
    split_random_state: int = 3825,
):
    logger.info(f"Running full training pipeline version {train_version}.")
    logger.info(f"Raw data version {raw_data_version}.")
    raw_data = get_raw_data_func(raw_data_version)
    train_artifacts = train_pipeline(
        raw_data, split_random_state=split_random_state, logger=logger
    )
    save_train_artifacts_func(train_version, train_artifacts)
    logger.info("Saved train artifacts.")
    metrics, plots = eval_pipeline(
        train_artifacts["model"],
        train_artifacts["feature_eng_params"],
        train_artifacts["raw_test_data"],  # type: ignore
        logger,
    )
    save_eval_artifacts_func(train_version, metrics, plots)
    logger.info("Saved eval artifacts.")
