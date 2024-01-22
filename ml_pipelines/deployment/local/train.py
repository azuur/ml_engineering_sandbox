import logging
import os
import sys
from logging import Logger
from typing import Union

import typer

from ml_pipelines.deployment.local.common import (
    get_latest_version,
    get_raw_data,
    make_version,
    save_eval_artifacts,
    save_train_artifacts,
)
from ml_pipelines.pipeline.eval_pipeline import eval_pipeline
from ml_pipelines.pipeline.train_pipeline import train_pipeline


def run_train_pipeline(  # noqa: PLR0913
    raw_data_version: str,
    raw_data_root_path: os.PathLike,
    train_version: str,
    train_artifacts_root_path: os.PathLike,
    logger: Logger,
    split_random_state: int = 3825,
):
    logger.info(f"Running full training pipeline version {train_version}.")
    logger.info(f"Raw data version {raw_data_version}.")
    raw_data = get_raw_data(raw_data_version, raw_data_root_path)
    train_artifacts = train_pipeline(
        raw_data, split_random_state=split_random_state, logger=logger
    )
    save_train_artifacts(train_version, train_artifacts_root_path, train_artifacts)
    logger.info("Saved train artifacts.")
    metrics, plots = eval_pipeline(
        train_artifacts["model"],
        train_artifacts["feature_eng_params"],
        train_artifacts["raw_test_data"],  # type: ignore
        logger,
    )
    save_eval_artifacts(train_version, train_artifacts_root_path, metrics, plots)
    logger.info("Saved eval artifacts.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    RAW_DATA_ROOT_DIR = os.environ["RAW_DATA_ROOT_DIR"]
    TRAIN_ARTIFACTS_ROOT_DIR = os.environ["TRAIN_ARTIFACTS_ROOT_DIR"]

    def main(
        raw_data_version: Union[str, None] = None,  # noqa: UP007
        train_version: Union[str, None] = None,  # noqa: UP007
        raw_data_root_path: str = RAW_DATA_ROOT_DIR,
        train_artifacts_root_path: str = TRAIN_ARTIFACTS_ROOT_DIR,
        split_random_state: int = 3825,
    ):
        logger = Logger(__file__)
        logger.addHandler(logging.StreamHandler(sys.stdout))

        if raw_data_version is None:
            raw_data_version = get_latest_version(
                raw_data_root_path,  # type: ignore
                "raw_data.csv",
            )

        if train_version is None:
            train_version = make_version(prefix="model")

        run_train_pipeline(  # noqa: PLR0913
            raw_data_version=raw_data_version,
            raw_data_root_path=raw_data_root_path,  # type: ignore
            train_version=train_version,
            train_artifacts_root_path=train_artifacts_root_path,  # type: ignore
            logger=logger,
            split_random_state=split_random_state,
        )

    typer.run(main)
