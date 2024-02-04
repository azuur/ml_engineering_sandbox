import logging
import os
import sys
from functools import partial
from logging import Logger
from typing import Union

import typer

from ml_pipelines.deployment.common.train import run_train_pipeline
from ml_pipelines.deployment.local.io import (
    get_latest_version,
    get_raw_data,
    make_version,
    save_eval_artifacts,
    save_train_artifacts,
)


def main(
    raw_data_root_path: Union[str, None] = None,  # noqa: UP007
    train_artifacts_root_path: Union[str, None] = None,  # noqa: UP007, E501
    raw_data_version: Union[str, None] = None,  # noqa: UP007
    train_version: Union[str, None] = None,  # noqa: UP007
    split_random_state: int = 3825,
):
    """
    Runs the feature engineering and training pipeline using local paths
    for inputs and outputs.

    If `raw_data_root_path` is null, the command searches for the RAW_DATA_ROOT_PATH
    environment variable, and if not present, assumes this to be "/".

    If `train_artifacts_root_path` is null, the command searches for the
    TRAIN_ARTIFACTS_ROOT_PATH environment variable, and if not present,
    assumes this to be "/".

    If `raw_data_version` is null, the command searches for the latest version in
    `raw_data_root_path`.

    If `train_version` is null, the command automatically generates a new model
    version to save train artifacts.
    """
    logger = Logger(__file__)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if raw_data_root_path is None:
        raw_data_root_path = os.environ.get("RAW_DATA_ROOT_PATH", "/")
    if train_artifacts_root_path is None:
        train_artifacts_root_path = os.environ.get("TRAIN_ARTIFACTS_ROOT_PATH", "/")

    if raw_data_version is None:
        raw_data_version = get_latest_version(
            raw_data_root_path,  # type: ignore
            "raw_data.csv",
        )

    if train_version is None:
        train_version = make_version(prefix="model")

    get_raw_data_func = partial(
        get_raw_data,
        root_path=raw_data_root_path,  # type: ignore
    )
    save_train_artifacts_func = partial(
        save_train_artifacts,
        root_path=train_artifacts_root_path,  # type: ignore
    )
    save_eval_artifacts_func = partial(
        save_eval_artifacts,
        root_path=train_artifacts_root_path,  # type: ignore
    )

    run_train_pipeline(  # noqa: PLR0913
        raw_data_version=raw_data_version,
        train_version=train_version,
        logger=logger,
        get_raw_data_func=get_raw_data_func,
        save_train_artifacts_func=save_train_artifacts_func,
        save_eval_artifacts_func=save_eval_artifacts_func,
        split_random_state=split_random_state,
    )


if __name__ == "__main__":
    typer.run(main)
