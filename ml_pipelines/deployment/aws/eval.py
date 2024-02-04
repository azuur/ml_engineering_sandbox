import logging
import os
import sys
from functools import partial
from logging import Logger
from typing import Union

import typer

from ml_pipelines.deployment.common.eval import run_eval_comparison_pipeline
from ml_pipelines.deployment.local.io import (
    get_all_available_train_versions,
    get_latest_version,
    get_raw_data,
    get_train_artifacts,
    tag_best_version,
)


def main(
    raw_data_root_path: Union[str, None] = None,  # noqa: UP007
    train_artifacts_root_path: Union[str, None] = None,  # noqa: UP007, E501
    raw_data_version: Union[str, None] = None,  # noqa: UP007
    train_versions: Union[list[str], None] = None,  # noqa: UP007
    tag_best_model: bool = False,
):
    """
    Runs the model evaluation and comparison pipeline using local paths
    for inputs and outputs.

    If `raw_data_root_path` is null, the command searches for the RAW_DATA_ROOT_PATH
    environment variable, and if not present, assumes this to be "/".

    If `train_artifacts_root_path` is null, the command searches for the
    TRAIN_ARTIFACTS_ROOT_PATH environment variable, and if not present,
    assumes this to be "/".

    If `raw_data_version` is null, the command searches for the latest version in
    `raw_data_root_path`.

    If `train_versions` is null or empty, the command automatically evaluates all
    models found in `train_artifacts_root_path`.

    If `tag_best_model` is set (to true) and more than one model version is evaluated,
    the best performing one is tagged as the best version.
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

    if not train_versions:
        train_versions = get_all_available_train_versions(  # type: ignore
            train_artifacts_root_path  # type: ignore
        )

    get_raw_data_func = partial(
        get_raw_data,
        root_path=raw_data_root_path,  # type: ignore
    )
    get_train_artifacts_func = partial(
        get_train_artifacts,
        root_path=train_artifacts_root_path,  # type: ignore
        load_data=False,
    )
    tag_best_version_func = partial(
        tag_best_version,
        train_artifacts_root_path=train_artifacts_root_path,  # type: ignore
    )

    run_eval_comparison_pipeline(  # noqa: PLR0913
        raw_data_version=raw_data_version,
        train_versions=train_versions,  # type: ignore
        get_raw_data_func=get_raw_data_func,
        get_train_artifacts_func=get_train_artifacts_func,
        tag_best_model_func=tag_best_version_func,
        tag_best_model=tag_best_model,
        logger=logger,
    )


if __name__ == "__main__":
    typer.run(main)
