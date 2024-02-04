import logging
import os
import sys
from logging import Logger
from typing import Union

import typer

from ml_pipelines.deployment.local.io import (
    get_all_available_train_versions,
    get_latest_version,
    get_raw_data,
    get_train_artifacts,
    tag_best_version,
)
from ml_pipelines.pipeline.eval_pipeline import eval_pipeline


def run_eval_comparison_pipeline(  # noqa: PLR0913
    raw_data_version: str,
    raw_data_root_path: os.PathLike,
    train_versions: list[str],
    train_artifacts_root_path: os.PathLike,
    tag_best_model: bool,
    logger: Logger,
):
    logger.info(f"Running eval pipeline on model versions: {train_versions}.")
    logger.info(f"Raw data version {raw_data_version}.")
    raw_data = get_raw_data(raw_data_version, raw_data_root_path)
    all_metrics = []
    for v in train_versions:
        train_artifacts = get_train_artifacts(
            v, train_artifacts_root_path, load_data=False
        )
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
        tag_best_version(best_version, train_artifacts_root_path)


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

    run_eval_comparison_pipeline(  # noqa: PLR0913
        raw_data_version=raw_data_version,
        raw_data_root_path=raw_data_root_path,  # type: ignore
        train_versions=train_versions,  # type: ignore
        train_artifacts_root_path=train_artifacts_root_path,  # type: ignore
        tag_best_model=tag_best_model,
        logger=logger,
    )


if __name__ == "__main__":
    typer.run(main)
