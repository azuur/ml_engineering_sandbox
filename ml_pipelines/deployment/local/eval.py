import logging
import os
import sys
from logging import Logger
from typing import Union

import typer

from ml_pipelines.deployment.local.common import (
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
    logger.info(f"Tagging best version as {best_version}")
    tag_best_version(best_version, train_artifacts_root_path)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    RAW_DATA_ROOT_DIR = os.environ["RAW_DATA_ROOT_DIR"]
    TRAIN_ARTIFACTS_ROOT_DIR = os.environ["TRAIN_ARTIFACTS_ROOT_DIR"]

    def main(
        raw_data_version: Union[str, None] = None,  # noqa: UP007
        train_versions: Union[list[str], None] = None,  # noqa: UP007
        raw_data_root_path: str = RAW_DATA_ROOT_DIR,
        train_artifacts_root_path: str = TRAIN_ARTIFACTS_ROOT_DIR,
    ):
        logger = Logger(__file__)
        logger.addHandler(logging.StreamHandler(sys.stdout))

        if raw_data_version is None:
            raw_data_version = get_latest_version(
                raw_data_root_path,  # type: ignore
                "raw_data.csv",
            )

        if not train_versions:
            train_versions = get_all_available_train_versions(  # type: ignore
                train_artifacts_root_path
            )

        run_eval_comparison_pipeline(  # noqa: PLR0913
            raw_data_version=raw_data_version,
            raw_data_root_path=raw_data_root_path,  # type: ignore
            train_versions=train_versions,  # type: ignore
            train_artifacts_root_path=train_artifacts_root_path,  # type: ignore
            logger=logger,
        )

    typer.run(main)
