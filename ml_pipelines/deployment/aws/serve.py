import logging
import os
import sys
from functools import partial
from logging import Logger
from typing import Union

import typer

from ml_pipelines.deployment.aws.io import (
    get_best_version,
    get_latest_version,
    get_train_artifacts,
    prediction_logging_func,
)
from ml_pipelines.deployment.common.serve import run_serve


def main(
    train_artifacts_bucket: Union[str, None] = None,  # noqa: UP007
    train_version: Union[str, None] = None,  # noqa: UP007
):
    """
    Serves a model in the /predict/ endpoint of a FastAPI app.

    If `train_artifacts_bucket` is null, the command searches for the
    TRAIN_ARTIFACTS_BUCKET environment variable.

    If `train_version` is null, the command loads the model tagged as 'best_version'
    in the `train_artifacts_root_path`, and if not found, loads the latest model.
    """
    logger = Logger(__file__)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if train_artifacts_bucket is None:
        train_artifacts_bucket = os.environ["TRAIN_ARTIFACTS_BUCKET"]

    if train_version is None:
        train_version = get_best_version(train_artifacts_bucket)  # type: ignore

    if train_version is None:
        train_version = get_latest_version(
            train_artifacts_bucket,  # type: ignore
            "model.pickle",
        )

    get_train_artifacts_func = partial(
        get_train_artifacts,
        train_artifacts_bucket,  # type: ignore
        load_data=False,
    )

    uvicorn_kwargs: dict = {}
    run_serve(  # noqa: PLR0913
        train_version=train_version,
        get_train_artifacts_func=get_train_artifacts_func,
        prediction_logging_func=prediction_logging_func,
        logger=logger,
        uvicorn_kwargs=uvicorn_kwargs,
    )


if __name__ == "__main__":
    typer.run(main)
