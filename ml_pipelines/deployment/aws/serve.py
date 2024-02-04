import logging
import os
import sys
from functools import partial
from logging import Logger
from typing import Union

import typer

from ml_pipelines.deployment.common.serve import run_serve
from ml_pipelines.deployment.local.io import (
    get_best_version,
    get_latest_version,
    get_train_artifacts,
    prediction_logging_func,
)


def main(
    train_artifacts_root_path: Union[str, None] = None,  # noqa: UP007
    train_version: Union[str, None] = None,  # noqa: UP007
):
    """
    Serves a model in the /predict/ endpoint of a FastAPI app.

    If `train_artifacts_root_path` is null, the command searches for the
    TRAIN_ARTIFACTS_ROOT_PATH environment variable, and if not present,
    assumes this to be "/".

    If `train_version` is null, the command loads the model tagged as 'best_version'
    in the `train_artifacts_root_path`, and if not found, loads the latest model.
    """
    logger = Logger(__file__)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if train_artifacts_root_path is None:
        train_artifacts_root_path = os.environ.get("TRAIN_ARTIFACTS_ROOT_PATH", "/")

    if train_version is None:
        train_version = get_best_version(train_artifacts_root_path)  # type: ignore

    if train_version is None:
        train_version = get_latest_version(
            train_artifacts_root_path,  # type: ignore
            "model.pickle",
        )

    get_train_artifacts_func = partial(
        get_train_artifacts,
        root_path=train_artifacts_root_path,  # type: ignore
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
