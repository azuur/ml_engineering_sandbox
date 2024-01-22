import logging
import os
import sys
from logging import Logger
from typing import Union

import typer
import uvicorn

from ml_pipelines.deployment.local.common import (
    get_best_version,
    get_train_artifacts,
)
from ml_pipelines.logic.serve.serve import Point, create_fastapi_app


def prediction_logging_func(predictions: list[tuple[Point, float]]):
    print("yay")  # noqa: T201
    return True, None


def run_serve(
    train_version: str,
    train_artifacts_root_path: os.PathLike,  # type: ignore
    logger: Logger,
    uvicorn_kwargs: dict,
):
    logger.info(f"Serving model {train_version}.")
    train_artifacts = get_train_artifacts(
        train_version, train_artifacts_root_path, load_data=False
    )
    model = train_artifacts["model"]
    feature_eng_params = train_artifacts["feature_eng_params"]
    app = create_fastapi_app(model, feature_eng_params, logger, prediction_logging_func)
    logger.info("Loaded model, set up endpoint.")

    uvicorn.run(app=app, **uvicorn_kwargs)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    TRAIN_ARTIFACTS_ROOT_DIR = os.environ["TRAIN_ARTIFACTS_ROOT_DIR"]

    def main(
        train_version: Union[str, None] = None,  # noqa: UP007
        train_artifacts_root_path: str = TRAIN_ARTIFACTS_ROOT_DIR,
    ):
        logger = Logger(__file__)
        logger.addHandler(logging.StreamHandler(sys.stdout))

        if train_version is None:
            train_version = get_best_version(train_artifacts_root_path)  # type: ignore
            # train_version = get_latest_version(
            #     train_artifacts_root_path,  # type: ignore
            #     "model.pickle",
            # )

        uvicorn_kwargs: dict = {}
        run_serve(  # noqa: PLR0913
            train_version=train_version,
            train_artifacts_root_path=train_artifacts_root_path,  # type: ignore
            logger=logger,
            uvicorn_kwargs=uvicorn_kwargs,
        )

    typer.run(main)
