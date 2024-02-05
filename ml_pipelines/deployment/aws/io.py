import json
import os
import pickle
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression

from ml_pipelines.logic.common.feature_eng import FeatureEngineeringParams
from ml_pipelines.logic.serve.serve import Point
from ml_pipelines.pipeline.train_pipeline import TrainArtifacts


def make_version(prefix: str) -> str:
    version = str(uuid4())
    return f"{prefix}_{version}"


def get_raw_data(bucket_name: str, version: str) -> pd.DataFrame:
    s3 = boto3.client("s3")
    s3_key = f"{version}/raw_data.csv"

    response = s3.get_object(Bucket=bucket_name, Key=s3_key)
    content = response["Body"].read().decode("utf-8")

    return pd.read_csv(StringIO(content))


class ModelVersionAlreadyExists(Exception):
    pass


def save_train_artifacts(
    bucket_name: str,
    version: str,
    artifacts: TrainArtifacts,
):
    s3 = boto3.client("s3")

    # Create a directory structure within the bucket for the version
    s3_prefix = f"{version}/"

    # Check if the version already exists in the bucket
    head_object = None
    try:
        head_object = s3.head_object(Bucket=bucket_name, Key=s3_prefix)
    except ClientError as ex:
        if ex.response["Error"]["Code"] == "404":
            pass
        else:
            raise
    if head_object is not None:
        raise ModelVersionAlreadyExists()

    model_pickle_key = s3_prefix + "model.pickle"
    with NamedTemporaryFile() as f:
        pickle.dump(artifacts["model"], f)
        s3.upload_file(f.name, bucket_name, model_pickle_key)

    feature_eng_params_key = s3_prefix + "feature_eng_params.json"
    with NamedTemporaryFile(mode="w") as f:
        f.write(artifacts["feature_eng_params"].json())
        s3.upload_file(f.name, bucket_name, feature_eng_params_key)

    data_keys = ["raw_train_data", "raw_test_data", "train_data", "test_data"]
    for key in data_keys:
        if key in artifacts:
            dataset: pd.DataFrame = artifacts[key]  # type: ignore
            csv_key = s3_prefix + f"{key}.csv"
            with NamedTemporaryFile(mode="w") as f:
                dataset.to_csv(f.name, index=False)
                s3.upload_file(f.name, bucket_name, csv_key)


def get_train_artifacts(bucket_name: str, version: str, load_data: bool = True):
    s3 = boto3.client("s3")

    s3_prefix = f"{version}/"

    model_key = s3_prefix + "model.pickle"
    s3.download_file(bucket_name, model_key, "/tmp/model.pickle")
    with open("/tmp/model.pickle", "rb") as f:
        model: LogisticRegression = pickle.load(f)

    feature_eng_params_key = s3_prefix + "feature_eng_params.json"
    s3.download_file(
        bucket_name, feature_eng_params_key, "/tmp/feature_eng_params.json"
    )
    with open("/tmp/feature_eng_params.json") as f:
        feature_eng_params = FeatureEngineeringParams(**json.loads(f.read()))

    if not load_data:
        return TrainArtifacts(
            model=model,
            feature_eng_params=feature_eng_params,
        )

    data_keys = ["raw_train_data", "raw_test_data", "train_data", "test_data"]
    data_dict = {}
    for key in data_keys:
        csv_key = s3_prefix + f"{key}.csv"
        data_dict[key] = pd.read_csv(
            StringIO(
                s3.get_object(Bucket=bucket_name, Key=csv_key)["Body"]
                .read()
                .decode("utf-8")
            )
        )

    return TrainArtifacts(
        model=model,
        feature_eng_params=feature_eng_params,
        raw_train_data=data_dict["raw_train_data"],
        raw_test_data=data_dict["raw_test_data"],
        train_data=data_dict["train_data"],
        test_data=data_dict["test_data"],
    )


def save_eval_artifacts(bucket_name: str, version: str, metrics: float, plots: Figure):
    s3 = boto3.client("s3")

    s3_prefix = f"{version}/"

    metrics_key = s3_prefix + "metrics.txt"
    with open("/tmp/metrics.txt", "w") as f:
        f.write(str(metrics))
    s3.upload_file("/tmp/metrics.txt", bucket_name, metrics_key)

    plot_key = s3_prefix + "calibration_plot.png"
    plots.savefig("/tmp/calibration_plot.png")
    s3.upload_file("/tmp/calibration_plot.png", bucket_name, plot_key)


def get_all_available_train_versions(bucket_name: str):
    s3 = boto3.client("s3")
    objects = s3.list_objects(Bucket=bucket_name, Prefix="")
    version_names = [Path(obj["Key"]).stem for obj in objects.get("Contents", [])]
    return version_names


def get_latest_version(bucket_name: str, filename: str) -> str:
    s3 = boto3.client("s3")

    # List objects in the specified prefix
    objects = s3.list_objects(Bucket=bucket_name, Prefix="")

    dirs = {
        obj["Key"].split("/")[0]
        for obj in objects.get("Contents", [])
        if obj["Size"] == 0 and obj["Key"].endswith("/")
    }
    versions = [
        (obj["Key"].split("/")[0], obj["LastModified"])
        for obj in objects.get("Contents", [])
        if obj["Key"].endswith(filename) and obj["Key"].split("/")[0] in dirs
    ]

    return max(versions, key=lambda t: t[1])[0]


def get_best_version(train_artifacts_root_path: os.PathLike):
    train_dir = Path(train_artifacts_root_path)
    if "best_model" not in set(f for f in train_dir.iterdir() if f.is_file()):
        return None
    with open(train_dir / "best_model") as f:
        return f.read()


def tag_best_version(train_version: str, train_artifacts_root_path: os.PathLike):
    train_dir = Path(train_artifacts_root_path)
    with open(train_dir / "best_model", "w") as f:
        f.write(train_version)


def prediction_logging_func(predictions: list[tuple[Point, float]]):
    print("yay")  # noqa: T201
    return True, None
