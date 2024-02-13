import json
import pickle
from io import BytesIO, StringIO
from tempfile import NamedTemporaryFile
from uuid import uuid4

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression

from ml_pipelines.core.common.feature_eng import FeatureEngineeringParams
from ml_pipelines.core.serve.serve import Point
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
    s3.upload_fileobj(
        Fileobj=BytesIO(pickle.dumps(artifacts["model"])),
        Bucket=bucket_name,
        Key=model_pickle_key,
    )

    feature_eng_params_key = s3_prefix + "feature_eng_params.json"
    s3.upload_fileobj(
        Fileobj=BytesIO(artifacts["feature_eng_params"].json().encode()),
        Bucket=bucket_name,
        Key=feature_eng_params_key,
    )

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

    model: LogisticRegression = pickle.load(
        s3.get_object(Bucket=bucket_name, Key=model_key)["Body"]
    )

    feature_eng_params_key = s3_prefix + "feature_eng_params.json"
    fep = s3.get_object(Bucket=bucket_name, Key=feature_eng_params_key)["Body"].read()
    feature_eng_params = FeatureEngineeringParams(**json.loads(fep.decode()))

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
                s3.get_object(Bucket=bucket_name, Key=csv_key)["Body"].read().decode()
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
    s3.upload_fileobj(
        Fileobj=BytesIO(str(metrics).encode()),
        Bucket=bucket_name,
        Key=metrics_key,
    )

    plot_key = s3_prefix + "calibration_plot.png"
    buf = BytesIO()
    plots.savefig(buf)
    buf.seek(0)
    s3.upload_fileobj(
        Fileobj=buf,
        Bucket=bucket_name,
        Key=plot_key,
    )


def get_all_available_train_versions(bucket_name: str):
    s3 = boto3.client("s3")

    # List objects in the specified prefix
    objects = s3.list_objects(Bucket=bucket_name, Prefix="")

    return list(
        {
            obj["Key"].split("/")[0]
            for obj in objects.get("Contents", [])
            if len(obj["Key"].split("/")) == 2  # noqa: PLR2004
        }
    )


def get_latest_version(bucket_name: str, filename: str) -> str:
    s3 = boto3.client("s3")
    objects = s3.list_objects(Bucket=bucket_name, Prefix="")
    depth = 2
    versions = [
        (obj["Key"].split("/")[0], obj["LastModified"])
        for obj in objects.get("Contents", [])
        if obj["Key"].endswith(filename) and len(obj["Key"].split("/")) == depth
    ]
    return max(versions, key=lambda t: t[1])[0]


def get_best_version(bucket_name: str):
    s3 = boto3.client("s3")
    objects = s3.list_objects(Bucket=bucket_name, Prefix="").get("Contents", [])
    keys = {o["Key"] for o in objects}
    if "best_model" not in keys:
        return None
    return (
        s3.get_object(Bucket=bucket_name, Key="best_model")["Body"]
        .read()
        .decode("utf-8")
    )


def tag_best_version(bucket_name: str, train_version: str):
    s3 = boto3.client("s3")
    s3.upload_fileobj(
        Fileobj=BytesIO(train_version.encode()),
        Bucket=bucket_name,
        Key="best_model",
    )


def prediction_logging_func(predictions: list[tuple[Point, float]]):
    print("logged preds!")  # noqa: T201
    return True, None
