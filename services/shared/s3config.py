from os import environ

from dotenv import load_dotenv

load_dotenv("../../.env")


def get_postgres_uri() -> str:
    """
    Construct the PostgreSQL URI from environment variables.
    """
    POSTGRES_USER = environ.get("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = environ.get("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST = environ.get("POSTGRES_HOST", "localhost")
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/dbcore"


def get_s3_params(timestamp: str) -> tuple:
    s3_path = f"s3://{environ.get('S3_SNAPSHOT_BUCKET_NAME')}/{timestamp}"

    s3_storage_options = {
        "aws_access_key_id": environ.get("MM_ACCESS_KEY", "dummy_access_key"),
        "aws_secret_access_key": environ.get("MM_SECRET_KEY", "dummy_secret_key"),
        "aws_endpoint_url": environ.get("MM_ENDPOINT_URL", "http://localhost:10000"),
        "aws_region": "us-east-1",
    }
    return s3_path, s3_storage_options


def get_s3_model_storage(timestamp: str):
    s3_path = f"s3://mlflow/artifacts/{timestamp}"
    s3_storage_options = {
        "key": environ.get("MM_ACCESS_KEY", "dummy_access_key"),
        "secret": environ.get("MM_SECRET_KEY", "dummy_secret_key"),
        "client_kwargs": {
            "endpoint_url": environ.get("MM_ENDPOINT_URL", "http://localhost:10000"),
            "region_name": "us-east-1",
        },
    }
    return s3_path, s3_storage_options


# def get_latest_model() -> DirectMultihorizonForecaster:
#     models = sorted([f for f in listdir(MODELS_PATH) if f.endswith(".pkl")], reverse=True)

#     path_to_latest_model = models[0] if models else None
#     return joblib.load(join(MODELS_PATH, path_to_latest_model))
