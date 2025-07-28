from os import environ

import polars as pl
from requests import post

query_table_map_raw = """SELECT {name_col}, {id_col} FROM {table_name}"""


API_URI = "http://dbcore-api:80"


def map_names_to_ids(
    df: pl.DataFrame, mappers: dict[str, dict[str, int]], renamers: dict[str, str]
) -> pl.DataFrame:
    """
    Map names in the DataFrame to IDs using the provided mappers.
    """
    return df.with_columns(
        [pl.col(colname).replace_strict(names_to_ids) for colname, names_to_ids in mappers.items()]
    ).rename(renamers)


def load_dataframe_to_db(df, endpoint):
    url = f"{API_URI}/{endpoint}"
    headers = {"Content-Type": "application/json"}

    # Convert date, datetime, and time columns to string before sending
    # This is necessary because Polars does not support direct JSON serialization of these types
    # and the API expects them as strings.
    df = df.with_columns(pl.col(pl.Date, pl.Datetime, pl.Time).cast(str))

    post_response = post(url, json=df.to_dicts(), headers=headers)
    print(url, post_response.status_code)
    if post_response.status_code != 200:
        print("Error:", post_response.text)
        return {}
    return post_response.json()


def get_s3_path(timestamp: str):
    """
    Construct the S3 path using the provided timestamp.
    """
    return f"s3://{environ.get('GATEWAY_BUCKET_NAME', 'snapshots')}/{timestamp}/"


def get_s3_credentials() -> tuple:
    return {
        "aws_access_key_id": environ.get("AWS_ACCESS_KEY", "admin"),
        "aws_secret_access_key": environ.get("AWS_SECRET_KEY", "password"),
        "aws_endpoint_url": environ.get("AWS_ENDPOINT_URL", "http://minio:9000"),
        "aws_region": environ.get("AWS_REGION", "us-east-1"),
    }


def get_postgres_uri(db: str = "dbcore") -> str:
    """
    Construct the PostgreSQL URI from environment variables.
    """
    POSTGRES_USER = environ.get("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD = environ.get("POSTGRES_PASSWORD", "password")
    POSTGRES_HOST = environ.get("POSTGRES_HOST", "database-server")
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{db}"


def get_datamap_to_ids(table_name: str, name_col: str, id_col: str):
    # Execute the query against the PostgreSQL database and get the polars DataFrame
    query_table_map = query_table_map_raw.format(
        name_col=name_col, id_col=id_col, table_name=table_name
    )
    data_df = pl.read_database_uri(query=query_table_map, uri=get_postgres_uri("dbcore"))
    return {row[name_col]: row[id_col] for row in data_df.to_dicts()}


def get_dataframe_from_s3(timestamp: str, table_name: str):
    """
    Load a DataFrame from S3 using the timestamp and table name.
    """
    return pl.read_parquet(
        source=f"{get_s3_path(timestamp)}{table_name}.parquet",
        storage_options=get_s3_credentials(),
    )
