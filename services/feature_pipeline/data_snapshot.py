import sys
from pathlib import Path

import click
import polars as pl

# Agregar el directorio padre (services) al path
current_file = Path(__file__)
services_dir = current_file.parent.parent
sys.path.insert(0, str(services_dir))

try:
    from shared.s3config import get_postgres_uri, get_s3_params
except ImportError:
    raise ImportError("shared.s3config module not found. Ensure the path is correct.")


def download_snapshot_dev(date_from: str, date_upto: str) -> None:
    """
    Download snapshot data for training.

    Args:
        date_from (str): Start date for training data.
        date_upto (str): End date for training data.
    """
    # Format the query with the provided date range
    import duckdb
    from snapshot import query_snapshot

    query_snapshot = query_snapshot.format(date_from=date_from, date_upto=date_upto)

    with duckdb.connect(database="../dbcore/data/core.db", read_only=True) as con:
        train_snapshot_df = con.execute(query_snapshot).pl()
    # "../../data/favorita_dataset/snapshots/train_snapshot.parquet"
    train_snapshot_df.write_parquet(
        "../dbcore/data/snapshots/" + "train_snapshot.parquet", compression="zstd"
    )
    return


def download_snapshot(date_from: str, date_upto: str, timestamp: str) -> None:
    """
    Download snapshot data for training.

    Args:
        date_from (str): Start date for training data.
        date_upto (str): End date for training data.
    """
    # Format the query with the provided date range
    from snapshot import query_snapshot

    # Execute the query against the PostgreSQL database and get the polars DataFrame
    query_snapshot = query_snapshot.format(date_from=date_from, date_upto=date_upto)
    train_snapshot_df = pl.read_database_uri(query=query_snapshot, uri=get_postgres_uri())

    # Then write the DataFrame to a Parquet file in S3 location
    s3_path, s3_storage_options = get_s3_params(timestamp)
    s3_snapshot_path = s3_path + "/train_snapshot.parquet"
    train_snapshot_df.write_parquet(s3_snapshot_path, storage_options=s3_storage_options)

    return s3_snapshot_path


@click.command()
@click.option("--train_from", default="2013-01-01", help="Start date for training data.")
@click.option("--train_upto", default="2016-08-15", help="End date for training data.")
@click.option("--timestamp", default="2016-08-15", help="Timestamp for the snapshot.")
def cli_download_snapshot(train_from: str, train_upto: str, timestamp: str) -> None:
    download_snapshot(train_from, train_upto, timestamp)


if __name__ == "__main__":
    cli_download_snapshot()
