import click
import duckdb

SNAPSHOTS_PATH = "../dbcore/data/snapshots/"


def download_snapshot(date_from: str, date_upto: str) -> None:
    """
    Download snapshot data for training.

    Args:
        date_from (str): Start date for training data.
        date_upto (str): End date for training data.
    """
    # Format the query with the provided date range
    from snapshot import query_snapshot

    query_snapshot = query_snapshot.format(date_from=date_from, date_upto=date_upto)

    with duckdb.connect(database="../dbcore/data/core.db", read_only=True) as con:
        train_snapshot_df = con.execute(query_snapshot).pl()
    # "../../data/favorita_dataset/snapshots/train_snapshot.parquet"
    train_snapshot_df.write_parquet(SNAPSHOTS_PATH + "train_snapshot.parquet", compression="zstd")
    return


@click.command()
@click.option("--train_from", default="2013-01-01", help="Start date for training data.")
@click.option("--train_upto", default="2016-08-15", help="End date for training data.")
def cli_download_snapshot(train_from: str, train_upto: str) -> None:
    download_snapshot(train_from, train_upto)


if __name__ == "__main__":
    cli_download_snapshot()
