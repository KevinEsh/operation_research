# execute duckdb query and transform the result into a parquet file
import sys
from pathlib import Path
from typing import List

import click
import polars as pl
import polars.selectors as cs

# Agregar el directorio padre (services) al path
current_file = Path(__file__)
services_dir = current_file.parent.parent
sys.path.insert(0, str(services_dir))

try:
    from shared.s3config import get_s3_params
except ImportError:
    raise ImportError("shared.s3config module not found. Ensure the path is correct.")


def extract_target(
    dataset: pl.DataFrame, target_cols: List[str], agg_level: List[str]
) -> pl.DataFrame:
    """
    Get target DataFrame for training.

    Args:
        dataset (pl.LazyFrame): LazyFrame containing training data.
        target_cols (str): Column name for the target variable.
    Returns:
        pl.LazyFrame: LazyFrame with target variable.
    """

    tmp = (
        dataset.group_by(agg_level, maintain_order=True)
        .agg(pl.col(name).fill_null(strategy="forward") for name in target_cols)
        .explode(target_cols)
    )

    return tmp.select(target_cols)


# get target set for train set and validation set
def pop_columns(df: pl.DataFrame, col_names: List[str]) -> pl.DataFrame:
    return pl.DataFrame([df.drop_in_place(col_name) for col_name in col_names])


def drop_columns(df: pl.DataFrame, col_names: List[str]) -> pl.DataFrame:
    """
    Drop specified columns from the DataFrame.

    Args:
        df (pl.DataFrame): DataFrame to drop columns from.
        col_names (List[str]): List of column names to drop.

    Returns:
        pl.DataFrame: DataFrame with specified columns dropped.
    """
    for col_name in col_names:
        df.drop_in_place(col_name)


def rolling_features(
    df: pl.LazyFrame,
    target: str,
    horizon: int,
    agg_level: List[str],
    fill_nulls: bool = False,
) -> pl.LazyFrame:
    """
    Feature engineering for sales data.

    Args:
        df (pl.LazyFrame): LazyFrame containing sales data.

    Returns:
        pl.LazyFrame: LazyFrame with engineered features.
    """

    # 2. Add calendar features
    df = df.with_columns(
        [
            pl.col("c_date").dt.weekday().alias("dayofweek"),
            pl.col("c_date").dt.day().alias("dayofmonth"),
            pl.col("c_date").dt.ordinal_day().alias("dayofyear"),
            pl.col("c_date").dt.week().alias("weekofyear"),
            pl.col("c_date").dt.month().alias("month"),
            pl.col("c_date").dt.year().alias("year"),
        ]
    )

    # 4. Add rolling features over the previous 3, 7, 14, and 28 days
    for window in [3, 7, 14, 21, 28]:
        # Only consider window before current c_date to avoid data leakage. This is done by using closed='left'
        tmp = (
            df.rolling("c_date", period=f"{window}d", closed="right", group_by=agg_level)
            .agg(
                # 4.1 Calculate rolling mean, median, std, min, max, and ewm_mean to log_units_sold
                pl.col(target).mean().alias(f"mean_{window}d_{target}"),
                pl.col(target).median().alias(f"median_{window}d_{target}"),
                pl.col(target).std().alias(f"std_{window}d_{target}"),
                pl.col(target).min().alias(f"min_{window}d_{target}"),
                pl.col(target).max().alias(f"max_{window}d_{target}"),
                # pl.col(target).ewm_mean(alpha=0.9, adjust=True).last().alias(f"ewm_{window}d_{target}"),
                pl.col(target).diff().mean().alias(f"diff_mean_{window}d_{target}"),
            )
            .with_columns(
                # 4.3 Calculate ratio of max to mean. Useful to identify outliers
                (pl.col(f"max_{window}d_{target}") / pl.col(f"mean_{window}d_{target}")).alias(
                    f"max_mean_ratio_{window}d_{target}"
                )
            )
        )
        # df = pl.concat([df, tmp], how="horizontal", parallel=True)
        df = df.join(tmp, on=agg_level + ["c_date"], how="left")

    # 5. Add weekday rolling mean. e.i. mean of the same weekday in the past 4 weeks
    # TODO: No funciona bien, el primer problema es que el rolling semana no es correcto, checa over()
    # el segundo problema es que no se esta considerando el dia de la semana, se deben de crear columns para cada dia de la semana
    # for weekday in range(1, 8):
    #     df = df.with_columns(
    #         pl.col(target)
    #             .rolling_mean_by('date', window_size=f"3w", closed="right")
    #             .over(agg_level + ["weekday"])
    #             .alias(f"mean_3w_{weekday}wd_{target}")
    #     )

    # 5. Add yearly rolling ewm. e.i. ewm of the same day in the past 3 years
    feature_name = f"ewm_3y_{target}"

    tmp = (
        df.rolling("c_date", period="3y", closed="right", group_by=agg_level + ["dayofyear"])
        .agg(pl.col(target).ewm_mean(alpha=0.9).last().alias(feature_name))
        .with_columns(
            (pl.col("c_date").dt.offset_by("1y").dt.offset_by(f"-{h}d")).alias(f"h{h}_c_date")
            for h in range(1, horizon + 1)
        )
    )

    for h in range(1, horizon + 1):
        df = df.join(
            tmp.select(*agg_level, f"h{h}_c_date", feature_name).rename(
                {feature_name: f"h{h}_{feature_name}"}
            ),
            left_on=agg_level + ["c_date"],
            right_on=agg_level + [f"h{h}_c_date"],
            how="left",
        )

    # # 8. Finally fills null values with 0
    # if fill_nulls:
    #     df = df.fill_null(0)

    # 9. Filter by date range
    df = df.with_columns(cs.string().cast(pl.Categorical))

    return df.sort(by=["product_id", "store_id", "c_date"])


# Get columns for horizons
def apply_horizon_shifting(df: pl.DataFrame, target: str, horizons: int, agg_level: List[str]):
    # Add predictions columns for horizons
    for horizon in range(1, horizons + 1):
        tmp = df.select(
            *agg_level,
            pl.col("c_date") - pl.duration(days=horizon),
            pl.col(target).alias(f"h{horizon}_{target}"),
        )

    return df.join(tmp, on=agg_level + ["c_date"], how="left")


def save_parquet(df: pl.DataFrame, path: str):
    # choossing the best compression for a LightGBM model
    df.write_parquet(
        path,
        compression="zstd",
        # row_group_size=1000000,  # Uncomment if you need to optimize for large datasets
        # partition_by=["store_id", "product_id"],  # Uncomment if you want to optimize
    )


def save_parquet_to_s3(df: pl.DataFrame, s3_file_path: str, storage_options: dict):
    # choossing the best compression for a LightGBM model
    df.write_parquet(
        file=s3_file_path,
        compression="zstd",
        storage_options=storage_options,
        # row_group_size=1000000,  # Uncomment if you need to optimize for large datasets
        # partition_by=["store_id", "product_id"],  # Uncomment if you want to optimize
    )


def run_feature_engineering(s3_snapshot_path: str, target: str, horizon: int, timestamp: str):
    s3_path, s3_storage_options = get_s3_params(timestamp)
    df_train_snapshot = pl.read_parquet(s3_snapshot_path, storage_options=s3_storage_options)

    # Get a dataframe with target variables
    agg_level = ["product_id", "store_id"]
    target_cols = [f"h{h}_{target}" for h in range(1, horizon + 1)]
    df_train_target = extract_target(df_train_snapshot, target_cols, agg_level)

    # Remove target columns from df_train_snapshot
    drop_columns(df_train_snapshot, target_cols)

    # Apply feature engineering
    df_train_input = rolling_features(df_train_snapshot, target, horizon, agg_level)
    df_train_dates = pop_columns(df_train_input, ["c_date"])

    # Save the final datasets
    input_path = s3_path + "/train_input.parquet"
    target_path = s3_path + "/train_target.parquet"
    dates_path = s3_path + "/train_dates.parquet"

    save_parquet_to_s3(df_train_input, input_path, s3_storage_options)
    save_parquet_to_s3(df_train_target, target_path, s3_storage_options)
    save_parquet_to_s3(df_train_dates, dates_path, s3_storage_options)

    return input_path, target_path, dates_path


def run_feature_engineering_dev(
    file_path: str,
    target: str,
    horizon: int,
    timestamp: str,
):
    # Load the snapshot data
    df_train_snapshot = pl.read_parquet(file_path)

    # Get a dataframe with target variables
    agg_level = ["product_id", "store_id"]
    target_cols = [f"h{h}_{target}" for h in range(1, horizon + 1)]
    df_train_target = extract_target(df_train_snapshot, target_cols, agg_level)

    # Remove target columns from df_train_snapshot
    drop_columns(df_train_snapshot, target_cols)

    # Apply feature engineering
    df_train_input = rolling_features(df_train_snapshot, target, horizon, agg_level)
    df_train_dates = pop_columns(df_train_input, ["c_date"])

    # Save the final datasets
    save_parquet(df_train_input, f"{timestamp}/train_input.parquet")
    save_parquet(df_train_target, f"{timestamp}/train_target.parquet")
    save_parquet(df_train_dates, f"{timestamp}/train_dates.parquet")

    return


@click.command()
@click.option(
    "--file_path",
    default="../dbcore/data/snapshots/train_snapshot.parquet",
    help="Path to the training snapshot data.",
)
@click.option("--target", default="log_units_sold", help="Target variable for prediction.")
@click.option("--horizon", default=7, help="Number of days to predict.")
@click.option("--timestamp", default="2016-08-15", help="Timestamp for the snapshot.")
def cli_run_feature_engineering(file_path, target: str, horizon: int, timestamp: str) -> None:
    if file_path.startswith("s3://"):
        run_feature_engineering(file_path, target, horizon, timestamp)
    else:
        run_feature_engineering_dev(file_path, target, horizon, timestamp)


if __name__ == "__main__":
    cli_run_feature_engineering()
