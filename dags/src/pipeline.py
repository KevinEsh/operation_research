import polars as pl
import polars.selectors as cs

from src.dataloaders import get_postgres_uri, get_s3_credentials, get_s3_path
from src.queries import query_snapshot_raw

# def save_parquet_to_s3(df: pl.DataFrame, s3_file_path: str, storage_options: dict):
#     # choossing the best compression for a LightGBM model
#     df.write_parquet(
#         file=s3_file_path,
#         # compression="zstd",
#         storage_options=storage_options,
#         # row_group_size=1000000,  # Uncomment if you need to optimize for large datasets
#         # partition_by=["store_id", "product_id"],  # Uncomment if you want to optimize
#     )


def create_train_dataset(timestamp: str, date_from: str, date_upto: str) -> None:
    """
    Download snapshot data for training.

    Args:
        date_from (str): Start date for training data.
        date_upto (str): End date for training data.
    """
    # Format the query with the provided date range

    # Execute the query against the PostgreSQL database and get the polars DataFrame
    query_snapshot = query_snapshot_raw.format(date_from=date_from, date_upto=date_upto)
    print(query_snapshot)
    train_snapshot_df = pl.read_database_uri(query=query_snapshot, uri=get_postgres_uri("dbcore"))
    print(train_snapshot_df)
    # Then write the DataFrame to a Parquet file in S3 location
    s3_snapshot_path = get_s3_path(timestamp) + "train_snapshot.parquet"
    train_snapshot_df.write_parquet(
        file=s3_snapshot_path,
        storage_options=get_s3_credentials(),
    )

    return s3_snapshot_path


def fill_nulls(df: pl.DataFrame, target_cols: list[str], agg_level: list[str]) -> pl.DataFrame:
    """
    Get target DataFrame for training.

    Args:
        df (pl.LazyFrame): LazyFrame containing training data.
        target_cols (str): Column name for the target variable.
    Returns:
        pl.LazyFrame: LazyFrame with target variable.
    """

    return df.with_columns(
        df.group_by(agg_level, maintain_order=True)
        .agg(pl.col(name).fill_null(strategy="forward") for name in target_cols)
        .explode(target_cols)
        .select(target_cols)
    )

    # return tmp.select(target_cols)


def split_train_valid(df_snapshot, split_date):
    from datetime import date

    split_date = date.fromisoformat(split_date)
    x_train = df_snapshot.filter(pl.col.c_date < split_date)
    x_valid = df_snapshot.filter(pl.col.c_date >= split_date)
    return x_train, x_valid


# get target set for train set and validation set
def pop_columns(df: pl.DataFrame, col_names: list[str]) -> pl.DataFrame:
    return pl.DataFrame([df.drop_in_place(col_name) for col_name in col_names])


def drop_columns(df: pl.DataFrame, col_names: list[str]) -> pl.DataFrame:
    """
    Drop specified columns from the DataFrame.

    Args:
        df (pl.DataFrame): DataFrame to drop columns from.
        col_names (list[str]): list of column names to drop.

    Returns:
        pl.DataFrame: DataFrame with specified columns dropped.
    """
    for col_name in col_names:
        df.drop_in_place(col_name)


def rolling_features(
    df: pl.LazyFrame,
    target: str,
    horizon: int,
    agg_level: list[str],
    fill_nulls: bool = False,
) -> pl.DataFrame:
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


def save_parquet_to_s3(df: pl.DataFrame, s3_file_path: str):
    # choossing the best compression for a LightGBM model
    df.write_parquet(
        file=s3_file_path,
        # compression="zstd",
        storage_options=get_s3_credentials(),
        # row_group_size=1000000,  # Uncomment if you need to optimize for large datasets
        # partition_by=["store_id", "product_id"],  # Uncomment if you want to optimize
    )


def load_parquet_from_s3(s3_file_path: str) -> pl.DataFrame:
    """
    Load a Parquet file from S3.

    Args:
        s3_file_path (str): S3 file path to load the Parquet file from.

    Returns:
        pl.DataFrame: Loaded DataFrame.
    """
    return pl.read_parquet(s3_file_path, storage_options=get_s3_credentials())


def feature_engineering(s3_snapshot_path: str, target: str, horizon: int, timestamp: str):
    df_snapshot = load_parquet_from_s3(s3_snapshot_path)

    # Get a dataframe with target variables
    agg_level = ["product_id", "store_id"]
    target_cols = [f"h{h}_{target}" for h in range(1, horizon + 1)]

    df_snapshot = fill_nulls(df_snapshot, target_cols, agg_level)

    df_train_input, df_valid_input = split_train_valid(df_snapshot, "2016-01-01")

    df_train_target = df_train_input.select(target_cols)
    df_valid_target = df_valid_input.select(target_cols)

    # Remove target columns from df_train_snapshot
    drop_columns(df_train_input, target_cols)
    drop_columns(df_valid_input, target_cols)

    # Apply feature engineering
    df_train_input = rolling_features(df_train_input, target, horizon, agg_level)
    df_train_dates = pl.DataFrame(df_train_input.drop_in_place("c_date"))

    df_valid_input = rolling_features(df_valid_input, target, horizon, agg_level)
    df_valid_dates = pl.DataFrame(df_valid_input.drop_in_place("c_date"))

    # Save the final datasets
    s3_train_input_path = get_s3_path(timestamp) + "train_input.parquet"
    s3_train_target_path = get_s3_path(timestamp) + "train_target.parquet"
    s3_train_dates_path = get_s3_path(timestamp) + "train_dates.parquet"

    s3_valid_input_path = get_s3_path(timestamp) + "valid_input.parquet"
    s3_valid_target_path = get_s3_path(timestamp) + "valid_target.parquet"
    s3_valid_dates_path = get_s3_path(timestamp) + "valid_dates.parquet"

    save_parquet_to_s3(df_train_input, s3_train_input_path)
    save_parquet_to_s3(df_train_target, s3_train_target_path)
    save_parquet_to_s3(df_train_dates, s3_train_dates_path)

    save_parquet_to_s3(df_valid_input, s3_valid_input_path)
    save_parquet_to_s3(df_valid_target, s3_valid_target_path)
    save_parquet_to_s3(df_valid_dates, s3_valid_dates_path)

    return {
        "train_paths": [s3_train_input_path, s3_train_target_path, s3_train_dates_path],
        "valid_paths": [s3_valid_input_path, s3_valid_target_path, s3_valid_dates_path],
    }
