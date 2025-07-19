from typing import List

import polars as pl


def extract_target_df(dataset: pl.DataFrame, target_cols: List[str], agg_level: List[str]) -> pl.DataFrame:
    """
    Get target DataFrame for training.

    Args:
        dataset (pl.LazyFrame): LazyFrame containing training data.
        target_cols (str): Column name for the target variable.
    Returns:
        pl.LazyFrame: LazyFrame with target variable.
    """
    # fill nulls in target columns using forward fill strategy

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


def feature_engineering(
    train_lzdf: pl.LazyFrame,
    target: str,
    horizon: int,
    agg_level: List[str],
    fill_nulls: bool = False,
) -> pl.LazyFrame:
    """
    Feature engineering for sales data.

    Args:
        train_lzdf (pl.LazyFrame): LazyFrame containing sales data.

    Returns:
        pl.LazyFrame: LazyFrame with engineered features.
    """

    # 2. Add calendar features
    train_lzdf = train_lzdf.with_columns(
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
            train_lzdf.rolling("c_date", period=f"{window}d", closed="right", group_by=agg_level)
            .agg(
                # 4.1 Calculate rolling mean, median, std, min, max, and ewm_mean to log_units_sold
                pl.col("log_units_sold").mean().alias(f"mean_{window}d_log_units_sold"),
                pl.col("log_units_sold").median().alias(f"median_{window}d_log_units_sold"),
                pl.col("log_units_sold").std().alias(f"std_{window}d_log_units_sold"),
                pl.col("log_units_sold").min().alias(f"min_{window}d_log_units_sold"),
                pl.col("log_units_sold").max().alias(f"max_{window}d_log_units_sold"),
                pl.col("log_units_sold").ewm_mean(alpha=0.9, adjust=True).last().alias(f"ewm_{window}d_log_units_sold"),
                pl.col("log_units_sold").diff().mean().alias(f"diff_mean_{window}d_log_units_sold"),
            )
            .with_columns(
                # 4.3 Calculate ratio of max to mean. Useful to identify outliers
                (pl.col(f"max_{window}d_log_units_sold") / pl.col(f"mean_{window}d_log_units_sold")).alias(
                    f"max_mean_ratio_{window}d_log_units_sold"
                )
            )
        )
        # train_lzdf = pl.concat([train_lzdf, tmp], how="horizontal", parallel=True)
        train_lzdf = train_lzdf.join(tmp, on=agg_level + ["c_date"], how="left")

    # 5. Add weekday rolling mean. e.i. mean of the same weekday in the past 4 weeks
    # TODO: No funciona bien, el primer problema es que el rolling semana no es correcto, checa over()
    # el segundo problema es que no se esta considerando el dia de la semana, se deben de crear columns para cada dia de la semana
    # for weekday in range(1, 8):
    #     train_lzdf = train_lzdf.with_columns(
    #         pl.col("log_units_sold")
    #             .rolling_mean_by('date', window_size=f"3w", closed="right")
    #             .over(agg_level + ["weekday"])
    #             .alias(f"mean_3w_{weekday}wd_log_units_sold")
    #     )

    # 5. Add yearly rolling ewm. e.i. ewm of the same day in the past 3 years
    feature_name = "ewm_3y_log_units_sold"

    tmp = (
        train_lzdf.rolling("c_date", period="3y", closed="right", group_by=agg_level + ["dayofyear"])
        .agg(pl.col.log_units_sold.ewm_mean(alpha=0.9).last().alias(feature_name))
        .with_columns(
            (pl.col("c_date").dt.offset_by("1y").dt.offset_by(f"-{h}d")).alias(f"h{h}_c_date")
            for h in range(1, horizon + 1)
        )
    )

    for h in range(1, horizon + 1):
        train_lzdf = train_lzdf.join(
            tmp.select(*agg_level, f"h{h}_c_date", feature_name).rename({feature_name: f"h{h}_{feature_name}"}),
            left_on=agg_level + ["c_date"],
            right_on=agg_level + [f"h{h}_c_date"],
            how="left",
        )

    # # 8. Finally fills null values with 0
    # if fill_nulls:
    #     train_lzdf = train_lzdf.fill_null(0)

    # 9. Filter by date range
    return train_lzdf.sort(by=["product_id", "store_id", "c_date"])


# Get columns for horizons
def apply_horizon_shifting(train_dataset: pl.DataFrame, horizons: int, agg_level: List[str]):
    # Add predictions columns for horizons
    for horizon in range(1, horizons + 1):
        tmp = train_dataset.select(
            *agg_level,
            pl.col("date") - pl.duration(days=horizon),
            pl.col("log_units_sold").alias(f"h{horizon}_log_units_sold"),
        )

        train_dataset = train_dataset.join(tmp, on=agg_level + ["date"], how="left")
    return train_dataset


def save_df(df: pl.DataFrame, path: str):
    # choossing the best compression for a LightGBM model
    df.write_parquet(
        path,
        compression="zstd",
        # row_group_size=1000000,  # Uncomment if you need to optimize for large datasets
        # partition_by=["store_id", "product_id"],  # Uncomment if you want to optimize
    )


def train_pipeline(
    dataset: pl.DataFrame,
    target: str,
    horizon: int,
    agg_level: List[str],
):
    target_cols = [f"h{h}_{target}" for h in range(1, horizon + 1)]
    target_df = extract_target_df(dataset, target_cols, agg_level)
    drop_columns(dataset, target_cols)
    input_df = feature_engineering(dataset, target, horizon, agg_level)
    dates_df = pop_columns(input_df, ["c_date"])

    save_df(target_df, "../../data/favorita_dataset/output/train_target.parquet")
    save_df(input_df, "../../data/favorita_dataset/output/train_input.parquet")
    save_df(dates_df, "../../data/favorita_dataset/output/train_dates.parquet")

    return input_df, target_df, dates_df
