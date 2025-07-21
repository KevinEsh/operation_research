# execute duckdb query and transform the result into a parquet file
from typing import List

import click
import polars as pl

SNAPSHOTS_PATH = "../dbcore/data/snapshots/"
FEATURE_STORE_PATH = "../dbcore/data/feature_store/"


def extract_target(dataset: pl.DataFrame, target_cols: List[str], agg_level: List[str]) -> pl.DataFrame:
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


def feature_engineering(
    df: pl.LazyFrame, target: str, horizon: int, agg_level: List[str], fill_nulls: bool = False
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
    feature_name = "ewm_3y_{target}"

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
            tmp.select(*agg_level, f"h{h}_c_date", feature_name).rename({feature_name: f"h{h}_{feature_name}"}),
            left_on=agg_level + ["c_date"],
            right_on=agg_level + [f"h{h}_c_date"],
            how="left",
        )

    # # 8. Finally fills null values with 0
    # if fill_nulls:
    #     df = df.fill_null(0)

    # 9. Filter by date range
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


def run_feature_engineering(
    dataset: pl.DataFrame,
    target: str,
    horizon: int,
    agg_level: List[str],
):
    target_cols = [f"h{h}_{target}" for h in range(1, horizon + 1)]
    df_train_target = extract_target(dataset, target_cols, agg_level)

    # Remove target columns from dataset
    drop_columns(dataset, target_cols)

    # Apply feature engineering
    df_train_input = feature_engineering(dataset, target, horizon, agg_level)
    df_train_dates = pop_columns(df_train_input, ["c_date"])

    save_parquet(df_train_target, "../../data/favorita_dataset/output/train_target.parquet")
    save_parquet(df_train_input, "../../data/favorita_dataset/output/train_input.parquet")
    save_parquet(df_train_dates, "../../data/favorita_dataset/output/train_dates.parquet")

    return df_train_input, df_train_target, df_train_dates


@click.command()
# @click.option("--today", default="2016-08-15", help="Start date for training data.")
@click.option("--target", default="log_units_sold", help="Target variable for prediction.")
@click.option("--horizon", default=7, help="Number of days to predict.")
@click.option("--agg_level", default=["product_id", "store_id"], help="Aggregation level for features.")
def cli_run_feature_engineering(target: str, horizon: int, agg_level: List[str]):
    """
    Main function to run the feature engineering pipeline.
    """

    # Load the snapshot data
    df_train_snapshot = pl.read_parquet(SNAPSHOTS_PATH + "train_snapshot.parquet")

    # Run feature engineering
    df_train_input, df_train_target, df_train_dates = run_feature_engineering(
        df_train_snapshot, target, horizon, agg_level
    )

    # Save the final datasets
    save_parquet(df_train_input, FEATURE_STORE_PATH + "train_input.parquet")
    save_parquet(df_train_target, FEATURE_STORE_PATH + "train_target.parquet")
    save_parquet(df_train_dates, FEATURE_STORE_PATH + "train_dates.parquet")


if __name__ == "__main__":
    cli_run_feature_engineering()
