# get the list of models
import sys
from datetime import date, timedelta
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
    from shared.s3config import get_s3_params, pull_model_from_s3
except ImportError:
    raise ImportError("shared.s3config module not found. Ensure the path is correct.")


def split_by_interval(x_train, c_train, date_interval):
    return (
        x_train.with_columns(c_train.get_column("c_date"))
        .filter(pl.col.c_date.is_between(*date_interval))
        .drop("c_date")
    )


def get_single_input(timestamp: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load the training data for the demand prediction.
    """
    s3_path, s3_storage_options = get_s3_params(timestamp)

    x_train = pl.read_parquet(s3_path + "/train_input.parquet", storage_options=s3_storage_options)
    c_train = pl.read_parquet(s3_path + "/train_dates.parquet", storage_options=s3_storage_options)

    date_interval = (date.fromisoformat(timestamp), date.fromisoformat(timestamp))
    return split_by_interval(x_train, c_train, date_interval)


def run_prediction(today: str, s3_model_path: str) -> None:
    x_input = get_single_input(today)
    demand_predictor = pull_model_from_s3(s3_model_path)

    inverse_transform = (pl.all().exp() - 1).round().cast(pl.Int32)
    df_demand_predictions = demand_predictor.predict(x_input).with_columns(inverse_transform)

    prediction_range = [
        date.fromisoformat(today) + timedelta(days=h)
        for h in range(1, demand_predictor.horizons + 1)
    ]

    df_demand_predictions = x_input.select(
        pl.col.product_id.alias("dp_p_id"),
        pl.col.store_id.alias("dp_s_id"),
        pl.lit(prediction_range).alias("dp_date"),
        pl.concat_list(df_demand_predictions).alias("dp_mean"),
    ).explode("dp_date", "dp_mean")

    # print(df_demand_predictions)
    s3_dp_path = save_predictions(df_demand_predictions, today)
    return s3_dp_path


def save_predictions(df_demand_predictions: pl.DataFrame, timestamp: str) -> None:
    """
    Save the predictions to a Parquet file.
    """
    s3_path, s3_storage_options = get_s3_params(timestamp)
    s3_demand_predictions_path = s3_path + "/demand_predictions.parquet"
    df_demand_predictions.write_parquet(
        s3_demand_predictions_path, storage_options=s3_storage_options
    )
    return s3_demand_predictions_path


@click.command()
@click.option("--today", default="2016-08-15", help="The date for which to run the prediction.")
@click.option(
    "--s3_model_path",
    default="s3://mlflow/artifacts/2016-08-15/demand_forecaster.pkl",
    help="S3 path to the model.",
)
def cli_run_prediction(today: str, s3_model_path: str) -> None:
    """Main function to run the demand prediction."""
    s3_demand_predictions_path = run_prediction(today, s3_model_path)
    print(f"Predictions saved to {s3_demand_predictions_path}")


if __name__ == "__main__":
    cli_run_prediction()
