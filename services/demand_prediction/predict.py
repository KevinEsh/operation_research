# get the list of models
import sys
from datetime import date, timedelta
from os import getcwd, listdir
from os.path import abspath, join

import click
import joblib
import polars as pl

sys.path.append(abspath(join(getcwd(), "..")))
from shared.forecasters import DirectMultihorizonForecaster

MODELS_PATH = "./models"


def get_latest_model() -> DirectMultihorizonForecaster:
    models = sorted([f for f in listdir(MODELS_PATH) if f.endswith(".pkl")], reverse=True)

    path_to_latest_model = models[0] if models else None
    return joblib.load(join(MODELS_PATH, path_to_latest_model))


def split_by_interval(x_train, c_train, date_interval):
    return (
        x_train.with_columns(c_train.get_column("c_date"))
        .filter(pl.col.c_date.is_between(*date_interval))
        .drop("c_date")
    )


def get_single_input(today: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load the training data for the demand prediction.
    """

    x_train = pl.read_parquet("../../data/favorita_dataset/output/train_input.parquet")
    c_train = pl.read_parquet("../../data/favorita_dataset/output/train_dates.parquet")
    # y_train = pl.read_parquet("../../data/favorita_dataset/output/train_target.parquet")
    # TODO: remove this when the data pipeline is fixed
    x_train = x_train.with_columns(pl.col.product_group.cast(pl.Categorical))

    return split_by_interval(x_train, c_train, (today, today))


def run_prediction(today: str) -> None:
    today = date.fromisoformat(today)

    x_input = get_single_input(today)
    demand_predictor = get_latest_model()

    inverse_transform = (pl.all().exp() - 1).round().cast(pl.Int32)
    df_demand_predictions = demand_predictor.predict(x_input).with_columns(inverse_transform)

    prediction_range = [today + timedelta(days=h) for h in range(1, demand_predictor.horizons + 1)]

    return x_input.select(
        pl.col.product_id.alias("dp_p_id"),
        pl.col.store_id.alias("dp_s_id"),
        pl.lit(prediction_range).alias("dp_date"),
        pl.concat_list(df_demand_predictions).alias("dp_mean"),
    ).explode("dp_date", "dp_mean")


def save_predictions(predictions: pl.DataFrame, today: str) -> None:
    """
    Save the predictions to a Parquet file.
    """
    output_path = f"../dbcore/data/demandpredictions_{today.replace('-', '')}.parquet"
    predictions.write_parquet(output_path, compression="snappy", row_group_size=1000000)
    print(f"Predictions saved to {output_path}")


@click.command()
@click.option("--today", default="2016-08-15", help="The date for which to run the prediction.")
def main(today: str) -> None:
    """Main function to run the demand prediction."""
    df_demand_predictions = run_prediction(today)
    save_predictions(df_demand_predictions, today)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Ejecuta la predicción de demanda.")
    # parser.add_argument("--today", type=str, default="2016-08-15", help="Fecha para la predicción (YYYY-MM-DD).")
    # args = parser.parse_args()
    # main(args.today)
    main()
