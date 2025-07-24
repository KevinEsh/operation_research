import sys
from pathlib import Path

import click
import polars as pl
import polars.selectors as cs

# Agregar el directorio padre (services) al path
current_file = Path(__file__)
services_dir = current_file.parent.parent
sys.path.insert(0, str(services_dir))

try:
    from shared.forecasters import DirectMultihorizonForecaster
    from shared.s3config import get_s3_params, push_model_to_s3
except ImportError:
    raise ImportError("shared.s3config module not found. Ensure the path is correct.")


def run_model_fit(timestamp: str) -> str:
    """
    Run the model fitting process.
    """
    # Load training data from S3
    # Assuming the S3 path and storage options are correctly set up
    s3_path, s3_storage_options = get_s3_params(timestamp)
    x_train = pl.read_parquet(s3_path + "/train_input.parquet", storage_options=s3_storage_options)
    y_train = pl.read_parquet(s3_path + "/train_target.parquet", storage_options=s3_storage_options)
    # c_train = pl.read_parquet(s3_path + "/train_dates.parquet", storage_options=s3_storage_options)

    categorical_cols = x_train.select(cs.integer(), cs.categorical()).columns

    model_params = {
        # "boosting_type":"dart", # "gdbt"
        "num_leaves": 31,  # 31
        # "max_depth": 10, #-1,
        "learning_rate": 0.05,
        "n_estimators": 100,
        # "subsample_for_bin": 200000,
        "objective": "regression",
        # class_weight: Optional[Union[Dict, str]] = None,
        "min_split_gain": 0.01,  # 'feature_fraction': 0.1,#0.8,
        # min_child_weight: float = 1e-3,
        "min_child_samples": 10,
        "subsample": 0.7,  #'bagging_fraction': 0.7,
        "subsample_freq": 1,  # 'bagging_freq': 1,
        # colsample_bytree: float = 1.0,
        # reg_alpha: float = 0.0,
        # reg_lambda: float = 0.0,
        # random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        "n_jobs": 16,  # 'num_threads': 16
        # importance_type: str = "split",
    }

    fit_params = {
        "eval_metric": "l2",
        "early_stopping_rounds": 100,
        "log_evaluation": 100,
        "categorical_feature": categorical_cols,
        # "sample_weight": weights,
        # "feature_name": "auto",
    }

    forecaster = DirectMultihorizonForecaster(horizons=7, params=model_params)
    forecaster.fit(x_train, y_train, fit_params)

    return push_model_to_s3(forecaster, timestamp)


@click.command()
@click.option("--timestamp", default="2016-08-15", help="Timestamp for the model.")
def cli_run_model_fit(timestamp: str) -> None:
    """
    Command line interface to run the model fitting process.
    """
    s3_model_path = run_model_fit(timestamp)
    print(f"Model saved to {s3_model_path}")


if __name__ == "__main__":
    cli_run_model_fit()
