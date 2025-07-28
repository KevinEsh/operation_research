import os
from datetime import date, timedelta

import mlflow
import polars as pl
import polars.selectors as cs
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from src.dataloaders import get_s3_credentials, get_s3_path, load_dataframe_to_db
from src.forecasters import DirectMultihorizonForecaster

EXPERIMENT_NAME = "demand_predictor_experiment"
MODEL_NAME = "demand_predictor_2016-08-15"
MLFLOW_HOST = os.environ.get("MLFLOW_HOST", "mlflow-server")

os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["AWS_REGION"] = "us-east-1"


def get_eval_sets(x: pl.DataFrame, y: pl.DataFrame) -> list:
    """
    Prepare evaluation sets for validation.
    """
    eval_sets = []
    xv = x.to_pandas()
    for h in range(7):
        yv = y.get_column(f"h{h + 1}_log_units_sold").to_numpy()
        eval_sets.append((xv, yv))
    return eval_sets


def run_model_experiment(timestamp: str, num_trials: int = 30) -> str:
    """
    Run the model fitting process.
    """
    x_train, y_train, x_valid, y_valid = fetch_train_and_valid_data(timestamp)
    eval_train_sets = get_eval_sets(x_train, y_train)
    eval_valid_sets = get_eval_sets(x_valid, y_valid)

    experiment_id = create_or_get_experiment(timestamp)
    mlflow.lightgbm.autolog(disable=True)  # Disable automatic logging to avoid conflicts

    fit_params = {
        "eval_metric": "l2",
        "early_stopping_rounds": 10,
        "log_evaluation": 100,
        "categorical_feature": x_train.select(cs.integer(), cs.categorical()).columns,
        # "sample_weight": weights,
        # "feature_name": "auto",
    }

    def objective(model_params):
        with mlflow.start_run(experiment_id=experiment_id):
            # Create and fit the forecaster
            forecaster = DirectMultihorizonForecaster(horizons=7, params=model_params)
            forecaster.fit(x_train, y_train, x_valid, y_valid, fit_params)

            # train_l2 = {
            #     f"train_h{h}_l2": model.best_score_["train"]["l2"]
            #     for h, model in enumerate(forecaster.models_, 1)
            # }

            # valid_l2 = {
            #     f"valid_h{h}_l2": model.best_score_["valid_1"]["l2"]
            #     for h, model in enumerate(forecaster.models_, 1)
            # }

            train_r2 = {
                f"train_h{h + 1}_score": model.score(*eval_train_sets[h])
                for h, model in enumerate(forecaster.models_)
            }

            valid_r2 = {
                f"valid_h{h + 1}_score": model.score(*eval_valid_sets[h])
                for h, model in enumerate(forecaster.models_)
            }

            val_l2 = sum([model.best_score_["valid_1"]["l2"] for model in forecaster.models_])

            mlflow.log_params(model_params)
            mlflow.log_metrics(train_r2)
            mlflow.log_metrics(valid_r2)
            mlflow.log_metric("valid_score", sum(valid_r2.values()))

        return {"loss": val_l2, "status": STATUS_OK}

    # search_space = {
    #     "max_depth": scope.int(hp.quniform("max_depth", 1, 20, 1)),
    #     "n_estimators": scope.int(hp.quniform("n_estimators", 10, 50, 1)),
    #     "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
    #     "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 4, 1)),
    #     "random_state": 42,
    # }

    search_space = {
        "num_leaves": 100,
        "max_depth": 50,
        "learning_rate": hp.uniform("learning_rate", 1e-5, 0.2),
        "n_estimators": 100,
        "min_child_samples": scope.int(hp.quniform("min_child_samples", 5, 50, 5)),
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 1.0),
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 1.0),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "subsample_freq": scope.int(hp.quniform("subsample_freq", 1, 10, 1)),
        # "min_split_gain": hp.uniform("min_split_gain", 1e-4, 0.1),
        "feature_fraction": 0.6,
        # "bagging_fraction": hp.uniform("bagging_fraction", 0.5, 1.0),
        "n_jobs": 16,
        "verbosity": -1,
    }

    # rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        # rstate=rstate,
    )

    return


def create_or_get_experiment(timestamp: str) -> str:
    """
    Get the experiment ID for the given timestamp.
    """
    mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")

    experiment_name = f"{EXPERIMENT_NAME}_{timestamp}"
    has_exp = mlflow.get_experiment_by_name(experiment_name)
    # If the experiment does not exist, create it
    if has_exp is None:
        return mlflow.create_experiment(experiment_name)
    else:
        return has_exp.experiment_id


# transform str to integer
def parse_params(params: dict) -> dict:
    """Transform string parameters to their appropriate types."""
    transformed_params = {}
    for key, value in params.items():
        if value.isnumeric():
            transformed_params[key] = int(value)
        elif "." in value:
            transformed_params[key] = float(value)
        else:
            transformed_params[key] = value
    return transformed_params


def get_best_params(timestamp: str) -> dict:
    """
    Retrieve the best run from the HPO experiment.
    """
    experiment_id = create_or_get_experiment(timestamp)

    client = MlflowClient()
    best_run = client.search_runs(
        experiment_ids=experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.valid_score DESC"],
    )
    return parse_params(best_run[0].data.params) if best_run else {}


def fetch_train_and_valid_data(timestamp: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fetch training and validation data from S3.
    """
    s3_path = get_s3_path(timestamp)

    x_train = pl.read_parquet(s3_path + "train_input.parquet", storage_options=get_s3_credentials())
    y_train = pl.read_parquet(
        s3_path + "train_target.parquet", storage_options=get_s3_credentials()
    )

    x_valid = pl.read_parquet(s3_path + "valid_input.parquet", storage_options=get_s3_credentials())
    y_valid = pl.read_parquet(
        s3_path + "valid_target.parquet", storage_options=get_s3_credentials()
    )

    return x_train, y_train, x_valid, y_valid


def run_train_and_register_best_model(timestamp: str) -> str:
    """
    Train the best model using the best hyperparameters from the HPO experiment with all the data
    """
    best_params = get_best_params(timestamp)

    # Load training data from S3
    x_train, y_train, x_valid, y_valid = fetch_train_and_valid_data(timestamp)

    x_total = pl.concat([x_train, x_valid], how="vertical")
    y_total = pl.concat([y_train, y_valid], how="vertical")

    fit_params = {
        "eval_metric": "l2",
        "early_stopping_rounds": 10,
        "log_evaluation": 100,
        "categorical_feature": x_total.select(cs.integer(), cs.categorical()).columns,
    }

    experiment_id = create_or_get_experiment(timestamp)
    register_model_name = f"{MODEL_NAME}_{timestamp}"
    mlflow.lightgbm.autolog(disable=True)  # Disable automatic logging to avoid conflicts

    with mlflow.start_run(experiment_id=experiment_id):
        forecaster = DirectMultihorizonForecaster(horizons=7, params=best_params)
        forecaster.fit(x_total, y_total, fit_params=fit_params)

        mlflow.pyfunc.log_model(
            python_model=forecaster,
            # artifact_path="model",
            registered_model_name=register_model_name,
            # code_paths=["..shared/forecasters.py"],
            # pip_requirements=["mlflow", "lightgbm", "polars", "hyperopt"]
        )

    return register_model_name


def load_model_from_registry(timestamp: str) -> DirectMultihorizonForecaster:
    """
    Load the model from the MLflow Model Registry.
    """
    mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
    model_name = f"demand_predictor_{timestamp}"
    model_version = "latest"

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Loading model from URI: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def run_predict_demand(timestamp: str) -> pl.DataFrame:
    """
    Run the demand prediction using the best model.
    """
    demand_predictor = load_model_from_registry(timestamp)
    x_input = get_single_input(timestamp)

    inverse_transform = (pl.all().exp() - 1).round().cast(pl.Int32)
    df_predictions = demand_predictor.predict(x_input).with_columns(inverse_transform)

    prediction_range = [date.fromisoformat(timestamp) + timedelta(days=h) for h in range(1, 7 + 1)]

    df_predictions = x_input.select(
        pl.col.product_id.alias("dp_p_id"),
        pl.col.store_id.alias("dp_s_id"),
        pl.lit(prediction_range).alias("dp_date"),
        pl.concat_list(df_predictions).alias("dp_mean"),
    ).explode("dp_date", "dp_mean")

    load_dataframe_to_db(df_predictions, "demandpredictions")
    return  # save_predictions(df_predictions, timestamp)


def get_single_input(timestamp: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load the training data for the demand prediction.
    """

    s3_path = get_s3_path(timestamp)

    x_valid = pl.read_parquet(s3_path + "valid_input.parquet", storage_options=get_s3_credentials())
    c_train = pl.read_parquet(s3_path + "valid_dates.parquet", storage_options=get_s3_credentials())

    return (
        x_valid.with_columns(c_train.get_column("c_date"))
        .filter(pl.col.c_date == date.fromisoformat(timestamp))
        .drop("c_date")
    )


def save_predictions(df_demand_predictions: pl.DataFrame, timestamp: str) -> None:
    """
    Save the predictions to a Parquet file.
    """
    s3_path = get_s3_path(timestamp)
    s3_demand_predictions_path = s3_path + "demand_predictions.parquet"
    df_demand_predictions.write_parquet(
        s3_demand_predictions_path, storage_options=get_s3_credentials()
    )
    return s3_demand_predictions_path
