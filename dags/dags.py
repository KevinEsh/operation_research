from airflow.decorators import dag, task
from airflow.models.param import Param

# get current date in string format from pendulum
from pendulum import datetime
from src.dataloaders import (
    get_dataframe_from_s3,
    get_datamap_to_ids,
    load_dataframe_to_db,
    map_names_to_ids,
)
from src.forecasttools import run_predict_demand, run_train_and_register_best_model
from src.fullfilment import run_fulfillment
from src.pipeline import create_train_dataset, feature_engineering

# current_date = now().date().isoformat()  # .to_iso8601_string()


@dag(
    dag_id="weekly_run",
    schedule=None,
    start_date=datetime(2025, 7, 28),
    catchup=False,
    params={
        "timestamp": Param(
            default="2016-08-15",
            type="string",
            title="timestamp",
            description="Timestamp for getting data and make prediction",
        ),
        "train_from": Param(
            default="2013-01-01",
            type="string",
            title="train_from",
            description="Start date for training data",
        ),
        "train_upto": Param(
            default="2016-08-15",
            type="string",
            title="train_upto",
            description="End date for training data",
        ),
    },
)
def weekly_run():
    @task()  # multiple_outputs=True
    def coreload_data(tablename: str, **kwargs: str):
        timestamp = kwargs["params"]["timestamp"]
        print(f"This is task 1 {timestamp}")
        # Simulate some data processing
        df = get_dataframe_from_s3(timestamp, tablename)

        print(df)

        if df.is_empty():
            print(f"No {tablename} data found for the given timestamp.")
            return

        response = load_dataframe_to_db(df, tablename)
        print(f"{tablename} loaded: {response}")
        return

    @task
    def coreload_dependent_data(
        tablename: str, mappers: dict[str, dict[str, int]], renamers: dict[str, str], **kwargs: str
    ):
        timestamp = kwargs["params"]["timestamp"]
        df = get_dataframe_from_s3(timestamp, tablename)
        print(df)
        if df.is_empty():
            print(f"No {tablename} data found for the given timestamp.")
            return

        df = map_names_to_ids(df, mappers, renamers)
        print(df)

        response = load_dataframe_to_db(df, tablename)
        print(f"{tablename} loaded: {response}")
        return

    @task
    def get_map_name_to_id(tablename: str, key: str, value: str):
        return get_datamap_to_ids(tablename, key, value)

    @task
    def create_train_snapshopt(**kwargs: str):
        timestamp = kwargs["params"]["timestamp"]
        train_from = kwargs["params"]["train_from"]
        train_upto = kwargs["params"]["train_upto"]
        return create_train_dataset(timestamp, train_from, train_upto)

    @task(multiple_outputs=True)
    def run_feature_engineering(s3_snapshot_path: str, **kwargs: str):
        timestamp = kwargs["params"]["timestamp"]
        target = "log_units_sold"
        horizon = 7
        s3_paths = feature_engineering(s3_snapshot_path, target, horizon, timestamp)
        return s3_paths

    @task
    def optimize_model_params_experiment(s3_paths: str, **kwargs: str):
        timestamp = kwargs["params"]["timestamp"]
        from src.forecasttools import run_model_experiment

        return run_model_experiment(timestamp)

    @task
    def train_and_register_best_model(**kwargs: str):
        timestamp = kwargs["params"]["timestamp"]

        return run_train_and_register_best_model(timestamp)

    @task
    def predict_demand(**kwargs: str):
        timestamp = kwargs["params"]["timestamp"]
        return run_predict_demand(timestamp)

    @task
    def fulfill_demand(**kwargs: str):
        timestamp = kwargs["params"]["timestamp"]
        return run_fulfillment(timestamp, procurement_window=7)

    @task
    def update_dashboard(**kwargs: str):
        """
        This task is a placeholder for UI snapshots.
        It can be used to visualize the data or results in the Airflow UI.
        """
        print("UI snapshots task executed.")
        return {"status": "success"}

    # Llama a task1 y accede a sus salidas por clave.
    # products_map = get_all_products_map()
    products = get_map_name_to_id.override(task_id="get_products")("products", "p_name", "p_id")
    stores = get_map_name_to_id.override(task_id="get_stores")("stores", "s_name", "s_id")
    workshops = get_map_name_to_id.override(task_id="get_workshops")("workshops", "w_name", "w_id")
    events = get_map_name_to_id.override(task_id="get_events")("events", "e_name", "e_id")

    coreload_data.override(task_id="coreload_products")("products") >> products
    coreload_data.override(task_id="coreload_stores")("stores") >> stores
    coreload_data.override(task_id="coreload_workshops")("workshops") >> workshops
    coreload_data.override(task_id="coreload_events")("events") >> events

    s3_snapshot_path = create_train_snapshopt()

    (
        coreload_dependent_data.override(task_id="coreload_transportlinks")(
            tablename="transportlinks",
            mappers={"p_name": products, "s_name": stores, "w_name": workshops},
            renamers={"p_name": "tl_p_id", "s_name": "tl_s_id", "w_name": "tl_w_id"},
        )
        >> s3_snapshot_path
    )

    (
        coreload_dependent_data.override(task_id="coreload_procurements")(
            tablename="procurements",
            mappers={"p_name": products, "s_name": stores},
            renamers={"p_name": "pc_p_id", "s_name": "pc_s_id"},
        )
        >> s3_snapshot_path
    )

    (
        coreload_dependent_data.override(task_id="coreload_promotions")(
            tablename="promotions",
            mappers={"p_name": products, "s_name": stores},
            renamers={"p_name": "pr_p_id", "s_name": "pr_s_id"},
        )
        >> s3_snapshot_path
    )

    (
        coreload_dependent_data.override(task_id="coreload_eventstores")(
            tablename="eventstores",
            mappers={"s_name": stores, "e_name": events},
            renamers={"s_name": "es_s_id", "e_name": "es_e_id"},
        )
        >> s3_snapshot_path
    )

    (
        coreload_dependent_data.override(task_id="coreload_stocks")(
            tablename="stocks",
            mappers={"p_name": products, "s_name": stores},
            renamers={"p_name": "sk_p_id", "s_name": "sk_s_id"},
        )
        >> s3_snapshot_path
    )
    (
        coreload_dependent_data.override(task_id="coreload_sales")(
            tablename="sales",
            mappers={"p_name": products, "s_name": stores},
            renamers={"p_name": "sa_p_id", "s_name": "sa_s_id"},
        )
        >> s3_snapshot_path
    )

    s3_paths = run_feature_engineering(s3_snapshot_path)
    # end_experiment = optimize_model_params_experiment(s3_paths=s3_paths)
    (
        optimize_model_params_experiment(s3_paths=s3_paths)
        >> train_and_register_best_model()
        >> predict_demand()
        >> fulfill_demand()
        >> update_dashboard()
    )
    return


weekly_run()
