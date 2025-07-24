import sys
from pathlib import Path

import click
import polars as pl
from ortools.sat.python import cp_model

# Agregar el directorio padre (services) al path
current_file = Path(__file__)
services_dir = current_file.parent.parent
sys.path.insert(0, str(services_dir))

try:
    from shared.dboperators import upload_json
    from shared.s3config import get_postgres_uri
except ImportError:
    raise ImportError("shared.s3config module not found. Ensure the path is correct.")


def polars_to_dict(df, key_cols, value_col):
    """
    Convierte un DataFrame de Polars a un diccionario con tuplas como clave.
    key_cols: lista de nombres de columnas para la clave.
    value_col: nombre de la columna para el valor.
    """
    return {tuple(row[col] for col in key_cols): row[value_col] for row in df.to_dicts()}


def empty_dict_from_polars(df, key_cols=None):
    if not key_cols:
        key_cols = df.columns
    return {tuple(row): None for row in zip(*[df[col] for col in key_cols])}


def solution_to_df(
    solution_dict, key_schema: list[tuple[str, type]], value_schema: tuple[str, type]
):
    """
    Convert a solution dictionary to a Polars DataFrame.
    solution_dict: dictionary with keys as tuples and values as integers.
    key_schema: list of tuples defining the schema for the keys.
    value_schema: tuple defining the name and type of the value column.

    Example:
    solution_to_df(met_demand_solution, key_schema=[("product", str), ("location", str), ("period_rank", int)], value_schema=("met_demand", int))
    """
    # get list of tuples keys from solution_dict
    return pl.DataFrame(list(solution_dict.keys()), schema=key_schema, orient="row").with_columns(
        pl.Series(value_schema[0], list(solution_dict.values())).cast(value_schema[1])
    )


def get_solver_solutions(solver, var_dict):
    """
    Extracts the solution from the solver for the given variable dictionary.
    var_dict: dictionary with variable keys.
    Returns a dictionary with the variable keys and their values.
    """
    return {keys: solver.Value(var_dict[keys]) for keys in var_dict.keys()}


def get_input_data_dev(today: str, procurement_window: int):
    """
    Fetches the input data for the fulfillment model.
    Args:
        today (str): The current date in 'YYYY-MM-DD' format.
        procurement_window (int): The number of days to consider for procurement.
    Returns:
        tuple: DataFrames for periods, procurements, and current stocks.
    """
    import duckdb
    from queries import query_current_stocks, query_periods, query_procurements

    with duckdb.connect("../dbcore/data/core.db") as con:
        df_periods = con.execute(
            query_periods.format(date_from=today, window=procurement_window)
        ).pl()
        df_procurements = con.execute(
            query_procurements.format(date_from=today, window=procurement_window)
        ).pl()
        df_current_stocks = con.execute(query_current_stocks.format(date_from=today)).pl()
    return df_periods, df_procurements, df_current_stocks


def get_input_data(today: str, procurement_window: int):
    """
    Fetches the input data for the fulfillment model.
    Args:
        today (str): The current date in 'YYYY-MM-DD' format.
        procurement_window (int): The number of days to consider for procurement.
    Returns:
        tuple: DataFrames for periods, procurements, and current stocks.
    """

    # Execute the query against the PostgreSQL database and get the polars DataFrame
    from queries import query_current_stocks, query_periods, query_procurements

    query_periods = query_periods.format(date_from=today, window=procurement_window)
    df_periods = pl.read_database_uri(query=query_periods, uri=get_postgres_uri())
    print(df_periods)

    query_procurements = query_procurements.format(date_from=today, window=procurement_window)
    df_procurements = pl.read_database_uri(query=query_procurements, uri=get_postgres_uri())
    print(df_procurements)

    query_current_stocks = query_current_stocks.format(date_from=today)
    df_current_stocks = pl.read_database_uri(query=query_current_stocks, uri=get_postgres_uri())
    print(df_current_stocks)

    return df_periods, df_procurements, df_current_stocks


def run_fulfillment(today: str, procurement_window: int) -> None:
    """
    Run the fulfillment model to determine procurement needs based on demand predictions and current stock levels.
    Args:
        today (str): The current date in 'YYYY-MM-DD' format.
        procurement_window (int): The number of days to consider for procurement.
    """

    agg_level = ["p_id", "s_id", "c_rank"]
    # Fetch input data
    df_periods, df_procurements, df_current_stocks = get_input_data(today, procurement_window)

    # CP-SAT model setup
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # Create variables for new orders. These is the recommended orders based on demand predictions.
    new_orders = polars_to_dict(df_procurements, key_cols=agg_level, value_col="needs_order")
    for (p, s, t), needs_order in new_orders.items():
        if needs_order:
            new_orders[p, s, t] = model.new_int_var(0, 1000, f"new_order_{p}_{s}_{t}")
        else:
            new_orders[p, s, t] = model.new_constant(0, f"no_order_{p}_{s}_{t}")

    # Create variables for demand predictions, met demand, unmet demand.
    # These are the expected demand predictions for each product, store, and time period.
    demand_predictions = polars_to_dict(
        df_procurements, key_cols=agg_level, value_col="pred_units_sold"
    )
    met_demand = demand_predictions.copy()
    unmet_demand = demand_predictions.copy()
    for p, s, t in demand_predictions.keys():
        met_demand[p, s, t] = model.new_int_var(0, 1000, f"met_demand_{p}_{s}_{t}")
        unmet_demand[p, s, t] = model.new_int_var(0, 1000, f"unmet_demand_{p}_{s}_{t}")

    current_stocks = polars_to_dict(
        df_current_stocks, key_cols=["p_id", "s_id"], value_col="ending_inventory"
    )
    expected_stocks = empty_dict_from_polars(df_procurements, agg_level)
    overstocks = expected_stocks.copy()
    understocks = expected_stocks.copy()

    for p, s, t in expected_stocks.keys():
        expected_stocks[p, s, t] = model.new_int_var(0, 1000, f"ending_inv_{p}_{s}_{t}")
        overstocks[p, s, t] = model.new_int_var(0, 1000, f"overstock_{p}_{s}_{t}")
        understocks[p, s, t] = model.new_int_var(0, 1000, f"understock_{p}_{s}_{t}")

    # Constraints
    for p, s, t in demand_predictions.keys():
        # met demand is the sum of all orders for product p, location l, and time t
        # model.add(met_demand[p,l,t] == sum(order_vars[p, l, t, o] for o in vars_tuples.o))

        # met demand should not exceed forecast demand
        # model.add(met_demand[p,s,t] <= demand_predictions[p,s,t])

        # unmet demand is the difference between demand and met demand
        model.add(unmet_demand[p, s, t] + met_demand[p, s, t] == demand_predictions[p, s, t])

        # # new orders should be at least the unmet demand
        # model.add(new_orders[p,s,t] >= unmet_demand[p,s,t])

    max_capacity = 120
    overstock_level = 80
    understock_level = 70
    safety_stocks = 30

    for p, s, t in expected_stocks.keys():
        if t == 1:  # if p, s has not recorded current stock, we use the current stock as 0
            initial_expected_stock = current_stocks.get((p, s), 0) + new_orders[p, s, 1]
            model.add(expected_stocks[p, s, 1] == initial_expected_stock - met_demand[p, s, 1])
        else:  # t > 1
            initial_expected_stock = expected_stocks[p, s, t - 1] + new_orders[p, s, t]
            model.add(expected_stocks[p, s, t] == initial_expected_stock - met_demand[p, s, t])

        # inventory should not exceed max capacity at the beginning of the period
        model.add(initial_expected_stock <= max_capacity)

        # inventory should not fall below safety stock levels at the end of the period
        model.add(expected_stocks[p, s, t] >= safety_stocks)

        # add understock_level and overstock_level penalization to ending stocks
        model.add_max_equality(
            overstocks[p, s, t], [0, initial_expected_stock - overstock_level]
        )  # overstock >= initial_expected_stock - overstock_level
        model.add_max_equality(
            understocks[p, s, t], [0, understock_level - initial_expected_stock]
        )  # understock >= understock_level - initial_expected_stock

    # Build the objective function with penalties for unmet demand, overstock, and understock
    unmet_penalty = 100
    overstock_penalty = 10
    understock_penalty = 12

    objective_terms = []
    objective_terms.extend([unmet_penalty * ud_var for ud_var in unmet_demand.values()])
    objective_terms.extend(
        [overstock_penalty * overstock_var for overstock_var in overstocks.values()]
    )
    objective_terms.extend(
        [understock_penalty * understock_var for understock_var in understocks.values()]
    )

    # This is the objective function to minimize the total cost
    # of unmet demand, overstock, and understock
    model.minimize(sum(objective_terms))

    solver.parameters.num_search_workers = 8
    solver.parameters.max_time_in_seconds = 120
    callback = cp_model.ObjectiveSolutionPrinter()
    or_status = solver.SolveWithSolutionCallback(model, callback)
    status = solver.StatusName(or_status)

    if status in ["OPTIMAL", "FEASIBLE"]:
        print(f"Solution: Total cost = {solver.ObjectiveValue()}")
    else:
        print("A solution could not be found, check the problem specification")

    # Get the solutions
    met_demand_solution = get_solver_solutions(solver, met_demand)
    unmet_demand_solution = get_solver_solutions(solver, unmet_demand)
    expected_stocks_solution = get_solver_solutions(solver, expected_stocks)
    new_orders_solution = get_solver_solutions(solver, new_orders)

    agg_level_schema = [("p_id", pl.Int32), ("s_id", pl.Int32), ("c_rank", pl.Int32)]

    df_met_demand = solution_to_df(
        met_demand_solution, key_schema=agg_level_schema, value_schema=("met_demand", pl.Int32)
    )
    df_unmet_demand = solution_to_df(
        unmet_demand_solution, key_schema=agg_level_schema, value_schema=("unmet_demand", pl.Int32)
    )

    df_expected_stocks = solution_to_df(
        expected_stocks_solution,
        key_schema=agg_level_schema,
        value_schema=("ending_inventory", pl.Int32),
    )

    df_new_orders = solution_to_df(
        new_orders_solution,
        key_schema=agg_level_schema,
        value_schema=("recommended_orders", pl.Int32),
    )

    return (
        (
            pl.concat([df_current_stocks, df_expected_stocks], how="vertical")
            .with_columns(
                pl.col("ending_inventory")
                .shift(1)
                .over("p_id", "s_id", order_by="c_rank")
                .alias("initial_expected_stock")
            )
            .filter(pl.col("c_rank") > 0)
            .join(df_met_demand, on=agg_level, how="left")
            .join(df_unmet_demand, on=agg_level, how="left")
            .join(df_new_orders, on=agg_level, how="left")
            .join(df_periods, on="c_rank", how="left")
            .drop("c_rank")
        )
        .rename({"c_date": "date"})
        .select(pl.all().name.prefix("pcpl_"))
    )


def save_output(df_output, output_path: str = "../dbcore/data/fulfillment_output.parquet"):
    """
    Save the output DataFrame to a DuckDB database.
    Args:
        df_output (pl.DataFrame): The output DataFrame to save.
        output_path (str): The path to the DuckDB database file.
    """
    df_output.write_parquet(output_path, compression="snappy")
    # with duckdb.connect(output_path) as con:
    #     con.execute("CREATE TABLE IF NOT EXISTS procurement_plans AS SELECT * FROM df_output")
    #     con.execute("INSERT INTO procurement_plans SELECT * FROM df_output")


@click.command()
@click.option("--today", default="2016-08-15", help="Current date in 'YYYY-MM-DD' format.")
@click.option("--procurement_window", default=7, help="Number of days to consider for procurement.")
def main(today, procurement_window):
    df_output = run_fulfillment(today, procurement_window)
    save_output(df_output)


if __name__ == "__main__":
    main()
