import polars as pl
from ortools.sat.python import cp_model

from src.dataloaders import get_postgres_uri, load_dataframe_to_db


def run_fulfillment(today: str, procurement_window: int) -> None:
    """
    Run the fulfillment model to determine procurement needs based on demand predictions and current stock levels.
    Args:
        today (str): The current date in 'YYYY-MM-DD' format.
        procurement_window (int): The number of days to consider for procurement.
    """
    agg_level = ["p_id", "s_id", "c_rank"]
    # Fetch input data
    df_periods, df_procurements, df_current_stocks, df_transportlinks, df_workshops = (
        get_input_data(today, procurement_window)
    )

    # CP-SAT model setup
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    workshops_ids = df_workshops.unique("w_id").get_column("w_id").to_list()
    workshop_capacities = polars_to_dict(
        df_workshops, key_cols=["w_id", "c_rank"], value_col="capacity"
    )

    # Create variables for new orders. These is the recommended orders based on demand predictions.
    new_orders = polars_to_dict(df_procurements, key_cols=agg_level, value_col="needs_order")
    for (p, s, t), needs_order in new_orders.items():
        if needs_order:
            new_orders[p, s, t] = model.new_int_var(0, 1000, f"new_order_{p}_{s}_{t}")
        else:
            new_orders[p, s, t] = model.new_constant(0, f"no_order_{p}_{s}_{t}")

    package_sizes = polars_to_dict(
        df_transportlinks, key_cols=["p_id", "s_id", "w_id", "c_rank"], value_col="package_size"
    )
    package_costs = polars_to_dict(
        df_transportlinks, key_cols=["p_id", "s_id", "w_id", "c_rank"], value_col="package_cost"
    )
    packages_orders = package_sizes.copy()

    for p, s, w, t in packages_orders.keys():
        packages_orders[p, s, w, t] = model.new_int_var(0, 1000, f"packages_order_{p}_{s}_{w}_{t}")

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

    for p, s, t in new_orders.keys():
        # new orders should be equal to the number of packages ordered times the package size
        model.add(
            new_orders[p, s, t]
            == sum(
                packages_orders.get((p, s, w, t), 0) * package_sizes.get((p, s, w, t), 0)
                for w in workshops_ids
            )
        )

    for w, t in workshop_capacities.keys():
        # The total number of packages ordered at workshop w at time t should not exceed its capacity
        model.add(  # TODO: de hecho tiene mas sentido expresa la capacidad como el numero de paquetes que se pueden enviar
            sum(
                packages_orders.get((p, s, w, t), 0) * package_sizes.get((p, s, w, t), 0)
                for p, s in current_stocks.keys()
            )
            <= workshop_capacities[w, t]
        )

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
    understock_level = 40
    safety_stocks = 20

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
    overstock_penalty = 1
    understock_penalty = 2
    package_cost_penalty = 1

    objective_terms = []
    # objective_terms.extend(
    #     [
    #         package_cost_penalty * package_costs[p, s, w, t] * pk_var
    #         for (p, s, w, t), pk_var in packages_orders.items()
    #     ]
    # )
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
    df_demand_fulfillments = get_demand_fulfillments(
        solver, met_demand, unmet_demand, expected_stocks, df_periods, agg_level
    )
    df_procurement_plans = get_procurement_plans(solver, packages_orders, package_sizes, df_periods)

    # print(df_demand_fulfillments.filter(df_p_id=2, df_s_id=2).sort("df_date"))

    # print(df_procurement_plans.filter(pl_p_id=2, pl_s_id=2).sort("pl_w_id", "pl_date"))

    # Save the output to a DuckDB database
    load_dataframe_to_db(df_demand_fulfillments, "demandfulfillments")
    load_dataframe_to_db(df_procurement_plans, "procurementplans")

    return


def get_demand_fulfillments(
    solver, met_demand, unmet_demand, expected_stocks, df_periods, agg_level
) -> pl.DataFrame:
    return (
        solution_to_df(
            get_solver_solutions(solver, met_demand),
            key_schema=agg_level,
            value_schema="df_met_demand",
        )
        .join(
            solution_to_df(
                get_solver_solutions(solver, unmet_demand),
                key_schema=agg_level,
                value_schema="df_unmet_demand",
            ),
            on=agg_level,
            how="left",
        )
        .join(
            solution_to_df(
                get_solver_solutions(solver, expected_stocks),
                key_schema=agg_level,
                value_schema="df_closing_stocks",
            ),
            on=agg_level,
            how="left",
        )
        .with_columns(df_initial_stocks=pl.col("df_met_demand") + pl.col("df_closing_stocks"))
        .with_columns(pl.all().cast(pl.Int32))
        .join(
            df_periods,
            on="c_rank",
            how="left",
        )
        .drop("c_rank")
        .rename({"c_date": "df_date", "p_id": "df_p_id", "s_id": "df_s_id"})
    )


def get_procurement_plans(solver, packages_orders, package_sizes, df_periods):
    packages_orders = get_solver_solutions(solver, packages_orders)
    for keys, size in package_sizes.items():
        packages_orders[keys] = packages_orders[keys] * size

    return (
        solution_to_df(
            get_solver_solutions(solver, packages_orders),
            key_schema=["pl_p_id", "pl_s_id", "pl_w_id", "c_rank"],
            value_schema="pl_expected_units",
        )
        .with_columns(pl.all().cast(pl.Int32))
        .filter(pl.col("pl_expected_units") > 0)
        .join(
            df_periods,
            on="c_rank",
            how="left",
        )
        .drop("c_rank")
        .rename({"c_date": "pl_date"})
    )


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
        pl.Series(value_schema, list(solution_dict.values()))
    )


def get_solver_solutions(solver, var_dict):
    """
    Extracts the solution from the solver for the given variable dictionary.
    var_dict: dictionary with variable keys.
    Returns a dictionary with the variable keys and their values.
    """
    return {keys: solver.Value(var) for keys, var in var_dict.items()}


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
    from src.queries import (
        query_current_stocks,
        query_periods,
        query_procurements,
        query_transportlinks,
        query_workshops,
    )

    query_periods = query_periods.format(date_from=today, window=procurement_window)
    df_periods = pl.read_database_uri(query=query_periods, uri=get_postgres_uri())
    # print(df_periods)

    query_procurements = query_procurements.format(date_from=today, window=procurement_window)
    df_procurements = pl.read_database_uri(query=query_procurements, uri=get_postgres_uri())
    # print(df_procurements)

    query_current_stocks = query_current_stocks.format(date_from=today)
    df_current_stocks = pl.read_database_uri(query=query_current_stocks, uri=get_postgres_uri())
    # print(df_current_stocks)

    query_transportlinks = query_transportlinks.format(date_from=today, window=procurement_window)
    df_transportlinks = pl.read_database_uri(query=query_transportlinks, uri=get_postgres_uri())
    # print(df_transportlinks)

    query_workshops = query_workshops.format(date_from=today, window=procurement_window)
    df_workshops = pl.read_database_uri(query=query_workshops, uri=get_postgres_uri())
    # print(df_workshops)

    return df_periods, df_procurements, df_current_stocks, df_transportlinks, df_workshops
