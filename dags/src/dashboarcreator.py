import os

import boto3
import plotly.graph_objects as go
import polars as pl

# sys.path.append("..")
from src.dataloaders import get_postgres_uri
from src.queries import (
    query_demand_evolution,
    query_sales_stocks,
    #     query_stores_locations,
    #     query_workshops_locations,
)

BUCKET_NAME = "dashboard"


def save_fig_to_s3(fig, file_path, s3_client):
    """
    Save the plotly figure as an HTML file.
    """
    # file_path = f"balance_{product.replace(' ', '_')}_{store}.html"

    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=file_path,
        Body=fig.to_html(full_html=True, include_plotlyjs="cdn"),
        ContentType="text/html",
    )
    return file_path


def get_html_from_s3(file_path, s3_client):
    """
    Retrieve the HTML file from S3.
    """
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_path)
        return response["Body"].read().decode("utf-8")
    except s3_client.exceptions.NoSuchKey:
        print(f"File {file_path} not found in S3 bucket {BUCKET_NAME}.")
        return None


def plot_demand_balance(df_demand_evol: pl.DataFrame, df_sales_stocks: pl.DataFrame, s3_client):
    """Plot the inventory balance for each product in each store over time.
    The balance is calculated as the difference between expected initial inventory, expected deliveries,
    expected demand fulfilled, and expected demand unfulfilled.

    Args:
        df_demand_evol (pl.DataFrame): DataFrame containing the demand evolution data with the following columns:
            - c_date: date of the demand evolution
            - p_name: name of the product
            - s_name: name of the store
            - init_stocks: expected initial inventory
            - units_sent: expected deliveries
            - met_demand: expected demand fulfilled
            - unmet_demand: expected demand unfulfilled
    """

    products_stores = (
        df_demand_evol.unique(subset=["p_name", "s_name"])
        .select("p_name", "s_name")
        .sort(["s_name", "p_name"])
        .rows()
    )

    for product, store in products_stores:
        df_subset = df_demand_evol.filter(p_name=product, s_name=store)
        df_subset_ss = df_sales_stocks.filter(p_name=product, s_name=store)

        fig = go.Figure()

        # --- DATA TRACES with new color palette ---

        # Inventory (Historical)
        fig.add_trace(
            go.Bar(
                x=df_subset_ss["c_date"],
                y=df_subset_ss["units"],
                name="Inventory",
                marker_color="rgb(252, 186, 3)",
                opacity=1,
                hovertemplate="<b>Date</b>: %{x}<br><b>Inventory</b>: %{y} units<extra></extra>",
            )
        )

        # Initial Inventory (Projected)
        fig.add_trace(
            go.Bar(
                x=df_subset["c_date"],
                y=df_subset["initial_stocks"],
                name="Projected Initial Inventory",
                marker_color="rgb(252, 186, 3)",
                opacity=0.6,
                hovertemplate="<b>Date</b>: %{x}<br><b>Projected Initial Inv.</b>: %{y} units<extra></extra>",
            )
        )

        # Deliveries (Projected)
        fig.add_trace(
            go.Bar(
                x=df_subset["c_date"],
                y=df_subset["expected_units"],
                name="Projected Deliveries",
                marker_color="rgb(3, 148, 252)",
                opacity=0.7,
                hovertemplate="<b>Date</b>: %{x}<br><b>Projected Deliveries</b>: %{y} units<extra></extra>",
                # name='Projected Deliveries', marker_color='rgb(52, 73, 94)', opacity=0.7,
                # hovertemplate='<b>Date</b>: %{x}<br><b>Projected Deliveries</b>: %{y} units<extra></extra>'
            )
        )

        # Sales (Historical)
        fig.add_trace(
            go.Bar(
                x=df_subset_ss["c_date"],
                y=-df_subset_ss["units_sold"],
                name="Sales",
                marker_color="rgb(23, 191, 99)",
                opacity=1,
                hovertemplate="<b>Date</b>: %{x}<br><b>Sales</b>: %{customdata} units<extra></extra>",
                customdata=df_subset_ss["units_sold"],
            )
        )

        # Met Demand (Projected)
        fig.add_trace(
            go.Bar(
                x=df_subset["c_date"],
                y=-df_subset["met_demand"],
                name="Projected Fulfilled Sales",
                marker_color="rgb(23, 191, 99)",
                opacity=0.6,
                hovertemplate="<b>Date</b>: %{x}<br><b>Projected Fulfilled</b>: %{customdata} units<extra></extra>",
                customdata=df_subset["met_demand"],
            )
        )

        # Unmet Demand (Projected)
        fig.add_trace(
            go.Bar(
                x=df_subset["c_date"],
                y=-df_subset["unmet_demand"],
                name="Projected Unfulfilled Sales",
                marker_color="rgb(52, 73, 94)",
                opacity=0.6,
                hovertemplate="<b>Date</b>: %{x}<br><b>Projected Unfulfilled</b>: %{customdata} units<extra></extra>",
                customdata=df_subset["unmet_demand"],
            )
        )

        # --- LAYOUT ENHANCEMENTS ---
        fig.update_traces(marker_line_width=0)

        fig.update_layout(
            # title=f'<b>Inventory & Sales Projection: {product} at {store}</b>',
            barmode="relative",
            paper_bgcolor="rgb(248, 248, 255)",
            plot_bgcolor="rgb(248, 248, 255)",
            xaxis=dict(
                title="Date",
                showgrid=False,
                tickfont=dict(family="Arial, sans-serif", size=12, color="rgb(100,100,100)"),
            ),
            yaxis=dict(
                title="Units",
                showgrid=True,
                gridcolor="rgb(220,220,220)",
                tickfont=dict(family="Arial, sans-serif", size=12, color="rgb(100,100,100)"),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family="Arial, sans-serif", size=10),
            ),
            font=dict(family="Arial, sans-serif", color="rgb(60,60,60)"),
            margin=dict(l=60, r=30, t=80, b=60),
        )
        # save to HTML file to export it later to streamlit
        file_path = f"balance_{product}_{store}.html"
        save_fig_to_s3(fig, file_path, s3_client)

        # fig.write_html(f"balance_{product}_{store}.html")
        # fig.show()


def run_dashboards(timestamp: str):
    """Run the demand balance plot with data from the database."""
    df_demand_evol = pl.read_database_uri(
        query=query_demand_evolution.format(date_from=timestamp), uri=get_postgres_uri()
    )
    df_sales_stocks = pl.read_database_uri(
        query=query_sales_stocks.format(date_from=timestamp, window=21), uri=get_postgres_uri()
    )
    s3_client = boto3.client(
        "s3",
        endpoint_url="http://minio:9000",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"],
    )
    plot_demand_balance(df_demand_evol, df_sales_stocks, s3_client)
