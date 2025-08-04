import os
import re
import sys

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

sys.path.append("dags")

from src.queries import (
    query_demand_evolution,
    query_sales_stocks,
    query_stores_locations,
    query_workshops_locations,
)

query_shipments = """
select
	pl_date as "Date",
	p_name as "Product",
	s_name as "Store",
	w_name as "Factory",
	pl_expected_units as "Units"
from procurementplans
left join products on p_id = pl_p_id
left join stores on s_id = pl_s_id
left join workshops on w_id = pl_w_id
where p_name = '{product}' and s_name = '{store}'
order by 1
"""


@st.cache_data
def get_shipment_data(product: str, store: str):
    """
    Fetch shipment data for a specific product and store.
    """
    import polars as pl

    query = query_shipments.format(product=product, store=store)
    return (
        pl.read_database_uri(query=query, uri=get_postgres_uri("dbcore"))
        .with_columns(Status=pl.lit("To Order"))
        .to_pandas()
    )


st.set_page_config(layout="wide")


@st.cache_data(max_entries=1)
def plot_map():
    """
    Plot a visually stunning map with stores and workshops locations.
    """
    print(get_postgres_uri())
    uri = "postgresql://admin:password@localhost:5432/dbcore"
    df_store = pl.read_database_uri(query=query_stores_locations, uri=uri)
    df_workshop = pl.read_database_uri(query=query_workshops_locations, uri=uri)

    fig = go.Figure()

    # --- STORES TRACE ---
    fig.add_trace(
        go.Scattermap(
            name="Stores",
            mode="markers+text",
            lat=df_store.get_column("s_latitude").to_list(),
            lon=df_store.get_column("s_longitude").to_list(),
            text=df_store.get_column("s_name").to_list(),
            marker=dict(
                symbol=["commercial"] * df_store.height,
                size=12,
                color="rgb(91, 192, 222)",  # A calm, professional blue
            ),
            textfont=dict(
                # family='Arial, sans-serif',
                size=10,
                color="rgb(0, 0, 0)",
            ),
            textposition="top center",
            hovertemplate="<b>Store</b>: %{text}<extra></extra>",
        )
    )

    # --- WORKSHOPS TRACE ---
    fig.add_trace(
        go.Scattermap(
            name="Workshops",
            mode="markers+text",
            lat=df_workshop.get_column("w_latitude"),
            lon=df_workshop.get_column("w_longitude"),
            text=df_workshop.get_column("w_name"),
            customdata=df_workshop.get_column("w_capacity"),
            marker=dict(
                symbol=["industry"] * df_workshop.height,
                size=12,
                color="rgb(240, 173, 78)",  # A warm, industrial orange
            ),
            textfont=dict(
                # family='Arial, sans-serif',
                size=10,
                color="rgb(0, 0, 0)",
            ),
            textposition="top center",
            hovertemplate="<b>Workshop</b>: %{text}<br><b>Capacity</b>: %{customdata} units<extra></extra>",
        )
    )

    # --- LAYOUT ENHANCEMENTS ---
    fig.update_layout(
        # title=dict(
        #     text='<b>Operational Footprint: Stores & Workshops</b>',
        #     y=0.95,
        #     x=0.5,
        #     xanchor='center',
        #     yanchor='top',
        #     font=dict(size=20, family='Arial, sans-serif', color='rgb(60,60,60)')
        # ),
        margin=dict(l=0, r=0, t=0, b=0),
        # height=400,
        # width=1400,
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="center",
            x=0.5,
            font=dict(family="Arial, sans-serif", size=12, color="rgb(60, 60, 60)"),
        ),
        map=dict(
            style="carto-positron",  # A clean, modern map style
            center=dict(lat=-0.210799, lon=-78.481357),
            pitch=0,
            bearing=-80,  # Standard north-up orientation
            zoom=11,
        ),
        paper_bgcolor="rgb(255, 255, 255)",
        plot_bgcolor="rgb(255, 255, 255)",
    )
    # fig.write_html("map_stores_workshops.html")

    return fig


def get_postgres_uri(db_name: str = "dbcore") -> str:
    """Get the PostgreSQL URI for the specified database."""
    return f"postgresql://admin:password@localhost:5432/{db_name}"


def get_s3_client():
    """Create and return an S3 client using the credentials from environment variables."""
    import boto3

    return boto3.client(
        "s3",
        aws_access_key_id="admin",
        aws_secret_access_key="password",
        endpoint_url="http://localhost:10000",  # "http://minio:9000",
        region_name="us-east-1",
    )


# --- Data Loading and Filtering ---
@st.cache_data
def get_available_files():
    import polars as pl

    products = (
        pl.read_database_uri(
            query="SELECT p_name FROM products ORDER BY p_name", uri=get_postgres_uri("dbcore")
        )
        .get_column("p_name")
        .to_list()
    )
    stores = (
        pl.read_database_uri(
            query="SELECT s_name FROM stores ORDER BY s_name", uri=get_postgres_uri("dbcore")
        )
        .get_column("s_name")
        .to_list()
    )

    return sorted(products), sorted(stores)


products, stores = get_available_files()


# --- Sidebar ---
with st.sidebar:
    st.title("📦 Inventory Hub")
    st.write("---")
    st.header("Filters")

    selected_product = st.selectbox("Select Product", products)
    selected_store = st.selectbox("Select Store", stores)

    st.write("---")
    st.info(
        "This dashboard provides insights into how many packages need to be sent from each factory to each store, along with the current stock levels and shipment status."
    )

# --- Custom CSS ---
st.markdown(
    """
<style>
    /* General Styles */
    .stApp {
        background-color: #fafafa;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1e3a5f;
    }

    /* Metrics Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        font-size: 0.9rem;
        color: #5a6a7a;
        margin-bottom: 5px;
    }
    .metric-card p {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1e3a5f;
    }
    .metric-icon {
        font-size: 2rem;
        color: #1e3a5f;
        margin-bottom: 5px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff;
    }

    /* Data Table */
    .stDataFrame, .stDataFrame div[data-testid="stHorizontalBlock"] {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

</style>
""",
    unsafe_allow_html=True,
)

# --- KPIs ---
total_sales = 12500

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">📦</div>
        <h3>Total Packages Sent</h3>
        <p>{total_sales}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">🏭</div>
        <h3>Number of Factories</h3>
        <p>6</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">🏪</div>
        <h3>Number of Stores</h3>
        <p>6</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">🍎</div>
        <h3>Number of Products</h3>
        <p>7</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">🚚</div>
        <h3>Shipments In Transit</h3>
        <p>{total_sales}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("<br>", unsafe_allow_html=True)

st.plotly_chart(plot_map(), use_container_width=True)

# --- Data Table ---

# Mock data for table
# shipment_data = {
#     "Warehouse": ["WH-NYC-1", "WH-NYC-1", "WH-LA-1", "WH-LA-2", "WH-CHI-1", "WH-CHI-1"],
#     "Store": ["Store-LA-1", "Store-LA-2", "Store-LA-1", "Store-LA-3", "Store-CHI-1", "Store-CHI-2"],
#     "Product": ["Donut", "Croissant", "Donut", "Egg Pack 18", "Pasta Macaroni", "Pasta Spaghetti"],
#     "Packages Sent": [120, 80, 150, 200, 90, 110],
#     "Status": ["Delivered", "In Transit", "Delivered", "Delivered", "In Transit", "Delivered"],
# }

df_shipments = get_shipment_data(selected_product, selected_store)

th_props = [
    ("font-size", "10px"),
    ("text-align", "center"),
    ("font-weight", "bold"),
    ("color", "black"),
    ("background-color", "white"),
    ("border", "1px solid #dddddd"),
]

td_props = [
    ("font-size", "8px"),
    ("color", "black"),
    ("background-color", "white"),
    ("text-align", "center"),
    ("border", "1px solid #dddddd"),
]

styles = [
    dict(selector="th", props=th_props),
    dict(selector="td", props=td_props),
]


def highlight_status(s):
    return [
        "background-color: #F0AD4E; color: black"
        if v == "In Transit"
        else "background-color: #5CB85C; color: black"
        for v in s
    ]


def highlight(s):
    return ["background-color: white; color: black"] * len(s)


st.dataframe(
    df_shipments.style.apply(highlight, subset=df_shipments.columns).apply(
        highlight_status, subset=["Status"]
    )
)
