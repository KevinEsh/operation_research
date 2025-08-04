import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

POSTGRES_USER = os.environ.get("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "password")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "admin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "password")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://localhost:10000")  # "http://minio:9000"

st.set_page_config(page_title="Inventory Intelligence Hub", page_icon="📦", layout="wide")

query_fulfillment = """
with df as (
	select *
	from demandfulfillments
),
pl as (
	select 
		pl_p_id,
		pl_s_id,
		pl_date,
		sum(pl_expected_units) as pl_expected_units
	from procurementplans
	group by 1, 2, 3
)
select
	df_date as "Date",
	p_name as "Product",
	s_name as "Store",
	df_initial_stocks as "Initial Stocks",
	df_closing_stocks as "Closing Stocks",
	df_met_demand as "Fulfilled Sales",
	df_unmet_demand as "Unfulfilled Sales",
	pl_expected_units as "Units To Deliver"
from df
join pl on pl_p_id =df_p_id and pl_s_id=df_s_id and pl_date = df_date
left join products on p_id = df_p_id
left join stores on s_id = df_s_id
where p_name = '{product}'
    and s_name = '{store}'
"""

# --- Custom CSS for a beautiful UI ---
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


@st.cache_data
def get_fulfillment_data(product: str, store: str):
    import polars as pl

    query = query_fulfillment.format(product=product, store=store)
    df = pl.read_database_uri(query=query, uri=get_postgres_uri("dbcore"))
    return df.to_pandas()


def get_postgres_uri(db_name: str = "dbcore") -> str:
    """Get the PostgreSQL URI for the specified database."""
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{db_name}"


def get_s3_client():
    """Create and return an S3 client using the credentials from environment variables."""
    import boto3

    return boto3.client(
        "s3",
        aws_access_key_id="admin",
        aws_secret_access_key="password",
        endpoint_url=S3_ENDPOINT_URL,  # "http://minio:9000",
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


@st.cache_data
def get_html_content_from_s3(product: str, store: str):
    file_path = f"balance_{product}_{store}.html"

    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket="dashboard", Key=file_path)
        html_content = response["Body"].read().decode("utf-8")
        return html_content
    except s3_client.exceptions.NoSuchKey:
        st.warning(f"No data available for {product} at {store}.")
        return None


# --- Sidebar ---
with st.sidebar:
    st.title("📦 Inventory Hub")
    st.write("---")
    st.header("Filters")

    selected_product = st.selectbox("Select Product", products)
    selected_store = st.selectbox("Select Store", stores)

    st.write("---")
    st.info(
        "This dashboard provides insights into inventory levels, sales, and demand projections."
    )

# --- Main Dashboard ---
# st.title(f"📊 Dashboard for {selected_product} at {selected_store}")
# st.write("An overview of key metrics and inventory balance projections.")

# --- Key Metrics ---
col1, col2, col3, col4, col5 = st.columns(
    5,
    gap="small",
)

# Placeholder data for metrics
total_sales = 1280
current_stock = 3450
unmet_demand = 150
deliveries = 800

with col1:
    # st.metric(label="Last Week Sales", value=total_sales, delta=None, help=None)
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">📈</div>
        <h3>Last Week Sales</h3>
        <p>{total_sales}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">📦</div>
        <h3>Current Stock</h3>
        <p>30</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">📉</div>
        <h3>Something</h3>
        <p>{unmet_demand}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">🚚</div>
        <h3>Deliveries Tomorrow</h3>
        <p>{deliveries}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-icon">🚚</div>
        <h3>Deliveries Tomorrow</h3>
        <p>{deliveries}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("<br>", unsafe_allow_html=True)

df_table = get_fulfillment_data(selected_product, selected_store)

# --- Analysis Section ---
col1_chart, col2_pie = st.columns([4, 1])

with col1_chart:
    if html_content := get_html_content_from_s3(selected_product, selected_store):
        st.components.v1.html(html_content, height=500, scrolling=True)
    else:
        st.warning(f"No data available for {selected_product} at {selected_store}.")
        st.info("Please select another combination of product and store.")

with col2_pie:
    st.header("Sales Breakdown")
    met_demand = df_table["Fulfilled Sales"].sum()
    unmet_demand = df_table["Unfulfilled Sales"].sum()

    # Pie chart data (using placeholder metrics)
    labels = ["Fulfilled Sales", "Unfulfilled Sales"]
    values = [met_demand, unmet_demand]
    colors = ["rgb(23, 191, 99)", "rgb(52, 73, 94)"]

    fig = go.Figure(
        data=[
            go.Pie(labels=labels, values=values, hole=0.4, marker_colors=colors, opacity=0.6),
        ]
    )
    fig.update_layout(
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(family="Arial, sans-serif", color="rgb(60,60,60)"),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgb(248, 248, 255)",
        plot_bgcolor="rgb(248, 248, 255)",
        font=dict(family="Arial, sans-serif", color="rgb(60,60,60)"),
    )
    st.plotly_chart(fig, use_container_width=True)


# st.dataframe(df_table, use_container_width=True)

# import numpy as np
# outputdframe = pd.DataFrame(np.array([["CS", "University", "KR", 7032], ["IE", "Bangalore", "Bengaluru", 7861], ["CS", "Bangalore", "Bengaluru", 11036]]), columns=['Branch', 'College', 'Location', 'Cutoff'])
# style
th_props = [
    ("font-size", "10px"),
    ("text-align", "center"),
    ("font-weight", "bold"),
    ("color", "#3c3c3c"),
    ("background-color", "#ffffff"),
]

td_props = [("font-size", "8px"), ("color", "#3c3c3c"), ("background-color", "#ffffff")]

styles = [dict(selector="th", props=th_props), dict(selector="td", props=td_props)]

# table
df2 = df_table.style.set_properties(**{"text-align": "left"}).set_table_styles(styles)
st.table(df2)
