import streamlit as st
import os
import re
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Inventory Intelligence Hub",
    page_icon="📦",
    layout="wide"
)

# --- Custom CSS for a beautiful UI ---
st.markdown("""
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
""", unsafe_allow_html=True)


# --- Data Loading and Filtering ---
@st.cache_data
def get_available_files():
    files = os.listdir("/home/sade/Documents/repos/operation_research/dags/src/")
    pattern = re.compile(r"inventory_balance_(.+)_(Market \d+)\.html")
    
    products = set()
    stores = set()

    for f in files:
        match = pattern.match(f)
        if match:
            products.add(match.group(1).replace("_", " "))
            stores.add(match.group(2))
            
    return sorted(list(products)), sorted(list(stores))

products, stores = get_available_files()


# --- Sidebar ---
with st.sidebar:
    st.title("📦 Inventory Hub")
    st.write("---")
    st.header("Filters")
    
    selected_product = st.selectbox("Select Product", products)
    selected_store = st.selectbox("Select Store", stores)
    
    st.write("---")
    st.info("This dashboard provides insights into inventory levels, sales, and demand projections.")

# --- Main Dashboard ---
# st.title(f"📊 Dashboard for {selected_product} at {selected_store}")
# st.write("An overview of key metrics and inventory balance projections.")

# --- Key Metrics ---
col1, col2, col3, col4, col5 = st.columns(5, gap="small", )

# Placeholder data for metrics
total_sales = 1280
current_stock = 3450
unmet_demand = 150
deliveries = 800

with col1:
    # st.metric(label="Last Week Sales", value=total_sales, delta=None, help=None)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📈</div>
        <h3>Last Week Sales</h3>
        <p>{total_sales}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📦</div>
        <h3>Current Stock</h3>
        <p>{current_stock}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📉</div>
        <h3>Something</h3>
        <p>{unmet_demand}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🚚</div>
        <h3>Deliveries Tomorrow</h3>
        <p>{deliveries}</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🚚</div>
        <h3>Deliveries Tomorrow</h3>
        <p>{deliveries}</p>
    </div>
    """, unsafe_allow_html=True)

# st.write("<br>", unsafe_allow_html=True)

# --- Analysis Section ---
col1_chart, col2_pie = st.columns([4, 1])

with col1_chart:
    file_path = f"/home/sade/Documents/repos/operation_research/dags/src/inventory_balance_{selected_product.replace(' ', '_')}_{selected_store}.html"

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            html_content = file.read()
        st.components.v1.html(html_content, height=500, scrolling=True)
    else:
        st.warning(f"No data available for {selected_product} at {selected_store}.")
        st.info("Please select another combination of product and store.")

with col2_pie:
    st.header("Sales Breakdown")
    
    # Pie chart data (using placeholder metrics)
    labels = ['Fulfilled Sales', 'Unfulfilled Sales']
    values = [total_sales, unmet_demand]
    colors = ["rgb(23, 191, 99)", "rgb(52, 73, 94)"]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=colors, opacity=0.6),])
    fig.update_layout(
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(family='Arial, sans-serif', color='rgb(60,60,60)'),),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        font=dict(family='Arial, sans-serif', color='rgb(60,60,60)'),
    )
    st.plotly_chart(fig, use_container_width=True)


data = {
    'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07']),
    'Initial Stock': [200, 180, 210, 190, 220, 200, 230],
    'Deliveries': [50, 60, 50, 60, 50, 60, 50],
    'Predicted Sales': [70, 50, 70, 50, 70, 50, 70],
    'Fulfilled Sales': [70, 50, 70, 50, 70, 50, 70],
    'Unfulfilled Sales': [0, 0, 0, 0, 0, 0, 0],
    'Final Stock': [180, 190, 190, 200, 200, 210, 210]
}
df_table = pd.DataFrame(data)
# st.dataframe(df_table, use_container_width=True)

# import numpy as np
# outputdframe = pd.DataFrame(np.array([["CS", "University", "KR", 7032], ["IE", "Bangalore", "Bengaluru", 7861], ["CS", "Bangalore", "Bengaluru", 11036]]), columns=['Branch', 'College', 'Location', 'Cutoff'])
# style
th_props = [
  ('font-size', '10px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#3c3c3c'),
  ('background-color', '#ffffff')
  ]
                               
td_props = [
  ('font-size', '8px'),
  ('color', '#3c3c3c'),
  ('background-color', '#ffffff')
  ]
                                 
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]

# table
df2=df_table.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
st.table(df2)