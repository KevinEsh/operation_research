import streamlit as st
import pandas as pd
import os
import re

st.set_page_config(layout="wide")

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

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-card h3 {
        font-size: 1.1rem;
        color: #5a6a7a;
        margin-bottom: 5px;
    }
    .metric-card p {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)

# --- KPIs ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown('<div class="metric-card"><h3>📦 Total Packages Sent</h3><p>12,500</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>🏭 Number of Warehouses</h3><p>3</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>🏪 Number of Stores</h3><p>5</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><h3>🍎 Number of Products</h3><p>{len(products)}</p></div>', unsafe_allow_html=True)
with col5:
    st.markdown('<div class="metric-card"><h3>🚚 Avg. Shipment Size</h3><p>125</p></div>', unsafe_allow_html=True)

# --- Map ---
# Mock data for map
map_data = pd.DataFrame({
    'lat': [40.7128, 40.7580, 34.0522, 34.0119, 33.9416, 41.8781, 41.9028, 41.8919],
    'lon': [-74.0060, -73.9855, -118.2437, -118.4944, -118.4085, -87.6298, -87.6558, -87.6074],
    'type': ['Warehouse', 'Warehouse', 'Warehouse', 'Store', 'Store', 'Store', 'Store', 'Store']
})

st.map(map_data, color="#0062ff", height=300)

# --- Data Table ---

# Mock data for table
shipment_data = {
    'Warehouse': ['WH-NYC-1', 'WH-NYC-1', 'WH-LA-1', 'WH-LA-2', 'WH-CHI-1', 'WH-CHI-1'],
    'Store': ['Store-LA-1', 'Store-LA-2', 'Store-LA-1', 'Store-LA-3', 'Store-CHI-1', 'Store-CHI-2'],
    'Product': ['Donut', 'Croissant', 'Donut', 'Egg Pack 18', 'Pasta Macaroni', 'Pasta Spaghetti'],
    'Packages Sent': [120, 80, 150, 200, 90, 110],
    'Status': ['Delivered', 'In Transit', 'Delivered', 'Delivered', 'In Transit', 'Delivered']
}
df_shipments = pd.DataFrame(shipment_data)

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
df2=df_shipments.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)

st.table(df2)#, use_container_width=True)
