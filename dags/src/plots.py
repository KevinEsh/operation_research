import pandas as pd
import plotly.graph_objects as go


def create_map_figure():
    # This data should eventually come from a dynamic source
    workshops_data = {
        "lat": [-0.2983, -0.2033, -0.179, -0.2306, -0.248],
        "lon": [-78.533, -78.492, -78.46, -78.51, -78.49],
        "name": ["Bakery 1", "Bakery 2", "Bakery 3", "Pasta Factory 1", "Egg Factory 1"],
        "capacity": [150, 200, 215, 250, 300],
    }
    stores_data = {
        "lat": [-0.223, -0.21, -0.23, -0.2, -0.25],
        "lon": [-78.5, -78.48, -78.52, -78.47, -78.495],
        "name": ["Market 1", "Market 2", "Market 3", "Market 4", "Market 5"],
    }
    df_workshops = pd.DataFrame(workshops_data)
    df_stores = pd.DataFrame(stores_data)

    fig = go.Figure()

    # Add workshops
    fig.add_trace(
        go.Scattermapbox(
            lat=df_workshops["lat"],
            lon=df_workshops["lon"],
            mode="markers",
            marker=go.scattermapbox.Marker(size=14, color="orange", symbol="factory"),
            text=[
                f"Workshop: {name}<br>Capacity: {cap} units"
                for name, cap in zip(df_workshops["name"], df_workshops["capacity"])
            ],
            hoverinfo="text",
            name="Workshops",
        )
    )

    # Add stores
    fig.add_trace(
        go.Scattermapbox(
            lat=df_stores["lat"],
            lon=df_stores["lon"],
            mode="markers",
            marker=go.scattermapbox.Marker(size=12, color="blue", symbol="shop"),
            text=[f"Store: {name}" for name in df_stores["name"]],
            hoverinfo="text",
            name="Stores",
        )
    )

    fig.update_layout(
        title="Store and Workshop Locations",
        mapbox_style="carto-positron",
        mapbox_zoom=11,
        mapbox_center={"lat": -0.22, "lon": -78.5},
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig
