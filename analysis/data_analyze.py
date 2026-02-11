import plotly.express as px
import pandas as pd
import plotly.io as pio

df = pd.read_csv("test.csv")

type_mapping = {
    "HAE": "Geometric",
    "MSL": "Physical",
    "Orthgonal Height": "Physical",
    "AGL": "Geometric",
    "Q-codes": "Mixed"
}
df["Height System"] = df["Height System"].apply(lambda x: f"{x}: {type_mapping.get(x, 'Unknown')}")

# Define metric order to ensure consistent plotting
metric_order = ["Conceptual simplicity", "Reference stability", "Measurement practicality",
                "Measurement accuracy", "Practical applications"]
df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)
df = df.sort_values(["Height System", "Metric"])

fig = px.line_polar(df, r="Score",
                    theta="Metric",
                    color="Height System",
                    line_close=True,
                    color_discrete_sequence=["#1E88E5", "#FF6F00", "#43A047", "#E53935", "#8E24AA"],
                    template="ggplot2")

fig.update_traces(mode='lines+markers', marker=dict(size=8), fill='toself', opacity=0.5)

fig.update_polars(angularaxis_showgrid=True,
                  angularaxis_gridcolor="black",
                  angularaxis_gridwidth=1,
                  angularaxis_griddash="dash",
                  radialaxis_gridwidth=1,
                  radialaxis_gridcolor="black",
                  radialaxis_griddash="dash",
                  radialaxis_showticklabels=False,
                  radialaxis_tickmode='linear',
                  radialaxis_tick0=0,
                  radialaxis_dtick=1
                  )

fig.update_layout(
    font=dict(family="Times New Roman", size=15),
    margin=dict(l=20, r=20, t=20, b=20),
    polar=dict(
        domain=dict(x=[0.1, 0.9], y=[0.1, 0.9]),
        angularaxis=dict(
            tickangle=0
        )
    )
)
fig.show()

# Export figure at 600 DPI (scale = 600/72 ≈ 8.33)
fig.write_image("height_system_comparison.png", width=800, height=800, scale=8.33)
