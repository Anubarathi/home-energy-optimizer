# app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Home Energy Usage Optimizer", layout="wide")
st.title(" AI-Powered Home Energy Usage Optimizer")
st.markdown("Predict your home electricity usage and see potential savings!")

# ---------------------------
# User Data Input
# ---------------------------
st.sidebar.header("Upload Your Past Energy Data")
st.sidebar.markdown("""
Upload a CSV file with at least two columns:  
- `timestamp` (YYYY-MM-DD HH:MM:SS)  
- `usage_kWh` (energy consumed in kWh)  

OR leave empty to use demo data.
""")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------------------
# Generate Demo Data if no upload
# ---------------------------
@st.cache_data
def generate_demo_data():
    hours = 24 * 7  # 7 days
    base_usage = 1.5
    np.random.seed(42)
    timestamps = [datetime.now() - timedelta(hours=x) for x in reversed(range(hours))]
    
    usage = []
    for i in range(hours):
        hour = timestamps[i].hour
        val = base_usage
        if 17 <= hour <= 22:
            val += 2.5
        elif 6 <= hour <= 9:
            val += 1.2
        val += np.random.normal(0, 0.3)
        usage.append(max(val, 0.1))
    return pd.DataFrame({"timestamp": timestamps, "usage_kWh": usage})

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        st.success("✅ User data loaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        data = generate_demo_data()
        st.info("Using demo data instead.")
else:
    data = generate_demo_data()
    st.info("No file uploaded. Using demo data.")

# ---------------------------
# Prepare Features for Model
# ---------------------------
def prepare_features(df):
    df = df.copy()
    df = df.sort_values("timestamp")
    for lag in range(1, 4):
        df[f"lag_{lag}"] = df["usage_kWh"].shift(lag)
    df.dropna(inplace=True)
    return df

df_features = prepare_features(data)

# ---------------------------
# Train Model
# ---------------------------
X = df_features[[f"lag_{i}" for i in range(1,4)]].values
y = df_features["usage_kWh"].values
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------------------
# Predict Next Hour
# ---------------------------
last_3_hours = data["usage_kWh"].tail(3).values.reshape(1, -1)
pred_next = model.predict(last_3_hours)[0]

# ---------------------------
# Sidebar: Appliance Reduction Sliders
# ---------------------------
st.sidebar.header("Customize Appliance Usage (%)")
lighting = st.sidebar.slider("💡 Lighting reduction", 0, 100, 0, step=5)
heating = st.sidebar.slider("🔥 Heating/Cooling reduction", 0, 100, 0, step=5)
appliances = st.sidebar.slider("🔌 Appliances reduction", 0, 100, 0, step=5)

reduction_factor = 0.3 * lighting/100 + 0.4 * heating/100 + 0.3 * appliances/100
pred_next_adjusted = pred_next * (1 - reduction_factor)
savings = pred_next - pred_next_adjusted

# ---------------------------
# Plot: Historical Usage + Predictions
# ---------------------------
fig = go.Figure()

# Past data
fig.add_trace(go.Scatter(
    x=data["timestamp"],
    y=data["usage_kWh"],
    mode="lines+markers",
    name="Historical Usage",
    line=dict(color="royalblue", width=3),
    marker=dict(size=5)
))

# Predicted next hour
fig.add_trace(go.Scatter(
    x=[data["timestamp"].max() + timedelta(hours=1)],
    y=[pred_next],
    mode="markers",
    marker=dict(color="red", size=15, symbol="diamond"),
    name="Predicted Usage (Next Hour)"
))

# Adjusted prediction
fig.add_trace(go.Scatter(
    x=[data["timestamp"].max() + timedelta(hours=1)],
    y=[pred_next_adjusted],
    mode="markers",
    marker=dict(color="green", size=20, symbol="star"),
    name="Predicted Usage (After Reduction)"
))

# ---------------------------
# Layout: Chart + Summary
# ---------------------------
col1, col2 = st.columns([3, 1])

with col1:
    fig.update_layout(
        title="Home Energy Usage (kWh)",
        xaxis_title="Time",
        yaxis_title="Energy Consumption (kWh)",
        legend=dict(y=0.95, x=0.01),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="#f9f9f9",
        font=dict(family="Arial, sans-serif", size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🔍 Savings Summary")
    st.markdown(f"""
    - **Without reduction:** {pred_next:.2f} kWh  
    - **With reduction:** {pred_next_adjusted:.2f} kWh  
    - **Energy saved:** {savings:.2f} kWh  
    - **Estimated cost saved:** ${savings * 0.12:.2f} (at $0.12/kWh)
    """)
    st.write("")  # whitespace
    st.markdown("""
    **Tip:** Adjust sliders to see how reducing lighting, heating, or appliances affects your next hour's usage.
    """)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Python, Streamlit & AI to help save energy!")
