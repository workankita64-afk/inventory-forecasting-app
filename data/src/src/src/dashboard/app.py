# dashboard/app.py
"""
Streamlit dashboard to load forecasts & recommendations from Drive (CSV) or run forecasting inline.
Designed for Streamlit Cloud / local Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src.demand_forecasting import train_and_forecast_per_sku
from src.inventory_optimization import compute_stats_from_history, compute_reorder_point, suggest_order_qty

st.set_page_config(page_title="Inventory Forecast Dashboard", layout="wide")
st.title("Inventory Forecast & Reorder Recommendations")

st.sidebar.header("Settings")
sku = st.sidebar.text_input("Product SKU", value="SKU-123")
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 90, 28)
service_level = st.sidebar.slider("Service level", 80, 99, 95) / 100.0
lead_time = st.sidebar.number_input("Lead time (days)", min_value=1, max_value=60, value=7)

st.markdown("### Instructions")
st.markdown("Mount your Google Drive in Colab and save `sales_cleaned.csv` to `MyDrive/inventory/` or upload a small sample here.")

uploaded = st.file_uploader("Upload sales CSV (optional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['Date'])
else:
    st.info("No upload. To run forecasting in the dashboard, deploy and connect to an API or upload a CSV.")
    st.stop()

# Basic preprocess: require Product and Date columns
if 'Product' not in df.columns or 'Date' not in df.columns:
    st.error("CSV must contain 'Product' and 'Date' columns.")
    st.stop()

df = df[df['Product']==sku].sort_values('Date')
st.subheader(f"History for {sku} (last 60 days)")
st.line_chart(df.set_index('Date')['Daily_Sales'].tail(60))

# Run forecasting (warning can be slow in Streamlit free environment)
if st.button("Run forecast for this SKU"):
    try:
        # Basic call using local functions (this will run in the dashboard process; for heavy work use Colab/API)
        from src.seasonality_detection import get_festival_calendar
        holidays = get_festival_calendar(df)
        model, forecast = train_and_forecast_per_sku(df, sku, horizon=horizon, regressors=['Promotion_Factor','Price_Index','Weather_Factor'], festival_calendar=holidays)
        st.subheader("Forecast (next days)")
        st.dataframe(forecast)
        st.line_chart(forecast.set_index('ds')['yhat'])
        # Compute inventory stats
        mean_daily, std_daily = compute_stats_from_history(df['Daily_Sales'].values, window_days=90)
        stats = compute_reorder_point(mean_daily, std_daily, lead_time, service_level)
        forecast_sum = float(forecast['yhat'].iloc[:horizon].sum())
        suggested = suggest_order_qty(int(df['Closing_Stock'].iloc[-1]) if 'Closing_Stock' in df.columns else 0, stats['reorder_point'], forecast_sum, stats['safety_stock'])
        st.metric("Reorder Point", f"{stats['reorder_point']:.1f}")
        st.metric("Safety stock", f"{stats['safety_stock']:.1f}")
        st.metric("Suggested order qty", f"{suggested}")
    except Exception as e:
        st.error(f"Forecast failed: {e}")
